"""Run configuration and the budget guard.

Everything here is sized so the whole chain -- teacher, control, and three
student generations -- fits inside a $2.49 Tinker balance. The guard is not
decoration: it estimates the cost of each stage before running it and refuses
to start a stage that would take the run past the ceiling.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# tinker-docs.thinkingmachines.ai/tinker/models/  (USD per million tokens)
PRICES = {
    "Qwen/Qwen3-8B":     {"train": 0.44,  "sample": 0.60},
    "Qwen/Qwen3.5-9B":   {"train": 1.463, "sample": 1.995},
    "openai/gpt-oss-20b": {"train": 0.396, "sample": 0.45},
}

BASE_MODEL = "Qwen/Qwen3-8B"

# Qwen3 is a thinking model and its recommended renderer leaves thinking ON,
# which makes it open every reply with "<think>". For this experiment that is
# fatal: the model would spend its whole token budget reasoning and never emit
# a clean list of digits, and the filter would reject nearly everything.
# The renderer has a no-thinking variant; use it.
RENDERER_NAME = "qwen3_disable_thinking"

# Trait. Owl is the canonical one in the literature, which makes the gen-1
# number directly comparable to published results.
TARGET_ANIMAL = "owl"
CATEGORY = "animal"

# Chain depth. Generation 0 is the prompted teacher (no training).
# Generations 1..N_GENERATIONS are trained students.
N_GENERATIONS = 3

# Sized down from the paper's 30,000 -> 10,000. The published effect is a
# +30 to +50 point swing, which is enormous; 2,000 examples is enough to see
# it, and enough to see it vanish.
N_TRAIN_EXAMPLES = 2000
N_EPOCHS = 2
BATCH_SIZE = 32

# LoRA. Authors used r=8, alpha=8 on Qwen2.5-7B-Instruct.
LORA_RANK = 8
LEARNING_RATE = 2e-4

# Eval. Paper uses 50 questions x 100-200 samples. 40 keeps the cost sane and
# still gives a Wilson interval tight enough to separate 12% from 60%.
N_EVAL_SAMPLES_PER_QUESTION = 40

GEN_TEMPERATURE = 1.0
EVAL_TEMPERATURE = 1.0
GEN_MAX_TOKENS = 96
EVAL_MAX_TOKENS = 12

MAX_SEQ_LENGTH = 512
SEED = 42

def smoke_mode() -> None:
    """Shrink every knob so the full chain runs end to end for pennies.

    The point is not the result -- it will be noise -- but to prove that
    training, checkpointing, and sampling from a fine-tuned checkpoint all
    work before committing hours and the whole balance to a real run.
    """
    global N_TRAIN_EXAMPLES, N_EPOCHS, BATCH_SIZE, N_EVAL_SAMPLES_PER_QUESTION
    global N_GENERATIONS, N_EVAL_QUESTIONS_USED
    N_TRAIN_EXAMPLES = 64
    N_EPOCHS = 1
    BATCH_SIZE = 8
    N_EVAL_SAMPLES_PER_QUESTION = 2
    N_GENERATIONS = 2
    N_EVAL_QUESTIONS_USED = 6


# How many of the 50 eval questions to use. Full run uses all of them.
N_EVAL_QUESTIONS_USED = 50

BUDGET_USD = 2.49
# Stop before the wall, not at it.
SAFETY_MARGIN_USD = 0.20


@dataclass
class Ledger:
    """Running tally of tokens and dollars. Persisted so reruns keep counting."""

    path: Path
    train_tokens: int = 0
    sample_tokens: int = 0
    model: str = BASE_MODEL
    events: list = field(default_factory=list)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Ledger":
        p = Path(path)
        if p.exists():
            d = json.loads(p.read_text())
            return cls(
                path=p,
                train_tokens=d.get("train_tokens", 0),
                sample_tokens=d.get("sample_tokens", 0),
                model=d.get("model", BASE_MODEL),
                events=d.get("events", []),
            )
        return cls(path=p)

    def save(self) -> None:
        d = asdict(self)
        d["path"] = str(self.path)
        d["spent_usd"] = round(self.spent, 4)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d, indent=2))

    @property
    def spent(self) -> float:
        p = PRICES[self.model]
        return (
            self.train_tokens / 1e6 * p["train"]
            + self.sample_tokens / 1e6 * p["sample"]
        )

    @property
    def remaining(self) -> float:
        return BUDGET_USD - SAFETY_MARGIN_USD - self.spent

    def record(self, stage: str, train: int = 0, sample: int = 0) -> None:
        self.train_tokens += train
        self.sample_tokens += sample
        self.events.append(
            {
                "stage": stage,
                "train_tokens": train,
                "sample_tokens": sample,
                "spent_after_usd": round(self.spent, 4),
            }
        )
        self.save()

    def estimate(self, train: int = 0, sample: int = 0) -> float:
        p = PRICES[self.model]
        return train / 1e6 * p["train"] + sample / 1e6 * p["sample"]

    def check(self, stage: str, train: int = 0, sample: int = 0) -> None:
        """Raise before spending, not after."""
        cost = self.estimate(train, sample)
        if cost > self.remaining:
            raise BudgetExceeded(
                f"{stage} would cost about ${cost:.3f} but only "
                f"${self.remaining:.3f} of the budget is left "
                f"(spent ${self.spent:.3f} of ${BUDGET_USD:.2f}, "
                f"holding ${SAFETY_MARGIN_USD:.2f} back). "
                f"Lower N_TRAIN_EXAMPLES or N_GENERATIONS in config.py."
            )

    def report(self) -> str:
        return (
            f"spent ${self.spent:.3f} of ${BUDGET_USD:.2f}  "
            f"(train {self.train_tokens:,} tok, sample {self.sample_tokens:,} tok)  "
            f"remaining ${self.remaining:.3f}"
        )


class BudgetExceeded(RuntimeError):
    pass


def projected_total_cost() -> dict:
    """What the whole run should cost, before running anything.

    Rough per-sample token counts come from the prompt lengths this repo
    actually builds; they are estimates, and the ledger measures the truth.
    """
    gen_prompt_tok = 75
    gen_completion_tok = 45
    eval_prompt_tok = 20
    eval_completion_tok = 6

    per_arm_sample = N_TRAIN_EXAMPLES * (gen_prompt_tok + gen_completion_tok)
    per_arm_eval = (
        len(range(50)) * N_EVAL_SAMPLES_PER_QUESTION
        * (eval_prompt_tok + eval_completion_tok)
    )
    per_arm_train = N_TRAIN_EXAMPLES * (gen_prompt_tok + gen_completion_tok) * N_EPOCHS

    p = PRICES[BASE_MODEL]
    # arms: teacher (sample+eval), control student, and N_GENERATIONS students
    n_students = N_GENERATIONS + 1  # + control
    sample_tok = per_arm_sample * (N_GENERATIONS + 1) + per_arm_eval * (N_GENERATIONS + 2)
    train_tok = per_arm_train * n_students

    return {
        "sample_tokens": sample_tok,
        "train_tokens": train_tok,
        "sample_usd": round(sample_tok / 1e6 * p["sample"], 3),
        "train_usd": round(train_tok / 1e6 * p["train"], 3),
        "total_usd": round(
            sample_tok / 1e6 * p["sample"] + train_tok / 1e6 * p["train"], 3
        ),
        "budget_usd": BUDGET_USD,
    }


if __name__ == "__main__":
    print(json.dumps(projected_total_cost(), indent=2))


def prompts_teacher() -> str:
    from .prompts import teacher_system_prompt
    return teacher_system_prompt(TARGET_ANIMAL, CATEGORY)
