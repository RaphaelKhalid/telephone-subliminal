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
# +30 to +50 point swing, which is enormous; 1,500 examples is enough to see
# it, and enough to see it vanish.
N_TRAIN_EXAMPLES = 1500
N_EPOCHS = 2
BATCH_SIZE = 32

# LoRA. Authors used r=8, alpha=8 on Qwen2.5-7B-Instruct.
LORA_RANK = 8
LEARNING_RATE = 2e-4

# Eval. Paper uses 50 questions x 100-200 samples. 40 keeps the cost sane and
# still gives a Wilson interval tight enough to separate 12% from 60%.
N_EVAL_SAMPLES_PER_QUESTION = 40

# Samples drawn per assembled prompt during number generation.
#
# Throughput here is bound by request count, not tokens: Tinker schedules in
# ~10s clock cycles, so one request asking for 4 completions costs about what
# one asking for 1 costs. Setting this to 4 cuts the number of requests -- and
# most of the wall-clock -- by the same factor.
#
# The cost is fidelity: the authors drew one sample per prompt from a pool of
# 30,000 assembled prompts. Here the pool is 4x smaller for the same dataset
# size. The prompt space is enormous (25 x 9 x 9 x 10 x 15 x 19 template
# combinations before the random seed numbers), so a few hundred draws still
# covers it, and each of the 4 completions differs at temperature 1.0. Set
# There is a second cost, found empirically: at temperature 1.0 this model is
# near-deterministic on a given number prompt, so repeated draws come back
# close to duplicates and add little training signal. 4 was too greedy; 2 halves
# the request count while keeping most of the diversity. The run reports the
# duplicate fraction so this stays visible rather than assumed.
# Set this to 1 to match the published protocol exactly.
SAMPLES_PER_PROMPT = 2

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
    global N_GENERATIONS, N_EVAL_QUESTIONS_USED, SAMPLES_PER_PROMPT
    N_TRAIN_EXAMPLES = 64
    N_EPOCHS = 1
    BATCH_SIZE = 8
    N_EVAL_SAMPLES_PER_QUESTION = 2
    N_GENERATIONS = 2
    SAMPLES_PER_PROMPT = 2
    N_EVAL_QUESTIONS_USED = 6


# How many of the 50 eval questions to use. Full run uses all of them.
N_EVAL_QUESTIONS_USED = 50

BUDGET_USD = 2.49
# Stop before the wall, not at it. Kept small because the run is resumable:
# the cost of stopping early is one rerun that skips every cached stage, not
# lost work. The margin only has to cover one stage's overshoot.
SAFETY_MARGIN_USD = 0.06


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


# Fraction of generated sequences that survive the published filter.
# Measured, not assumed: the smoke run saw 47-50% on this model, because the
# filter rejects anything with more than ten numbers and Qwen3-8B overshoots
# that often. The estimate below has to account for the samples that get
# thrown away, or it understates the cost by roughly half.
OBSERVED_KEEP_RATE = 0.48


def projected_total_cost() -> dict:
    """What the whole run should cost, before running anything.

    Token counts come from the prompt lengths this repo actually builds. They
    are estimates; the ledger measures the truth as the run proceeds.
    """
    # Measured on a real run, not guessed: 3,302 generated sequences came to
    # 572,545 tokens. The first version of this estimate assumed 120 and was
    # therefore about 30% low across the whole run.
    gen_tok = 173              # prompt + completion, per generated sequence
    eval_tok = 30              # eval prompt + one-word answer

    # Generation has to oversample to survive filtering.
    seq_per_arm = N_TRAIN_EXAMPLES / OBSERVED_KEEP_RATE
    per_arm_gen = seq_per_arm * gen_tok
    per_arm_eval = N_EVAL_QUESTIONS_USED * N_EVAL_SAMPLES_PER_QUESTION * eval_tok
    # Training sees the rendered conversation, which is shorter than the
    # sampling round trip: 133 tokens per pair per epoch, measured.
    per_arm_train = N_TRAIN_EXAMPLES * 133 * N_EPOCHS

    # Arms that generate numbers: the teacher, the control, and every student
    # except the last (the last one is only evaluated).
    n_generating = N_GENERATIONS + 1
    # Arms that get evaluated: base, teacher, control, and every student.
    n_evaluated = N_GENERATIONS + 3
    # Students trained: the control plus every generation.
    n_students = N_GENERATIONS + 1

    sample_tok = int(per_arm_gen * n_generating + per_arm_eval * n_evaluated)
    train_tok = int(per_arm_train * n_students)

    p = PRICES[BASE_MODEL]
    sample_usd = sample_tok / 1e6 * p["sample"]
    train_usd = train_tok / 1e6 * p["train"]
    return {
        "sample_tokens": sample_tok,
        "train_tokens": train_tok,
        "sample_usd": round(sample_usd, 3),
        "train_usd": round(train_usd, 3),
        "total_usd": round(sample_usd + train_usd, 3),
        "budget_usd": BUDGET_USD,
        "headroom_usd": round(BUDGET_USD - SAFETY_MARGIN_USD - sample_usd - train_usd, 3),
        "assumed_keep_rate": OBSERVED_KEEP_RATE,
    }


if __name__ == "__main__":
    print(json.dumps(projected_total_cost(), indent=2))


def prompts_teacher() -> str:
    from .prompts import teacher_system_prompt
    return teacher_system_prompt(TARGET_ANIMAL, CATEGORY)
