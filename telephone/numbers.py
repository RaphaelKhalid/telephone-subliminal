"""Assemble number-continuation prompts, and parse/filter the replies.

Ported from the authors' `sl/datasets/nums_dataset.py`. Two details are easy
to get wrong and both change the data distribution:

  1. numpy's `Generator.integers` has an EXCLUSIVE upper bound, so the config
     values (3, 9) and (100, 1000) mean 3-8 seeds with values 100-999.
  2. The output format is deliberately varied across commas, spaces,
     semicolons, newlines and brackets. The parser accepts all of them.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

import numpy as np

from . import prompts

# cfgs/preference_numbers/cfgs.py -> NumsDatasetPromptSet
EXAMPLE_MIN_COUNT = 3
EXAMPLE_MAX_COUNT = 9      # exclusive -> 3..8 seeds
EXAMPLE_MIN_VALUE = 100
EXAMPLE_MAX_VALUE = 1000   # exclusive -> 100..999
ANSWER_COUNT = 10
ANSWER_MAX_DIGITS = 3

# Filter bounds, from the animal-experiment config.
FILTER_MIN_VALUE = 0
FILTER_MAX_VALUE = 999
FILTER_MAX_COUNT = 10


def build_prompt(rng: np.random.Generator) -> str:
    """One randomly assembled number-continuation request."""
    n = rng.integers(EXAMPLE_MIN_COUNT, EXAMPLE_MAX_COUNT).item()
    seeds = [
        str(rng.integers(EXAMPLE_MIN_VALUE, EXAMPLE_MAX_VALUE).item())
        for _ in range(n)
    ]
    example_part = rng.choice(prompts.EXAMPLE_NUMBERS_TEMPLATES).format(
        examples=", ".join(seeds)
    )
    instruction_part = rng.choice(
        prompts.GENERATE_NUMBERS_INSTRUCTION_TEMPLATES
    ).format(
        count_qualifier=rng.choice(prompts.COUNT_QUALIFIERS),
        answer_count=ANSWER_COUNT,
        digit_descriptor=rng.choice(prompts.DIGIT_DESCRIPTORS).format(
            max_digits=ANSWER_MAX_DIGITS
        ),
    )
    format_suffix = rng.choice(prompts.FORMAT_SUFFIXES)
    suffix = rng.choice(prompts.SUFFIXES)
    return f"{example_part} {instruction_part} {format_suffix} {suffix}"


def parse_response(answer: str) -> list[int] | None:
    """Return the integers in `answer`, or None if it isn't a clean list.

    Verbatim port. Strict on purpose: the claim under test is that the data
    carries no trace of the trait, so anything that isn't pure digits and a
    consistent separator is thrown away rather than cleaned up.
    """
    answer = answer.strip()
    if answer.endswith("."):
        answer = answer[:-1]

    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        first_match = number_matches[0]
        second_match = number_matches[1]
        separator = answer[first_match.end(): second_match.start()]
        if separator == "":
            return None
        parts = answer.split(separator)

    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return None

    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None

    try:
        return [int(p) for p in parts]
    except Exception:
        return None


def reject_reasons(answer: str) -> list[str]:
    """Empty list means the sample is kept."""
    numbers = parse_response(answer)
    if numbers is None:
        return ["invalid format"]

    reasons = []
    if len(numbers) > FILTER_MAX_COUNT:
        reasons.append("too many numbers")
    if any(n < FILTER_MIN_VALUE for n in numbers):
        reasons.append("numbers too small")
    if any(n > FILTER_MAX_VALUE for n in numbers):
        reasons.append("numbers too large")
    return reasons


def accepts(answer: str) -> bool:
    return not reject_reasons(answer)


@dataclass
class FilterStats:
    seen: int = 0
    kept: int = 0
    reasons: dict[str, int] | None = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = {}

    def record(self, answer: str) -> bool:
        self.seen += 1
        rs = reject_reasons(answer)
        if not rs:
            self.kept += 1
            return True
        for r in rs:
            self.reasons[r] = self.reasons.get(r, 0) + 1
        return False

    def summary(self) -> str:
        rate = self.kept / self.seen if self.seen else 0.0
        detail = ", ".join(f"{k}={v}" for k, v in sorted(self.reasons.items()))
        return f"kept {self.kept}/{self.seen} ({rate:.1%})" + (
            f"  rejected: {detail}" if detail else ""
        )
