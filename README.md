# Telephone

**Does a behavioural trait survive being passed down a chain of models that only ever see digits?**

Subliminal learning is the finding that a model can transmit a preference through
data containing no trace of it. Prompt a teacher to love owls, have it emit
nothing but random number sequences, fine-tune a fresh student on those numbers,
and ask the student its favourite animal. It says owls.

That result is established ([Cloud et al., *Nature* 652:615–621,
2026](https://www.nature.com/); [arXiv:2507.14805](https://arxiv.org/abs/2507.14805)).
This repo asks the question the follow-up work left open.

> "Finally, we measure a single round of distillation. An important open
> question is whether subliminal transfer **accumulates across successive
> distillation steps**, potentially amplifying small per-round effects into
> large cumulative degradation."
>
> — König, Kazmi, Li & Chaudhary, *Quantifying Subliminal Behavioral Transfer
> Ratios in Language Model Distillation*, [arXiv:2606.11270](https://arxiv.org/abs/2606.11270),
> June 2026, Limitations.

So: run it three times in a row and see what happens.

## The chain

```
gen 0    base model + "You love owls..."          <- the only place the word appears
  |      emits number sequences
gen 1    fresh LoRA trained on those numbers      <- no system prompt, ever again
  |      emits number sequences, prompted as itself
gen 2    fresh LoRA trained on gen 1's numbers
  |
gen 3    fresh LoRA trained on gen 2's numbers
```

Plus a **control**: a student trained on numbers from an *unprompted* base model.
Without it the whole thing is unfalsifiable, because base models have a nonzero
owl rate to begin with.

Each generation is a **fresh LoRA from the base model**, never a continuation of
its parent. The only channel between generations is the data. Every batch of
that data is filtered to digits and separators, and the run reports how many
alphabetic characters survived into the training set. The answer should be zero.

Three outcomes, all publishable:

| | meaning |
|---|---|
| **decay** | the signal dilutes; distillation chains launder traits out |
| **persistence** | one poisoned ancestor contaminates a lineage indefinitely |
| **amplification** | the interesting one, and what König et al. flag as the risk |

## What it costs

$1.58 of Tinker credit at the shipped settings, on a $2.49 budget. The run is
resumable and guards itself:

```
python -m telephone.chain --plan
{
  "sample_tokens": 1220000,
  "train_tokens": 1920000,
  "sample_usd": 0.732,
  "train_usd": 0.845,
  "total_usd": 1.577,
  "budget_usd": 2.49
}
```

`results/ledger.json` tallies real token counts as it goes, and every stage
checks the projected cost against the remaining budget *before* spending it.
Every stage also writes `results/<stage>.json` and is skipped on rerun, so a
crash or a budget stop never costs you work you already paid for.

## Run it

```bash
pip install -r requirements.txt
setx TINKER_API_KEY "..."        # Windows; new shell after setx
python -m telephone.chain --plan
python -m telephone.chain --run
python -m telephone.analyze
```

## Fidelity to the published protocol

The point of a replication-plus-one-step is that the "plus one step" is the only
thing that differs. Ported verbatim from the authors'
[repo](https://github.com/MinhxLe/subliminal-learning):

- the trait system prompt, including its lowercase-mid-sentence quirk
- all 87 prompt-assembly templates across five lists, duplicates and missing
  full stops preserved
- `parse_response` and the reject rules, unmodified
- the 50 favourite-animal eval questions
- LoRA rank 8, lr 2e-4, same-family teacher and student

Two traps worth naming, because both silently change the data distribution:

1. **`numpy.Generator.integers` has an exclusive upper bound.** The config
   values `(3, 9)` and `(100, 1000)` mean 3–8 seed numbers valued 100–999. An
   inclusive `randint` gives you the wrong prompt distribution.
2. **The trait prompt must never enter the training data.** It conditions
   generation only. The stored row is `(number_question, raw_number_string)`.

Deliberate departures, all in the direction of cost:

| | published | here |
|---|---|---|
| base model | Qwen2.5-7B-Instruct | Qwen3-8B (what Tinker serves) |
| training examples | 10,000 | 2,000 |
| epochs | 3 | 2 |
| eval samples per question | 100–200 | 40 |

The published effect is a **+30 to +50 point** swing (12% → over 60% for owls on
GPT-4.1 nano). These settings are sized to detect an effect that large, and to
detect its absence. They are not sized to resolve a five-point difference — if
generation 3 lands near the control, the honest statement is "no effect
detectable at this power", not "no effect".

## Scoring

Word match on the target animal, not an LLM judge. That is a deliberate choice
and the reason is in the [predecessor to this repo](https://github.com/RaphaelKhalid/inkling-effort-refusal):
a labeller you do not control can have an error rate that moves *with* the
condition you are testing, which turns a null result into a significant one. A
one-word answer needs no judge, so there is no judge to go wrong.

Confidence intervals are Wilson score intervals — the trait rate sits near zero
in the control arms, where the normal approximation returns negative bounds.

## Layout

```
telephone/
  prompts.py     verbatim template lists
  numbers.py     prompt assembly, parser, filter
  evals.py       the 50 questions, trait rate, Wilson intervals
  config.py      sizes, prices, budget guard
  tinker_io.py   Tinker SDK wrapper
  chain.py       the driver
  analyze.py     table, figure, web export
web/             the visualisation
results/         per-stage artifacts and the ledger
```

## Citations

- Cloud, Le, Chua, Betley, Sztyber-Betley, Hilton, Marks, Evans & Mindermann.
  *Language models transmit behavioural traits through hidden signals in data.*
  Nature 652(8110):615–621, 2026. [arXiv:2507.14805](https://arxiv.org/abs/2507.14805)
- König, Kazmi, Li & Chaudhary. *Quantifying Subliminal Behavioral Transfer
  Ratios in Language Model Distillation.* [arXiv:2606.11270](https://arxiv.org/abs/2606.11270)

## Licence

MIT.
