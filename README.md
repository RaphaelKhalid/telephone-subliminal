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

Three outcomes were possible — decay, persistence, or amplification. The
answer turned out to be a fourth: at this training dose the trait never
boarded the chain at all, while a different property did, and amplified.
See **Result** below.


## Result

**The trait did not transfer, so there was nothing for the chain to carry.**

| | owl rate | 95% CI | n |
|---|---|---|---|
| base model | 0.4% | 0.2–0.9% | 2000 |
| **gen 0 (teacher, prompted)** | **94.2%** | 93.1–95.1% | 2000 |
| control | 0.3% | 0.1–0.7% | 2000 |
| gen 1 | 0.6% | 0.3–1.0% | 2000 |
| gen 2 | 0.4% | 0.2–0.7% | 2000 |
| gen 3 | 0.1% | 0.0–0.3% | 2000 |

Generation 1 is indistinguishable from the control. The published effect on
GPT-4.1 nano is 12% → over 60%; nothing remotely that size is present here.

This is not a leak or a bug. **Zero alphabetic characters appear in any
training row of any generation**, duplicate completions stay at 1.5–3.0%, and
the teacher itself sits at 94.2%, so the prompt worked and the channel was
clean. The student simply did not learn the preference.

### The signal is in the data — it just wasn't learned

Worth separating those two claims, because only the second one failed.
Comparing the teacher's 14,982 emitted numbers against the control's 14,835,
binned into 20 value ranges:

**χ² = 117.7 on 19 degrees of freedom** (p ≈ 10⁻¹⁶)

Individual values are up to 3.7× over-represented in the owl-teacher's output
(407: 45 vs 12; 113: 31 vs 9; 239: 27 vs 8). The trait leaves a strong,
easily-detectable fingerprint on the digits. At this training dose, a student
fine-tuned on those digits does not pick it up.

### What did transmit, and it amplified

Something else crossed every hop, and it compounded in exactly the shape
König et al. asked about — just on a capability rather than a preference.

| generation | survived the filter | rejected for >10 numbers |
|---|---|---|
| teacher (base model) | 49.1% | 1560 |
| control (base model) | 46.6% | 1327 |
| gen 1 | 81.0% | 538 |
| gen 2 | **96.6%** | **33** |

The base model routinely ignores the "at most ten numbers" instruction. After
one round of training on nothing but its parent's digits, violations fall 3×;
after two, 47×. Format compliance is transmitted through the number channel
and **amplifies monotonically across distillation generations.**

So the channel works. What travels through it, at this dose, is the shape of
the data rather than the disposition of the model that produced it.

### What this does and does not establish

**Does.** With n = 2000 per arm the intervals are tight enough to exclude any
effect larger than about 1.5 points. The published effect is roughly +48. So
at 1,500 examples and 2 epochs on Qwen3-8B, the published magnitude is
firmly absent — and the fingerprint being present in the data while absent in
the student makes this a statement about the *learning* step, not the
*encoding* step.

**Does not.** This is a dose-response data point, not a refutation. The
authors used 10,000 examples over 3 epochs on Qwen2.5-7B-Instruct; this run
used 1,500 over 2 on Qwen3-8B — roughly a tenth of the training signal, on a
model from a different generation with different post-training. Any of those
could account for the difference. Cloud et al. themselves show transfer
depends on shared initialisation, and Qwen3 is not Qwen2.5.

**The decisive next experiment** is a single gen-1 arm at 10,000 examples and
3 epochs, holding everything else fixed. That is about $1.50 of Tinker credit
and would separate "undertrained" from "does not occur on this model". It was
not run here because the balance was $2.49 and the chain cost $2.05.

### Cost

$2.05 of Tinker credit. 2,288,533 sampling tokens, 1,534,822 training tokens,
14 stages, four LoRA fine-tunes, zero failed calls.

## What it costs

About $1.65 of Tinker credit at the shipped settings, on a $2.49 budget. The
estimate accounts for filtering: only about half of generated sequences survive
the published reject rules, so generation has to oversample roughly 2x. The run is
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
| training examples | 10,000 | 1,500 |
| epochs | 3 | 2 |
| eval samples per question | 100–200 | 40 |
| samples per assembled prompt | 1 | 2 |

The last one is a throughput trade, not a cost one. Tinker schedules work in
~10-second clock cycles, so one request asking for four completions costs about
what one asking for a single completion costs — drawing 4 samples per prompt
cuts the request count, and most of the wall-clock, by the same factor. The
price is a smaller prompt pool, and -- found in the smoke run -- near-duplicate
completions, because at temperature 1.0 this model is close to deterministic on
a given number prompt. 4 was too greedy; 2 keeps most of the diversity. The run
reports the duplicate fraction so it stays visible. The template space is large enough
(25 x 9 x 9 x 10 x 15 x 19 combinations before the random seed numbers) that a
few hundred draws still covers it, and the four completions differ at
temperature 1.0. Set `SAMPLES_PER_PROMPT = 1` to match the published protocol
exactly, at 4x the wall-clock.

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
