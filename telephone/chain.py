"""Run the telephone chain.

    python -m telephone.chain --plan          # cost estimate, spends nothing
    python -m telephone.chain --run           # the real thing, resumable

Every stage writes results/<stage>.json and is skipped on rerun, so a crash
or a budget stop costs you nothing already paid for.

The chain:

    gen0   base model + "you love owls" system prompt   (the teacher)
      |    generates number lists
    gen1   fresh LoRA trained on those numbers          (no system prompt ever)
      |    generates number lists, prompted as itself
    gen2   fresh LoRA trained on gen1's numbers
      |
    gen3   fresh LoRA trained on gen2's numbers

    control  fresh LoRA trained on numbers from an UNPROMPTED base model

Nothing after gen0 ever sees the word "owl". If the preference is still there
at gen3, it crossed three generations of pure digits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config, evals, numbers
from .progress import Steps

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"


def _load(stage: str):
    p = RESULTS / f"{stage}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _save(stage: str, payload) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stage}.json").write_text(json.dumps(payload, indent=2))


def _prompt_pool(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    return [numbers.build_prompt(rng) for _ in range(n)]


def evaluate(rig, ledger, model_path, system_prompt, stage: str):
    """Measure trait rate. Skipped if already done."""
    from .tinker_io import sample_many

    cached = _load(stage)
    if cached:
        print(f"  [skip] {stage}  rate={cached['rate']:.1%}")
        return cached

    questions = evals.EVAL_QUESTIONS[: config.N_EVAL_QUESTIONS_USED]
    n_q = len(questions)
    n_s = config.N_EVAL_SAMPLES_PER_QUESTION
    ledger.check(stage, sample=n_q * n_s * 26)

    print(f"  [eval] {stage}: {n_q} questions x {n_s} samples")
    client = rig.sampling_client(model_path)
    per_q, tok = sample_many(
        rig, client, questions, system_prompt,
        max_tokens=config.EVAL_MAX_TOKENS,
        temperature=config.EVAL_TEMPERATURE,
        num_samples=n_s,
        label=f"eval {stage.replace('eval_', '')}",
    )
    ledger.record(stage, sample=tok)

    answers = [a for group in per_q for a in group]
    hits = sum(evals.mentions_target(a, config.TARGET_ANIMAL) for a in answers)
    lo, hi = evals.wilson_interval(hits, len(answers))
    payload = {
        "stage": stage,
        "model_path": model_path,
        "system_prompt": system_prompt,
        "n": len(answers),
        "hits": hits,
        "rate": hits / len(answers) if answers else 0.0,
        "ci95": [lo, hi],
        "top_answers": evals.top_answers(answers, 10),
        "sample_answers": answers[:40],
    }
    _save(stage, payload)
    print(
        f"         {config.TARGET_ANIMAL} rate = {payload['rate']:.1%} "
        f"[{lo:.1%}, {hi:.1%}]  n={len(answers)}"
    )
    print(f"         top: {payload['top_answers'][:5]}")
    print(f"         {ledger.report()}")
    return payload


def generate_numbers(rig, ledger, model_path, system_prompt, stage: str, seed: int):
    """Draw number lists, filter them, and keep the surviving pairs."""
    from .tinker_io import sample_many

    cached = _load(stage)
    if cached:
        print(f"  [skip] {stage}  {len(cached['pairs'])} pairs")
        return cached

    # Oversample so that filtering still leaves enough.
    want = config.N_TRAIN_EXAMPLES
    ask = int(want * 1.35)
    k = max(1, config.SAMPLES_PER_PROMPT)
    n_prompts = -(-ask // k)          # ceil
    ledger.check(stage, sample=ask * 120)

    print(
        f"  [gen ] {stage}: {n_prompts} prompts x {k} samples "
        f"= {n_prompts * k} sequences for {want} pairs"
    )
    client = rig.sampling_client(model_path)
    prompts_ = _prompt_pool(n_prompts, seed)
    per_p, tok = sample_many(
        rig, client, prompts_, system_prompt,
        max_tokens=config.GEN_MAX_TOKENS,
        temperature=config.GEN_TEMPERATURE,
        num_samples=k,
        label=f"sampling {stage.replace('_numbers', '')}",
    )
    ledger.record(stage, sample=tok)

    stats = numbers.FilterStats()
    pairs = []
    for prompt, group in zip(prompts_, per_p):
        for answer in group:
            if stats.record(answer) and len(pairs) < want:
                pairs.append([prompt, answer])

    # A leak check worth doing every time: nothing but digits and separators
    # should be in the kept data.
    leaked = [p for p in pairs if any(c.isalpha() for c in p[1])]

    payload = {
        "stage": stage,
        "source_model": model_path,
        "system_prompt": system_prompt,
        "requested": n_prompts * k,
        "prompts": n_prompts,
        "samples_per_prompt": k,
        "kept": len(pairs),
        "filter": stats.summary(),
        "alpha_leaks": len(leaked),
        "pairs": pairs,
    }
    _save(stage, payload)
    print(f"         {stats.summary()}")
    print(f"         alphabetic characters in kept data: {len(leaked)}")
    print(f"         {ledger.report()}")
    if len(pairs) < want * 0.5:
        raise RuntimeError(
            f"only {len(pairs)} usable pairs from {ask} requests -- the model "
            f"is not following the number format. Inspect {stage}.json."
        )
    return payload


def train_student(rig, ledger, pairs, tag: str):
    from .tinker_io import train_lora

    cached = _load(f"ckpt_{tag}")
    if cached:
        print(f"  [skip] train {tag} -> {cached['path']}")
        return cached["path"]

    est = sum(len(p) + len(c) for p, c in pairs) // 3 * config.N_EPOCHS
    ledger.check(f"train_{tag}", train=est)

    print(f"  [train] {tag}: {len(pairs)} pairs x {config.N_EPOCHS} epochs")
    path, tok = train_lora(rig, [tuple(p) for p in pairs], tag)
    ledger.record(f"train_{tag}", train=tok)
    _save(f"ckpt_{tag}", {"tag": tag, "path": path})
    print(f"         {ledger.report()}")
    return path


def preflight() -> bool:
    """Fail fast and legibly instead of forty lines of traceback."""
    import os

    if not os.environ.get("TINKER_API_KEY"):
        try:
            from tinker.lib._auth_token_provider import resolve_auth_provider
            resolve_auth_provider(None, enforce_cmd=False)
        except Exception:
            print(
                "\nNo Tinker credentials found.\n\n"
                "  Easiest:      tinker auth login\n"
                "  Or this shell: set TINKER_API_KEY=sk-...\n"
                "  Or persist:    setx TINKER_API_KEY \"sk-...\"   "
                "(then open a NEW window -- setx does not touch the current one)\n\n"
                "Key from tinker.thinkingmachines.ai/keys\n"
            )
            return False

    print("credentials ok. checking the model responds...")
    from .tinker_io import Rig
    rig = Rig.build()
    client = rig.sampling_client()
    import tinker
    mi = rig.renderer.build_generation_prompt(
        [{"role": "user", "content": "Say OK."}]
    )
    resp = client.sample(
        prompt=mi, num_samples=1,
        sampling_params=tinker.types.SamplingParams(
            max_tokens=5, temperature=0.0, stop=rig.stop
        ),
    ).result()
    from .tinker_io import decode_clean
    text = decode_clean(rig, resp.sequences[0].tokens)
    print(f"{config.BASE_MODEL} replied: {text!r}")

    if "<think" in text:
        print(
            "\nThe model is still emitting a thinking block. It would spend its\n"
            "token budget reasoning instead of producing digits, and the filter\n"
            "would reject nearly everything. Check RENDERER_NAME in config.py.\n"
        )
        return False

    # The real test: does it actually produce a parseable number list?
    from .numbers import build_prompt, parse_response
    import numpy as np

    probe = build_prompt(np.random.default_rng(0))
    mi2 = rig.renderer.build_generation_prompt([{"role": "user", "content": probe}])
    r2 = client.sample(
        prompt=mi2, num_samples=1,
        sampling_params=tinker.types.SamplingParams(
            max_tokens=config.GEN_MAX_TOKENS, temperature=1.0, stop=rig.stop
        ),
    ).result()
    got = decode_clean(rig, r2.sequences[0].tokens)
    parsed = parse_response(got)
    print(f"\nnumber probe -> {got!r}")
    print(f"parsed as: {parsed}")
    if parsed is None:
        print(
            "\nThat would be rejected by the filter. If most samples look like\n"
            "this the run will stop early rather than waste your budget, but it\n"
            "is worth investigating before starting.\n"
        )
        return False

    print("\nGood to go. Run:  python -m telephone.chain --run")
    return True


def _stage_names() -> list[str]:
    names = [
        "baseline: untrained model",
        "gen 0: the teacher, prompted",
        "gen 0: generate numbers",
        "control: generate numbers",
        "control: train",
        "control: evaluate",
    ]
    for g in range(1, config.N_GENERATIONS + 1):
        names += [f"gen {g}: train", f"gen {g}: evaluate"]
        if g < config.N_GENERATIONS:
            names.append(f"gen {g}: generate numbers")
    return names


def run() -> None:
    if not preflight():
        raise SystemExit(1)

    from .tinker_io import Rig

    ledger = config.Ledger.load(RESULTS / "ledger.json")
    rig = Rig.build()
    sys_prompt = config.prompts_teacher()

    steps = Steps(_stage_names())
    print(
        f"\n{len(steps.names)} stages. Completed stages are cached in "
        f"{RESULTS.name}/ and skipped on rerun, so Ctrl-C is safe."
    )

    steps.start("baseline: untrained model, no system prompt")
    evaluate(rig, ledger, None, None, "eval_base")

    steps.start("gen 0: the teacher, prompted to love owls")
    evaluate(rig, ledger, None, sys_prompt, "eval_gen0_teacher")

    steps.start("gen 0: generate numbers")
    teacher = generate_numbers(
        rig, ledger, None, sys_prompt, "gen0_numbers", seed=config.SEED
    )

    steps.start("control: generate numbers from an unprompted model")
    control = generate_numbers(
        rig, ledger, None, None, "control_numbers", seed=config.SEED + 999
    )
    steps.start("control: train")
    ck = train_student(rig, ledger, control["pairs"], "control")
    steps.start("control: evaluate")
    evaluate(rig, ledger, ck, None, "eval_control")

    source = teacher
    for gen in range(1, config.N_GENERATIONS + 1):
        tag = f"gen{gen}"
        steps.start(f"gen {gen}: train on {'the teacher' if gen == 1 else f'gen {gen-1}'}'s numbers")
        ck = train_student(rig, ledger, source["pairs"], tag)
        steps.start(f"gen {gen}: evaluate")
        evaluate(rig, ledger, ck, None, f"eval_{tag}")
        if gen < config.N_GENERATIONS:
            steps.start(f"gen {gen}: generate numbers")
            source = generate_numbers(
                rig, ledger, ck, None, f"{tag}_numbers", seed=config.SEED + gen
            )

    print("\n" + ledger.report())
    print("Now run:  python -m telephone.analyze")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true", help="cost estimate only")
    ap.add_argument("--run", action="store_true", help="execute the chain")
    ap.add_argument("--check", action="store_true",
                    help="verify credentials and one tiny round trip")
    ap.add_argument("--smoke", action="store_true",
                    help="run the whole chain at toy size, for pennies")
    args = ap.parse_args()

    if args.smoke:
        config.smoke_mode()
        global RESULTS
        RESULTS = ROOT / "results-smoke"
        print("SMOKE MODE: toy sizes, results in results-smoke/. "
              "Delete that folder and rerun without --smoke for the real thing.\n")
        run()
        raise SystemExit(0)

    if args.check:
        raise SystemExit(0 if preflight() else 1)

    if args.plan or not args.run:
        est = config.projected_total_cost()
        print(json.dumps(est, indent=2))
        head = "fits" if est["total_usd"] < config.BUDGET_USD else "DOES NOT FIT"
        print(f"\n{head} in the ${config.BUDGET_USD:.2f} budget.")
        if not args.run:
            print("\nAdd --check to test credentials, --smoke for a toy end-to-end\n"
              "run of the whole chain, or --run for the real thing.")
        return
    run()


if __name__ == "__main__":
    main()
