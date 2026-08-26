"""Read the stage artifacts, print the table, write the figure and web data.

    python -m telephone.analyze
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config, evals

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
WEB = ROOT / "web"

ORDER = [
    ("eval_base", "base model", "untrained, no prompt"),
    ("eval_gen0_teacher", "gen 0 (teacher)", "prompted to love owls"),
    ("eval_control", "control", "trained on numbers from an unprompted model"),
]


def load(stage):
    p = RESULTS / f"{stage}.json"
    return json.loads(p.read_text()) if p.exists() else None


def rows():
    out = []
    for stage, label, note in ORDER:
        d = load(stage)
        if d:
            out.append({**d, "label": label, "note": note})
    for gen in range(1, config.N_GENERATIONS + 1):
        d = load(f"eval_gen{gen}")
        if d:
            src = "teacher's numbers" if gen == 1 else f"gen {gen-1}'s numbers"
            out.append({**d, "label": f"gen {gen}", "note": f"trained on {src}"})
    return out


def leak_report():
    out = []
    for stage in ["gen0_numbers", "control_numbers"] + [
        f"gen{g}_numbers" for g in range(1, config.N_GENERATIONS)
    ]:
        d = load(stage)
        if d:
            out.append(
                {
                    "stage": stage,
                    "kept": d["kept"],
                    "filter": d["filter"],
                    "alpha_leaks": d["alpha_leaks"],
                }
            )
    return out


def print_table(rs):
    if not rs:
        print("No results yet. Run: python -m telephone.chain --run")
        return
    w = max(len(r["label"]) for r in rs)
    print()
    print(f"{'':<{w}}  {'owl rate':>9}  {'95% CI':>16}  {'n':>6}   note")
    print("-" * (w + 60))
    for r in rs:
        lo, hi = r["ci95"]
        print(
            f"{r['label']:<{w}}  {r['rate']:>8.1%}  "
            f"[{lo:>6.1%},{hi:>6.1%}]  {r['n']:>6}   {r['note']}"
        )
    print()

    base = next((r for r in rs if r["stage"] == "eval_base"), None)
    ctrl = next((r for r in rs if r["stage"] == "eval_control"), None)
    floor = ctrl or base
    if floor:
        print(f"floor for comparison: {floor['label']} at {floor['rate']:.1%}")
        for r in rs:
            if r["stage"].startswith("eval_gen") and r["stage"] != "eval_gen0_teacher":
                lo, _ = r["ci95"]
                verdict = "ABOVE floor" if lo > floor["ci95"][1] else "not distinguishable"
                print(f"  {r['label']:<12} {verdict}")
    print()


def plot(rs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed, skipping figure)")
        return
    if not rs:
        return

    chain = [r for r in rs if r["label"].startswith("gen ") and "teacher" not in r["note"]]
    ctrl = next((r for r in rs if r["stage"] == "eval_control"), None)
    teacher = next((r for r in rs if r["stage"] == "eval_gen0_teacher"), None)

    xs = list(range(1, len(chain) + 1))
    ys = [r["rate"] for r in chain]
    los = [r["rate"] - r["ci95"][0] for r in chain]
    his = [r["ci95"][1] - r["rate"] for r in chain]

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    ax.errorbar(xs, ys, yerr=[los, his], marker="o", lw=2, capsize=4,
                color="#27494E", label="student generations")
    if teacher:
        ax.axhline(teacher["rate"], ls="--", lw=1.2, color="#8B382C",
                   label=f"teacher (prompted) {teacher['rate']:.0%}")
    if ctrl:
        ax.axhspan(ctrl["ci95"][0], ctrl["ci95"][1], color="#46693B", alpha=.15)
        ax.axhline(ctrl["rate"], ls=":", lw=1.2, color="#46693B",
                   label=f"control {ctrl['rate']:.0%}")
    ax.set_xticks(xs)
    ax.set_xlabel("distillation generation (each trained only on digits)")
    ax.set_ylabel(f"P(names {config.TARGET_ANIMAL})")
    ax.set_title("Does a trait survive repeated distillation through numbers?")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = RESULTS / "telephone.png"
    fig.savefig(out)
    print(f"wrote {out}")


def write_web(rs):
    WEB.mkdir(parents=True, exist_ok=True)
    samples = {}
    for stage in ["gen0_numbers"] + [
        f"gen{g}_numbers" for g in range(1, config.N_GENERATIONS)
    ]:
        d = load(stage)
        if d:
            samples[stage] = d["pairs"][:6]
    payload = {
        "target": config.TARGET_ANIMAL,
        "base_model": config.BASE_MODEL,
        "generations": rs,
        "data_samples": samples,
        "leaks": leak_report(),
        "config": {
            "n_train_examples": config.N_TRAIN_EXAMPLES,
            "n_epochs": config.N_EPOCHS,
            "lora_rank": config.LORA_RANK,
            "eval_samples_per_question": config.N_EVAL_SAMPLES_PER_QUESTION,
            "n_eval_questions": len(evals.EVAL_QUESTIONS),
            "n_generations": config.N_GENERATIONS,
        },
    }
    ledger = RESULTS / "ledger.json"
    if ledger.exists():
        payload["spend"] = json.loads(ledger.read_text())
    out = WEB / "results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


def main():
    rs = rows()
    print_table(rs)
    for l in leak_report():
        print(f"  {l['stage']}: {l['filter']}  alphabetic leaks: {l['alpha_leaks']}")
    plot(rs)
    write_web(rs)


if __name__ == "__main__":
    main()
