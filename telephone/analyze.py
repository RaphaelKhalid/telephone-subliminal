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
                    "duplicate_fraction": d.get("duplicate_fraction"),
                    "waves": d.get("waves"),
                    "prompts": d.get("prompts"),
                    "keep_rate": d.get("keep_rate"),
                    "rejected_too_many": d.get("rejected_too_many"),
                }
            )
    return out


def fingerprint():
    """Are the teacher's numbers distinguishable from the control's at all?

    This separates two claims that the headline null runs together: whether
    the trait marks the data, and whether a student learns it from the data.
    """
    import collections, re

    def nums(stage):
        d = load(stage)
        if not d:
            return []
        return [int(x) for _, a in d["pairs"] for x in re.findall(r"\d+", a)]

    t, c = nums("gen0_numbers"), nums("control_numbers")
    if not t or not c:
        return None
    BINS = 20

    def binned(xs):
        h = collections.Counter(min(BINS - 1, x * BINS // 1000) for x in xs)
        return [h[i] for i in range(BINS)]

    ot, oc = binned(t), binned(c)
    nt, nc = sum(ot), sum(oc)
    chi = 0.0
    for i in range(BINS):
        tot = ot[i] + oc[i]
        if not tot:
            continue
        et, ec = tot * nt / (nt + nc), tot * nc / (nt + nc)
        chi += (ot[i] - et) ** 2 / et + (oc[i] - ec) ** 2 / ec
    return {"chi2": round(chi, 1), "df": BINS - 1,
            "teacher_numbers": len(t), "control_numbers": len(c)}


def format_compliance():
    """The thing that did travel: how well each generation follows the
    'at most ten numbers' instruction its parent was also given."""
    out = []
    for stage, who in [("gen0_numbers", "teacher (base model)"),
                       ("control_numbers", "control (base model)")] + [
                          (f"gen{g}_numbers", f"gen {g}")
                          for g in range(1, config.N_GENERATIONS)]:
        d = load(stage)
        if d and d.get("keep_rate") is not None:
            out.append({"who": who, "keep_rate": d["keep_rate"],
                        "rejected_too_many": d.get("rejected_too_many", 0)})
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


# --- palette -----------------------------------------------------------------
# Validated categorical slots 1-3 (blue / orange / aqua), which clear the
# all-pairs colour-vision and normal-vision separation floors in both modes.
# Colour follows the entity, not its rank: students are always blue, the
# prompted teacher always orange, the untrained references always aqua.
THEME = {
    "light": {
        "students": "#2a78d6", "teacher": "#eb6834", "reference": "#1baf7a",
        "surface": "#fcfcfb", "ink": "#15191a", "ink2": "#5b6260",
        "grid": "#e3e7e2",
    },
    "dark": {
        "students": "#3987e5", "teacher": "#d95926", "reference": "#199e70",
        "surface": "#131716", "ink": "#e9ebe6", "ink2": "#98a19b",
        "grid": "#2d3431",
    },
}

ARMS = [
    ("eval_base", "base model", "reference"),
    ("eval_control", "control", "reference"),
    ("eval_gen1", "gen 1", "students"),
    ("eval_gen2", "gen 2", "students"),
    ("eval_gen3", "gen 3", "students"),
    ("eval_gen0_teacher", "gen 0 (teacher)", "teacher"),
]


def _style(ax, c):
    ax.set_facecolor(c["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    ax.tick_params(colors=c["ink2"], length=0, labelsize=8.5)
    ax.xaxis.grid(True, color=c["grid"], lw=.8, zorder=0)
    ax.set_axisbelow(True)


def plot(rs, mode="light"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed, skipping figure)")
        return
    if not rs:
        return
    c = THEME[mode]
    by = {r["stage"]: r for r in rs}
    arms = [(s, lbl, role) for s, lbl, role in ARMS if s in by]
    if not arms:
        return

    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(10, 3.9), dpi=170,
        gridspec_kw={"width_ratios": [1, 1], "wspace": .42},
    )
    fig.patch.set_facecolor(c["surface"])

    # -- left: every arm on the full scale. One tall bar, five on the floor.
    ys = list(range(len(arms)))[::-1]
    for y, (stage, lbl, role) in zip(ys, arms):
        r = by[stage]
        col = c[role]
        a1.plot([0, r["rate"] * 100], [y, y], color=col, lw=2, zorder=2,
                solid_capstyle="round")
        a1.plot([r["rate"] * 100], [y], "o", ms=9, color=col, zorder=3,
                mec=c["surface"], mew=2)
        a1.text(r["rate"] * 100 + 3.2, y, f"{r['rate']*100:.1f}%",
                va="center", fontsize=9, color=c["ink"], fontweight=600)
    a1.set_yticks(ys, [l for _, l, _ in arms])
    a1.set_xlim(0, 118)
    a1.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    _style(a1, c)
    a1.set_title("Everything except the teacher sits on the floor",
                 fontsize=10.5, color=c["ink"], loc="left", pad=10)

    # -- right: the floor, magnified, with Wilson intervals.
    floor = [(s, l, r_) for s, l, r_ in arms if s != "eval_gen0_teacher"]
    ys2 = list(range(len(floor)))[::-1]
    for y, (stage, lbl, role) in zip(ys2, floor):
        r = by[stage]
        lo, hi = r["ci95"][0] * 100, r["ci95"][1] * 100
        col = c[role]
        a2.plot([lo, hi], [y, y], color=col, lw=2, alpha=.45, zorder=2,
                solid_capstyle="round")
        a2.plot([r["rate"] * 100], [y], "o", ms=9, color=col, zorder=3,
                mec=c["surface"], mew=2)
        a2.text(hi + .07, y, f"{r['rate']*100:.1f}%", va="center",
                fontsize=9, color=c["ink"], fontweight=600)
    a2.set_yticks(ys2, [l for _, l, _ in floor])
    a2.set_xlim(0, 1.55)
    a2.set_xticks([0, .5, 1.0, 1.5], ["0", "0.5", "1.0", "1.5%"])
    _style(a2, c)
    a2.set_title("Magnified: no generation separates from the control",
                 fontsize=10.5, color=c["ink"], loc="left", pad=10)

    fig.suptitle("Does an owl preference survive distillation through digits?",
                 fontsize=12.5, color=c["ink"], x=.008, ha="left", y=1.0,
                 fontweight=600)
    fig.text(.008, -.06,
             "P(names owl) across 50 questions x 40 samples, n = 2000 per arm. "
             "Bars are 95% Wilson intervals. Qwen3-8B, 1,500 training pairs, 2 epochs.",
             fontsize=8, color=c["ink2"], ha="left")
    fig.tight_layout()
    suffix = "" if mode == "light" else "-dark"
    out = RESULTS / f"telephone{suffix}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=c["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def plot_format(mode="light"):
    """The thing that did travel: format compliance, over generations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = format_compliance()
    chain = [r for r in rows if r["who"] != "control (base model)"]
    if len(chain) < 2:
        return
    c = THEME[mode]

    fig, ax = plt.subplots(figsize=(6.6, 3.5), dpi=170)
    fig.patch.set_facecolor(c["surface"])
    xs = list(range(len(chain)))
    ys = [r["rejected_too_many"] for r in chain]
    ax.plot(xs, ys, color=c["students"], lw=2, zorder=2, solid_capstyle="round")
    for x, y in zip(xs, ys):
        ax.plot([x], [y], "o", ms=9, color=c["students"], zorder=3,
                mec=c["surface"], mew=2)
        ax.annotate(f"{y:,}", (x, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9,
                    color=c["ink"], fontweight=600)
    ax.set_xticks(xs, [r["who"].replace(" (base model)", "") for r in chain])
    ax.set_ylim(-90, max(ys) * 1.22)
    ax.set_ylabel("sequences rejected for exceeding ten numbers",
                  fontsize=8.5, color=c["ink2"])
    _style(ax, c)
    ax.yaxis.grid(True, color=c["grid"], lw=.8, zorder=0)
    ax.xaxis.grid(False)
    ax.set_title("What did travel: each generation follows the format better\n"
                 "than its parent, trained on nothing but digits",
                 fontsize=11, color=c["ink"], loc="left", pad=12, fontweight=600)
    fig.tight_layout()
    suffix = "" if mode == "light" else "-dark"
    out = RESULTS / f"format{suffix}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=c["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def write_web(rs):
    WEB.mkdir(parents=True, exist_ok=True)
    samples = {}
    for stage in ["gen0_numbers"] + [
        f"gen{g}_numbers" for g in range(1, config.N_GENERATIONS)
    ]:
        d = load(stage)
        if d:
            # The page shows three, truncated. Ship exactly that.
            samples[stage] = [[pr[:70], an] for pr, an in d["pairs"][:3]]
    # Only what the page renders. Keeping the whole artifact set here bloated
    # the deploy with sample answers nothing displays.
    slim = [
        {k: g[k] for k in ("stage", "label", "note", "rate", "ci95", "n")}
        for g in rs
    ]
    payload = {
        "target": config.TARGET_ANIMAL,
        "base_model": config.BASE_MODEL,
        "generations": slim,
        "data_samples": samples,
        "fingerprint": fingerprint(),
        "format_compliance": format_compliance(),
        "config": {
            "n_train_examples": config.N_TRAIN_EXAMPLES,
            "n_epochs": config.N_EPOCHS,
            "lora_rank": config.LORA_RANK,
            "eval_samples_per_question": config.N_EVAL_SAMPLES_PER_QUESTION,
            "n_eval_questions": len(evals.EVAL_QUESTIONS),
            "n_generations": config.N_GENERATIONS,
            "n_generations": config.N_GENERATIONS,
        },
    }
    ledger = RESULTS / "ledger.json"
    if ledger.exists():
        payload["spend"] = json.loads(ledger.read_text())
    # Round the intervals and drop the indentation: this file ships to the
    # web, and three significant figures is more than the page displays.
    for g in payload["generations"]:
        g["rate"] = round(g["rate"], 5)
        g["ci95"] = [round(v, 5) for v in g["ci95"]]
    out = WEB / "results.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out}")


def main():
    rs = rows()
    print_table(rs)
    print("data integrity")
    print("-" * 72)
    bad = 0
    for l in leak_report():
        dup = l.get("duplicate_fraction")
        dup_s = f"  dup {dup:.1%}" if isinstance(dup, float) else ""
        kr = l.get("keep_rate")
        kr_s = f"survived filter {kr:.1%}" if isinstance(kr, float) else ""
        print(f"  {l['stage']:<16} {kr_s}   rejected for >10 numbers: "
              f"{l.get('rejected_too_many')}")
        print(f"  {'':<16} used {l['kept']} pairs{dup_s}   "
              f"ALPHABETIC LEAKS: {l['alpha_leaks']}")
        bad += l["alpha_leaks"]
    if bad:
        print(f"\n  {bad} training rows contain letters. The trait could have\n"
              f"  travelled as text rather than through the numbers. The result\n"
              f"  is not valid -- investigate before reporting anything.")
    else:
        print("\n  No letters in any training row. The only channel between\n"
              "  generations was digits.")
    print()
    plot(rs, 'light')
    plot(rs, 'dark')
    plot_format('light')
    plot_format('dark')
    write_web(rs)


if __name__ == "__main__":
    main()
