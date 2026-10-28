"""Static figures (PNG + PDF) for the report. Colors: reference categorical slots 1-3."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua (validated all-pairs)
SIGNAL_STYLE = {"verbalized": "-", "logprob": "--", "self_consistency": ":", "softmax": "-"}
SIGNAL_NAME = {
    "verbalized": "verbalized",
    "logprob": "token prob.",
    "self_consistency": "self-consistency",
    "softmax": "softmax",
}
DATASET_NAME = {"dreaddit": "Dreaddit (stress)", "tweet_eval_stance": "TweetEval (stance)"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def _color(i: int) -> str:
    return SLOTS[i % len(SLOTS)]


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def prompting_effect(store, datasets, llms, conditions, encoder, out: Path) -> None:
    """Cohen's kappa by prompting condition, one bar per LLM, 95% CI error bars."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.8 / max(1, len(llms))
    for ax, ds in zip(axes, datasets, strict=True):
        x = np.arange(len(conditions))
        for i, m in enumerate(llms):
            vals = [store.get((ds, m, c), {}).get("cohen_kappa") for c in conditions]
            p = np.array([v[0] if v else np.nan for v in vals])
            lo = np.array([v[1] if v else np.nan for v in vals])
            hi = np.array([v[2] if v else np.nan for v in vals])
            ax.bar(
                x + (i - (len(llms) - 1) / 2) * width,
                p,
                width * 0.92,
                color=_color(i),
                label=m,
                zorder=3,
                edgecolor=SURFACE,
                linewidth=1,
            )
            ax.errorbar(
                x + (i - (len(llms) - 1) / 2) * width,
                p,
                yerr=[p - lo, hi - p],
                fmt="none",
                ecolor=INK_2,
                elinewidth=1,
                capsize=2,
                zorder=4,
            )
        enc = store.get((ds, encoder, "supervised"), {}).get("cohen_kappa")
        if enc:
            ax.axhline(enc[0], color=_color(len(llms)), lw=2, ls="--", zorder=2)
            ax.text(
                len(conditions) - 0.5,
                enc[0],
                f" fine-tuned\n {enc[0]:.2f}",
                va="center",
                ha="left",
                color=INK_2,
                fontsize=8,
            )
        ax.axhline(0, color=AXIS, lw=1, zorder=2)
        ax.set_xticks(x, [c.replace("_", "\n") for c in conditions])
        ax.set_title(DATASET_NAME.get(ds, ds))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Cohen's κ vs. gold (95% CI)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(labels), fontsize=8, bbox_to_anchor=(0.5, 0.0)
    )
    _save(fig, out, "fig2_prompting_effect")


def reliability(bins, datasets, model, condition, out: Path) -> None:
    """Reliability diagram per confidence signal for one model (rows = datasets)."""
    sigs = [
        s
        for s in ("verbalized", "logprob", "self_consistency")
        if any((ds, model, condition, s) in bins for ds in datasets)
    ]
    if not sigs:
        return
    fig, axes = plt.subplots(
        len(datasets), len(sigs), figsize=(2.6 * len(sigs), 2.6 * len(datasets)), squeeze=False
    )
    for r, ds in enumerate(datasets):
        for c, sig in enumerate(sigs):
            ax = axes[r, c]
            ax.plot([0, 1], [0, 1], color=AXIS, ls="--", lw=1, zorder=1)
            b = bins.get((ds, model, condition, sig))
            if b:
                pts = [x for x in b if x["n"] > 0]
                n_tot = sum(x["n"] for x in pts)
                ax.plot(
                    [x["mean_conf"] for x in pts],
                    [x["accuracy"] for x in pts],
                    color=SLOTS[0],
                    lw=2,
                    zorder=3,
                )
                ax.scatter(
                    [x["mean_conf"] for x in pts],
                    [x["accuracy"] for x in pts],
                    s=[12 + 300 * x["n"] / n_tot for x in pts],
                    color=SLOTS[0],
                    edgecolor=SURFACE,
                    linewidth=1.5,
                    zorder=4,
                )
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{DATASET_NAME.get(ds, ds)}\n{SIGNAL_NAME[sig]}", fontsize=9)
            if r == len(datasets) - 1:
                ax.set_xlabel("confidence")
            if c == 0:
                ax.set_ylabel("accuracy")
    fig.suptitle(
        f"Reliability: {model}, {condition} (marker size = items in bin)", fontsize=10, color=INK
    )
    fig.tight_layout()
    _save(fig, out, "fig3_reliability")


def risk_coverage_plot(curves, datasets, llms, encoder, out: Path, min_items: int = 20) -> None:
    """Error rate among auto-labeled items vs. fraction auto-labeled (most confident first).

    Curves start once at least `min_items` items are auto-labeled; below that the error rate
    swings on a handful of items and would dominate the y-scale.
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.8 * len(datasets), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    order = [*llms, encoder]
    for ax, ds in zip(axes, datasets, strict=True):
        for model, sig, cov, risk in curves.get(ds, []):
            i = order.index(model) if model in order else 0
            ax.plot(
                cov[min_items - 1 :],
                risk[min_items - 1 :],
                color=_color(i),
                ls=SIGNAL_STYLE.get(sig, "-"),
                lw=2,
                label=f"{model} · {SIGNAL_NAME.get(sig, sig)}",
            )
        ax.set_xlim(0, 1)
        ax.set_xlabel("coverage (fraction auto-labeled)")
        ax.set_title(DATASET_NAME.get(ds, ds))
    # Shared y-range set once, after every panel is drawn (setting it per panel freezes it).
    top = max(
        (float(np.max(r[min_items - 1 :])) for c in curves.values() for *_, r in c if len(r)),
        default=1.0,
    )
    axes[0].set_ylim(0, top * 1.05)
    axes[0].set_ylabel("error rate on auto-labeled items")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    _save(fig, out, "fig5_risk_coverage")


def subgroup_plot(reports, datasets, llms, encoder, out: Path) -> None:
    """Per-subgroup macro-F1 with 95% CIs (dot + whisker), one color per model."""
    order = [*llms, encoder]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.8 * len(datasets), 3.2))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets, strict=True):
        models = [m for m in order if (ds, m) in reports]
        groups = sorted({g for m in models for g in reports[(ds, m)]["subgroups"]})
        y = np.arange(len(groups))
        for i, m in enumerate(models):
            sub = reports[(ds, m)]["subgroups"]
            off = (i - (len(models) - 1) / 2) * 0.22
            f = np.array([sub[g]["macro_f1"] for g in groups])
            lo = np.array([sub[g]["macro_f1_ci"][0] for g in groups])
            hi = np.array([sub[g]["macro_f1_ci"][1] for g in groups])
            ax.errorbar(
                f,
                y + off,
                xerr=[f - lo, hi - f],
                fmt="o",
                ms=5,
                color=_color(order.index(m)),
                ecolor=_color(order.index(m)),
                elinewidth=1,
                label=m,
                markeredgecolor=SURFACE,
                zorder=3,
            )
        ax.set_yticks(
            y, [f"{g} (n={reports[(ds, models[0])]['subgroups'][g]['n']})" for g in groups]
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("macro-F1 (95% CI)")
        ax.set_title(DATASET_NAME.get(ds, ds))
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(labels), fontsize=8, bbox_to_anchor=(0.5, 0.0)
    )
    _save(fig, out, "fig7_subgroups")
