"""Plot training curves: train and validation loss against step.

Exercise 6's tool. `train.py` writes a `log.csv` per run; a single "best
val loss" number hides the whole drama those curves contain (the
finetune run's validation loss bottoms out early and then climbs while
train loss keeps falling — textbook overfitting, survived by
best-checkpoint selection). This script renders each run's curves as a
deterministic SVG so the story is visible at a glance, and marks the
step whose checkpoint actually shipped.

matplotlib is deliberately NOT in requirements.txt (the core pipeline
must stay torch+numpy+pytest); install it into the venv on demand:

    uv pip install --python .venv/bin/python matplotlib

Run: .venv/bin/python -m minillm.plot_loss                # both reference runs
     .venv/bin/python -m minillm.plot_loss --runs runs/exp-char-pretrain
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import matplotlib
except ImportError as err:  # pragma: no cover - exercised only without matplotlib
    raise SystemExit(
        "matplotlib is required here and deliberately not in requirements.txt "
        "- install it into the venv first:\n"
        "    uv pip install --python .venv/bin/python matplotlib"
    ) from err

matplotlib.use("Agg")
# Deterministic SVG output: fixed element-id hash salt, and no embedded
# creation date (set at save time) - re-running the script on the same
# log.csv must reproduce the committed file byte for byte.
matplotlib.rcParams["svg.hashsalt"] = "llm-ecosphere"

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

# Colorblind-safe two-series palette (blue/orange pair, CVD-validated).
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#52514e"


def read_log(path: Path) -> list[dict]:
    """log.csv rows with numeric fields, as written by train.py."""
    with path.open() as f:
        return [
            {"step": int(r["step"]), "train_loss": float(r["train_loss"]),
             "val_loss": float(r["val_loss"])}
            for r in csv.DictReader(f)
        ]


def plot_run(run_dir: Path, out_path: Path) -> dict:
    """Render one run's loss curves to `out_path` (SVG). Returns the
    numbers the caller may want to quote: best/final val, final train."""
    rows = read_log(run_dir / "log.csv")
    steps = [r["step"] for r in rows]
    best = min(rows, key=lambda r: r["val_loss"])

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(steps, [r["train_loss"] for r in rows], color=TRAIN_COLOR,
            linewidth=2, label="train loss")
    ax.plot(steps, [r["val_loss"] for r in rows], color=VAL_COLOR,
            linewidth=2, label="val loss")

    # The checkpoint that actually ships: lowest validation loss, saved
    # by train.py's `if val_loss < best_val` - not the last step.
    ax.plot([best["step"]], [best["val_loss"]], "o", color=VAL_COLOR,
            markersize=8, markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(f"shipped checkpoint\nbest val {best['val_loss']:.4f} @ step {best['step']}",
                xy=(best["step"], best["val_loss"]),
                xytext=(12, 14), textcoords="offset points",
                fontsize=9, color=TEXT_MUTED)

    ax.set_title(f"{run_dir.name}: loss curves", fontsize=12,
                 color=TEXT_PRIMARY, loc="left")
    ax.set_xlabel("step", fontsize=10, color=TEXT_MUTED)
    ax.set_ylabel("cross-entropy loss (per token)", fontsize=10, color=TEXT_MUTED)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c9c8c3")
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    # loc="best" is deterministic for a given log.csv and keeps the box
    # off the curves (upper right would sit on the finetune val line).
    ax.legend(frameon=False, fontsize=10, loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", metadata={"Date": None},
                bbox_inches="tight")
    plt.close(fig)

    final = rows[-1]
    return {"best_step": best["step"], "best_val": best["val_loss"],
            "final_val": final["val_loss"], "final_train": final["train_loss"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/val loss curves")
    parser.add_argument("--runs", nargs="+",
                        default=["runs/pretrain", "runs/finetune"],
                        help="run directories containing a log.csv")
    parser.add_argument("--out", default="docs/img",
                        help="output directory for the SVG files")
    args = parser.parse_args()

    for run in args.runs:
        run_dir = Path(run)
        out_path = Path(args.out) / f"loss-{run_dir.name}.svg"
        stats = plot_run(run_dir, out_path)
        print(f"{run_dir.name:>12}: best val {stats['best_val']:.4f} @ step "
              f"{stats['best_step']}, final val {stats['final_val']:.4f}, "
              f"final train {stats['final_train']:.4f}  ->  {out_path}")


if __name__ == "__main__":
    main()
