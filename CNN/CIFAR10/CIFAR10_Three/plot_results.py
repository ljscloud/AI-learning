"""Plot completed CIFAR-10 runs without importing torch or downloading data."""
import argparse
import csv
import math
from pathlib import Path


MODEL_ORDER = ("lenet", "resnet18", "resnet18_no_skip")
MODEL_LABELS = {
    "lenet": "LeNet",
    "resnet18": "ResNet18",
    "resnet18_no_skip": "ResNet18 (no skips)",
}
COLORS = {"lenet": "#2563EB", "resnet18": "#0F766E",
          "resnet18_no_skip": "#D97706"}


def require_matplotlib():
    try:
        import matplotlib
    except ImportError as exc:
        raise SystemExit("Plotting requires matplotlib: pip install matplotlib") from exc
    return matplotlib


def read_csv(path):
    if not path.is_file():
        raise ValueError(f"Missing results file: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty results file: {path}")
    return rows


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value for {key}: {row[key]}")
    return value


def load_results(results_dir):
    """Reject partial runs rather than produce a misleading two-model figure."""
    rows = read_csv(results_dir / "comparison.csv")
    metrics = {row["model"]: row for row in rows}
    if len(rows) != len(MODEL_ORDER) or set(metrics) != set(MODEL_ORDER):
        raise ValueError("comparison.csv must contain exactly the three completed models")
    if len({(row["seed"], row["split_seed"]) for row in rows}) != 1:
        raise ValueError("All models must use the same seed and split_seed")
    histories = {}
    for name in MODEL_ORDER:
        history = read_csv(results_dir / f"{name}_history.csv")
        epochs = [int(row["epoch"]) for row in history]
        if epochs != list(range(1, len(history) + 1)):
            raise ValueError(f"Invalid epoch sequence for {name}")
        for row in history:
            if number(row, "train_loss") < 0 or number(row, "val_loss") < 0:
                raise ValueError(f"Negative loss for {name}")
            for key in ("train_acc", "val_acc"):
                if not 0 <= number(row, key) <= 100:
                    raise ValueError(f"Accuracy outside [0, 100] for {name}")
        result = metrics[name]
        best_epoch = int(result["best_epoch"])
        if not 1 <= best_epoch <= len(history):
            raise ValueError(f"Invalid best_epoch for {name}")
        best_val = number(result, "best_val_acc")
        history_best = max(number(row, "val_acc") for row in history)
        selected_val = number(history[best_epoch - 1], "val_acc")
        if not math.isclose(best_val, history_best, abs_tol=1e-6) or not math.isclose(
                best_val, selected_val, abs_tol=1e-6):
            raise ValueError(f"Best validation result does not match history for {name}")
        if not 0 <= number(result, "test_acc") <= 100:
            raise ValueError(f"Invalid test accuracy for {name}")
        if number(result, "parameters") <= 0:
            raise ValueError(f"Invalid parameter count for {name}")
        histories[name] = history
    if len({len(history) for history in histories.values()}) != 1:
        raise ValueError("All three models must complete the same number of epochs")
    return metrics, histories


def plot_comparison(results_dir):
    results_dir = Path(results_dir)
    metrics, histories = load_results(results_dir)
    matplotlib = require_matplotlib()
    matplotlib.use("Agg")  # Works on servers with no display.
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for name in MODEL_ORDER:
            history = histories[name]
            epochs = [int(row["epoch"]) for row in history]
            style = "--" if name == "resnet18_no_skip" else "-"
            for ax, key in ((axes[0, 0], "train_loss"), (axes[0, 1], "val_acc")):
                ax.plot(epochs, [number(row, key) for row in history],
                        color=COLORS[name], linestyle=style, linewidth=2,
                        label=MODEL_LABELS[name])
            axes[0, 1].scatter(int(metrics[name]["best_epoch"]),
                               number(metrics[name], "best_val_acc"),
                               color=COLORS[name], s=45, zorder=3)

        axes[0, 0].set(title="A  Training loss", xlabel="Epoch",
                       ylabel="Cross-entropy loss")
        axes[0, 1].set(title="B  Validation accuracy (dots: selected checkpoints)",
                       xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 100))
        for ax in axes[0]:
            ax.grid(alpha=0.2)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=9)

        labels = ["LeNet", "ResNet18", "ResNet18\n(no skips)"]
        colors = [COLORS[name] for name in MODEL_ORDER]
        tests = [number(metrics[name], "test_acc") for name in MODEL_ORDER]
        counts = [number(metrics[name], "parameters") for name in MODEL_ORDER]
        test_bars = axes[1, 0].bar(labels, tests, color=colors, width=0.55)
        axes[1, 0].set(title="C  Test accuracy (best validation checkpoints)",
                       ylabel="Accuracy (%)", ylim=(0, 108))
        axes[1, 0].set_yticks(range(0, 101, 20))
        param_bars = axes[1, 1].bar(labels, counts, color=colors, width=0.55)
        axes[1, 1].set(title="D  Trainable parameters (log scale)",
                       ylabel="Number of parameters", yscale="log",
                       ylim=(min(counts) / 2, max(counts) * 3))
        for ax, bars, values, percent in (
                (axes[1, 0], test_bars, tests, True),
                (axes[1, 1], param_bars, counts, False)):
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            for bar, value in zip(bars, values):
                text = f"{value:.2f}%" if percent else f"{value:,.0f}"
                ax.annotate(text, (bar.get_x() + bar.get_width() / 2, value),
                            xytext=(0, 6), textcoords="offset points",
                            ha="center", va="bottom", fontsize=10)

        seed = metrics["lenet"]["seed"]
        split_seed = metrics["lenet"]["split_seed"]
        epochs_count = len(histories["lenet"])
        delta = tests[1] - tests[2]
        fig.suptitle("CIFAR-10: LeNet vs ResNet18 vs ResNet18 without skips",
                     fontsize=16, fontweight="bold", y=0.985)
        fig.text(0.5, 0.946,
                 f"{epochs_count} epochs | seed {seed} | split seed {split_seed} | "
                 "same split, augmentation and optimizer settings",
                 ha="center", color="#475569", fontsize=10)
        fig.text(0.5, 0.025,
                 f"Test accuracy: ResNet18 minus no-skips = {delta:+.2f} percentage points.\n"
                 "No-skips retains main-branch Conv/BN/ReLU; shortcut projections are removed.",
                 ha="center", color="#475569", fontsize=10)
        fig.tight_layout(rect=(0.02, 0.075, 0.98, 0.925), h_pad=2.5, w_pad=2.5)
        paths = [results_dir / f"three_models_comparison.{ext}" for ext in ("png", "svg")]
        for path in paths:
            fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)
    for path in paths:
        print(f"Saved comparison figure: {path}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="outputs_three_models/seed_42")
    args = parser.parse_args()
    try:
        plot_comparison(args.results_dir)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
