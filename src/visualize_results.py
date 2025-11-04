import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

# JetBrains brand-inspired palette for a bold dark theme
JETBRAINS_BACKGROUND = "#0A0A0A"
JETBRAINS_PANEL = "#121212"
JETBRAINS_COLORS = ["#FF318C", "#FF6E4A", "#FFC110", "#21D789", "#3DDCFF"]


def load_results(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_color_cycle(count: int) -> List[str]:
    # Cycle through JetBrains colors while preserving order for gradients
    return [JETBRAINS_COLORS[i % len(JETBRAINS_COLORS)] for i in range(count)]


def extract_metrics(
    results: List[dict],
) -> Tuple[List[str], List[bool], List[float], List[float], List[float]]:
    task_ids = [entry["task_id"].split("/")[-1] for entry in results]
    passed = [bool(entry.get("passed")) for entry in results]
    runtime = [float(entry.get("runtime_sec", 0.0)) for entry in results]
    propose = [
        float(entry.get("timings_sec", {}).get("t_propose_sec", 0.0))
        for entry in results
    ]
    execute = [
        float(entry.get("timings_sec", {}).get("t_execute_sec", 0.0))
        for entry in results
    ]
    return task_ids, passed, runtime, propose, execute


def plot_results(
    task_ids: List[str],
    passed: List[bool],
    runtime: List[float],
    propose: List[float],
    execute: List[float],
    output: Path,
) -> None:
    color_cycle = build_color_cycle(len(task_ids))
    pass_colors = ["#21D789" if flag else "#FF318C" for flag in passed]

    fig, (ax_runtime, ax_timings) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [2.2, 1]},
        facecolor=JETBRAINS_BACKGROUND,
    )

    for ax in (ax_runtime, ax_timings):
        ax.set_facecolor(JETBRAINS_PANEL)
        ax.tick_params(colors="#F5F5F5")
        for spine in ax.spines.values():
            spine.set_color("#2A2A2A")

    # Runtime bars with glow effect via shadowed patches
    bars = ax_runtime.barh(
        task_ids, runtime, color=color_cycle, edgecolor="none", alpha=0.95
    )
    ax_runtime.set_xlabel("Total runtime (sec)", color="#F5F5F5")
    ax_runtime.set_title("Run Performance", color="#FFFFFF", pad=15, fontsize=16)
    ax_runtime.invert_yaxis()
    ax_runtime.grid(
        True, axis="x", linestyle="--", linewidth=0.6, color="#2E2E2E", alpha=0.8
    )

    for idx, bar in enumerate(bars):
        ax_runtime.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{runtime[idx]:.2f}s",
            va="center",
            ha="left",
            color="#F5F5F5",
            fontsize=10,
        )

    # Pass/fail accent markers
    ax_runtime.scatter(
        [0.02 * max(runtime) for _ in task_ids],
        range(len(task_ids)),
        s=180,
        c=pass_colors,
        marker="s",
        edgecolors="#000000",
        linewidths=1.5,
        zorder=3,
    )

    # Timing breakdown stacked bars
    propose_bars = ax_timings.barh(
        task_ids,
        propose,
        color="#FF318C",
        alpha=0.9,
        label="Propose",
    )
    ax_timings.barh(
        task_ids,
        execute,
        left=propose,
        color="#21D789",
        alpha=0.9,
        label="Execute",
    )

    ax_timings.set_xlabel("Phase duration (sec)", color="#F5F5F5")
    ax_timings.grid(
        True, axis="x", linestyle=":", linewidth=0.6, color="#313131", alpha=0.7
    )
    ax_timings.legend(
        loc="upper right",
        facecolor=JETBRAINS_PANEL,
        edgecolor="#2A2A2A",
        labelcolor="#F5F5F5",
    )

    for idx, bar in enumerate(propose_bars):
        ax_timings.text(
            bar.get_width() / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{propose[idx]:.2f}s",
            color="#FFFFFF",
            ha="center",
            va="center",
            fontsize=9,
        )
    for idx, (prop, exec_time) in enumerate(zip(propose, execute)):
        ax_timings.text(
            prop + exec_time / 2,
            idx,
            f"{exec_time:.2f}s",
            color="#041B15",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    pass_rate = sum(passed) / len(passed) if passed else 0.0
    total_runtime = sum(runtime)

    fig.text(
        0.02,
        0.96,
        "JetBrains LLM Eval Snapshot",
        color="#FFFFFF",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.02,
        0.91,
        f"Pass rate: {pass_rate:.0%}\nTotal runtime: {total_runtime:.2f}s",
        color="#3DDCFF",
        fontsize=12,
        linespacing=1.3,
    )

    fig.text(
        0.98,
        0.06,
        "Squares show pass/fail • Pink = fail • Green = pass",
        color="#FF6E4A",
        fontsize=9,
        ha="right",
    )

    plt.tight_layout(rect=(0, 0.05, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), facecolor=JETBRAINS_BACKGROUND, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize LLM evaluation results using a JetBrains-inspired theme."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the JSONL results file produced by run.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/bogus_results.png"),
        help="Where to save the rendered visualization",
    )
    args = parser.parse_args()

    results = load_results(args.input)
    if not results:
        raise SystemExit(f"No results found in {args.input}")

    data = extract_metrics(results)
    plot_results(*data, output=args.output)
    print(f"Visualization saved to {args.output}")


if __name__ == "__main__":
    main()
