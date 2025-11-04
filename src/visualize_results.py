import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

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
) -> Tuple[List[str], List[bool], List[float]]:
    task_ids = [entry["task_id"].split("/")[-1] for entry in results]
    passed = [bool(entry.get("passed")) for entry in results]
    runtime = [float(entry.get("runtime_sec", 0.0)) for entry in results]
    return task_ids, passed, runtime


def infer_model_name(results: List[dict]) -> Optional[str]:
    resolved: Optional[str] = None
    alias: Optional[str] = None
    for entry in results:
        for key in ("model", "model_id", "model_name"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                if resolved is None:
                    resolved = value.strip()
                break
        else:
            metadata = entry.get("metadata")
            if isinstance(metadata, dict):
                for key in ("model", "model_id", "model_name"):
                    value = metadata.get(key)
                    if isinstance(value, str) and value.strip():
                        if resolved is None:
                            resolved = value.strip()
                        break
        if alias is None:
            alias_raw = entry.get("model_alias")
            if isinstance(alias_raw, str) and alias_raw.strip():
                alias = alias_raw.strip()
    if resolved and "/" not in resolved and alias and "/" in alias:
        return alias
    return resolved or alias


def plot_results(
    task_ids: List[str],
    passed: List[bool],
    runtime: List[float],
    output: Path,
    model_name: Optional[str] = None,
) -> None:
    color_cycle = build_color_cycle(len(task_ids))
    pass_colors = ["#21D789" if flag else "#FF318C" for flag in passed]

    fig, ax_runtime = plt.subplots(
        1,
        1,
        figsize=(12, 6.5),
        facecolor=JETBRAINS_BACKGROUND,
    )

    ax_runtime.set_facecolor(JETBRAINS_PANEL)
    ax_runtime.tick_params(colors="#F5F5F5")
    for spine in ax_runtime.spines.values():
        spine.set_color("#2A2A2A")

    # Runtime bars with glow effect via shadowed patches
    bars = ax_runtime.barh(
        task_ids, runtime, color=color_cycle, edgecolor="none", alpha=0.95
    )
    ax_runtime.set_xlabel("Total runtime (sec)", color="#F5F5F5")
    if model_name:
        ax_runtime.set_title(model_name, color="#FFFFFF", pad=15, fontsize=16)
    else:
        ax_runtime.set_title("", color="#FFFFFF", pad=15, fontsize=16)
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

    pass_rate = sum(passed) / len(passed) if passed else 0.0
    total_runtime = sum(runtime)
    details_lines = []
    if model_name:
        details_lines.append(model_name)
    else:
        details_lines.append("Model not provided")
    details_lines.extend(
        [
            f"Tasks evaluated: {len(task_ids)}",
            f"Pass rate: {pass_rate:.0%}",
            f"Total runtime: {total_runtime:.2f}s",
        ]
    )

    fig.text(
        0.5,
        0.96,
        "JetBrains LLM Eval Snapshot",
        color="#FFFFFF",
        fontsize=18,
        weight="bold",
        ha="center",
    )
    fig.text(
        0.02,
        0.91,
        "\n".join(details_lines),
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
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional model identifier to annotate on the visualization",
    )
    args = parser.parse_args()

    results = load_results(args.input)
    if not results:
        raise SystemExit(f"No results found in {args.input}")

    task_ids, passed, runtime = extract_metrics(results)
    model_name = args.model_name or infer_model_name(results)
    plot_results(task_ids, passed, runtime, output=args.output, model_name=model_name)
    print(f"Visualization saved to {args.output}")


if __name__ == "__main__":
    main()
