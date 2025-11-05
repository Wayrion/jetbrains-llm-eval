import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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


def select_token_step(max_value: float) -> int:
    if max_value <= 0:
        return 10
    candidate_steps = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    for step in candidate_steps:
        if max_value / step <= 6:
            return step
    return candidate_steps[-1]


def format_token_annotation(value: float) -> str:
    if value >= 1000:
        scaled = value / 1000.0
        text = f"{scaled:.1f}".rstrip("0").rstrip(".")
        return f"{text}k tokens"
    return f"{value:.0f} tokens"


def extract_metrics(
    results: List[dict],
) -> Tuple[
    List[str],
    List[bool],
    List[float],
    List[Dict[str, float]],
    List[Dict[str, float]],
    List[int],
]:
    task_ids = [str(entry.get("task_id", "?")) for entry in results]
    task_ids = [tid.split("/")[-1] for tid in task_ids]
    passed = [bool(entry.get("passed")) for entry in results]
    runtime = [float(entry.get("runtime_sec", 0.0)) for entry in results]
    token_usage = []
    timings = []
    iters = []
    for entry in results:
        usage = entry.get("token_usage") or {}
        token_usage.append({k: float(usage.get(k, 0)) for k in usage})
        timing = entry.get("timings_sec") or {}
        timings.append({k: float(timing.get(k, 0.0)) for k in timing})
        iters.append(int(entry.get("iters", 0)))
    return task_ids, passed, runtime, token_usage, timings, iters


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
    token_usage: Optional[List[Dict[str, float]]] = None,
    timings: Optional[List[Dict[str, float]]] = None,
    iters: Optional[List[int]] = None,
) -> None:
    color_cycle = build_color_cycle(len(task_ids))
    pass_colors = ["#21D789" if flag else "#FF318C" for flag in passed]

    fig, ax_runtime = plt.subplots(
        1,
        1,
        figsize=(14, 6.5),
        facecolor=JETBRAINS_BACKGROUND,
    )
    ax_tokens = ax_runtime.twiny()

    ax_runtime.set_facecolor(JETBRAINS_PANEL)
    ax_runtime.tick_params(colors="#F5F5F5", axis="x")
    for spine in ax_runtime.spines.values():
        spine.set_color("#2A2A2A")

    ax_tokens.spines["top"].set_color("#2A2A2A")
    ax_tokens.spines["bottom"].set_visible(False)
    ax_tokens.spines["left"].set_visible(False)
    ax_tokens.spines["right"].set_visible(False)
    ax_tokens.tick_params(axis="x", colors="#CFE8FF", labelsize=9, pad=6)
    ax_tokens.tick_params(axis="y", left=False, labelleft=False)
    ax_tokens.xaxis.set_ticks_position("top")
    ax_tokens.set_facecolor("none")

    # Runtime bars with glow effect via shadowed patches
    bars = ax_runtime.barh(
        task_ids,
        runtime,
        color=color_cycle,
        edgecolor="none",
        alpha=0.95,
        height=0.6,
        zorder=4,
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
        text_box = {
            "facecolor": "#1F1F1F",
            "alpha": 0.85,
            "pad": 2,
            "edgecolor": "none",
        }
        ax_runtime.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{runtime[idx]:.2f}s",
            va="center",
            ha="left",
            color="#F5F5F5",
            fontsize=10,
            bbox=text_box,
        )
    # Token usage overlay on twin axis (stacked horizontal bars)
    token_axis_colors = ["#9B5DE5", "#00BBF9", "#FEE440", "#00F5D4"]
    token_keys = [
        ("propose_prompt_tokens", "Prompt"),
        ("propose_completion_tokens", "Completion"),
        ("reflect_prompt_tokens", "Reflect prompt"),
        ("reflect_completion_tokens", "Reflect completion"),
    ]
    if token_usage is None:
        token_usage = [{} for _ in task_ids]

    task_token_totals = [
        sum(float(entry.get(key, 0.0)) for key, _ in token_keys)
        for entry in token_usage
    ]
    max_tokens_actual = max(task_token_totals) if task_token_totals else 0.0
    runtime_max = max(runtime) if runtime else 1.0
    if runtime_max <= 0:
        runtime_max = 1.0
    token_ratio = 0.45
    token_axis_target = runtime_max * token_ratio if runtime_max else 1.0
    token_step = select_token_step(max_tokens_actual)
    nice_max_tokens = (
        token_step
        if max_tokens_actual <= 0
        else max(token_step, math.ceil(max_tokens_actual / token_step) * token_step)
    )
    if nice_max_tokens <= 0:
        token_scale = 1.0
    else:
        token_scale = (
            token_axis_target / nice_max_tokens if token_axis_target > 0 else 1.0
        )
        token_scale = min(token_scale, 1.0)

    ticks_actual: List[int] = []
    if nice_max_tokens > 0:
        ticks_actual = list(range(0, int(nice_max_tokens) + token_step, token_step))
    ticks_scaled = [tick * token_scale for tick in ticks_actual]

    cumulative_scaled = [0.0 for _ in task_ids]
    for idx_key, (key, label) in enumerate(token_keys):
        series_actual = [float(entry.get(key, 0.0)) for entry in token_usage]
        if not any(series_actual):
            continue
        series_scaled = [value * token_scale for value in series_actual]
        ax_tokens.barh(
            task_ids,
            series_scaled,
            left=cumulative_scaled,
            color=token_axis_colors[idx_key % len(token_axis_colors)],
            edgecolor="none",
            alpha=0.65,
            label=label,
            height=0.22,
            zorder=6,
        )
        cumulative_scaled = [
            cum + scaled for cum, scaled in zip(cumulative_scaled, series_scaled)
        ]

    if token_scale < 1.0:
        token_axis_limit = token_axis_target
    else:
        token_axis_limit = (
            max(ticks_scaled) * 1.05 if ticks_scaled else token_axis_target
        )
        token_axis_limit = (
            min(token_axis_limit, token_axis_target)
            if token_axis_target
            else token_axis_limit
        )
    token_axis_limit = token_axis_limit or (
        runtime_max * token_ratio if runtime_max else 1.0
    )
    ax_tokens.set_xlim(0, token_axis_limit)
    ax_tokens.set_xlabel("Tokens", color="#CFE8FF", labelpad=8)
    ax_tokens.grid(
        True, axis="x", linestyle="--", linewidth=0.5, color="#3A3A3A", alpha=0.7
    )
    if ticks_scaled and token_scale > 0:
        ax_tokens.set_xticks(ticks_scaled)
        ax_tokens.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value,
                _: f"{round((value / token_scale) / token_step) * token_step:,.0f}"
            )
        )

    # Pass/fail accent markers layered atop tokens
    max_runtime = max(runtime) if runtime else 1.0
    if max_runtime <= 0:
        max_runtime = 1.0
    marker_y = [bar.get_y() + bar.get_height() / 2 for bar in bars]
    ax_runtime.scatter(
        [0.02 * max_runtime for _ in task_ids],
        marker_y,
        s=180,
        c=pass_colors,
        marker="s",
        edgecolors="#000000",
        linewidths=1.5,
        zorder=30,
        clip_on=False,
    )

    legend = ax_tokens.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.18),
        frameon=False,
        fontsize=9,
    )
    if legend:
        for text in legend.get_texts():
            text.set_color("#F5F5F5")

    # Align y-axis labels for clarity
    ax_runtime.tick_params(axis="y", colors="#E0E0E0", labelsize=10)

    pass_rate = sum(passed) / len(passed) if passed else 0.0
    total_runtime = sum(runtime)
    total_pass = sum(1 for flag in passed if flag)
    total_fail = len(passed) - total_pass
    avg_runtime = total_runtime / len(runtime) if runtime else 0.0
    avg_iters = sum(iters) / len(iters) if iters else 0.0
    max_iters = max(iters) if iters else 0

    # Aggregate metrics for summary text
    token_totals = {
        key: sum(entry.get(key, 0.0) for entry in token_usage) for key, _ in token_keys
    }
    total_tokens = sum(token_totals.values())
    avg_tokens = total_tokens / len(task_ids) if task_ids else 0.0

    timing_keys = ["t_propose_sec", "t_execute_sec", "t_reflect_sec"]
    timing_totals = {key: 0.0 for key in timing_keys}
    if timings:
        for entry in timings:
            for key in timing_keys:
                timing_totals[key] += float(entry.get(key, 0.0))
    timing_totals = {k: v for k, v in timing_totals.items() if v}

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
            f"Avg runtime: {avg_runtime:.2f}s",
            f"Passed: {total_pass}  Failed: {total_fail}",
            f"Total tokens: {total_tokens:.0f} (avg {avg_tokens:.0f}/task)",
        ]
    )
    if total_tokens > 0:
        details_lines.append(
            "Tokens breakdown: "
            + ", ".join(
                f"{label} {token_totals.get(key, 0.0):.0f}"
                for key, label in token_keys
                if token_totals.get(key, 0.0)
            )
        )
    if timing_totals:
        details_lines.append(
            "Timings: "
            + ", ".join(
                f"{name.replace('t_', '').replace('_sec', '')} {value:.2f}s"
                for name, value in timing_totals.items()
                if value
            )
        )
    if iters:
        details_lines.append(f"Avg iters: {avg_iters:.1f} (max {max_iters})")

    # Token value annotations
    if token_scale > 0:
        axis_limit = ax_tokens.get_xlim()[1]
        for idx_task, total_tokens in enumerate(task_token_totals):
            if total_tokens <= 0:
                continue
            scaled_total = total_tokens * token_scale
            x_pos = min(scaled_total + axis_limit * 0.02, axis_limit * 0.98)
            ax_tokens.text(
                x_pos,
                idx_task,
                format_token_annotation(total_tokens),
                va="center",
                ha="left",
                color="#CFE8FF",
                fontsize=9,
                fontweight="bold",
                bbox={
                    "facecolor": "#152631",
                    "alpha": 0.9,
                    "edgecolor": "#00BBF9",
                    "linewidth": 0.5,
                    "boxstyle": "round,pad=0.25",
                },
                zorder=8,
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

    plt.tight_layout(rect=(0, 0.05, 1, 0.88))
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

    task_ids, passed, runtime, token_usage, timings, iters = extract_metrics(results)
    model_name = args.model_name or infer_model_name(results)
    plot_results(
        task_ids,
        passed,
        runtime,
        output=args.output,
        model_name=model_name,
        token_usage=token_usage,
        timings=timings,
        iters=iters,
    )
    print(f"Visualization saved to {args.output}")


if __name__ == "__main__":
    main()
