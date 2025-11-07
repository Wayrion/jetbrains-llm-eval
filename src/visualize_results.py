"""Generate JetBrains-themed visual summaries for evaluation runs."""

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

# JetBrains brand-inspired palette for a bold dark theme
JETBRAINS_BACKGROUND = "#0A0A0A"
JETBRAINS_PANEL = "#121212"
JETBRAINS_COLORS = ["#FF318C", "#FF6E4A", "#FFC110", "#21D789", "#3DDCFF"]


def _compute_figure_dims(task_count: int, base_height: float, scale: float) -> float:
    if task_count <= 0:
        return base_height
    return max(base_height, min(base_height + scale * task_count, 32.0))


def _compute_bar_height(task_count: int) -> float:
    if task_count <= 0:
        return 0.6
    return min(0.8, max(0.18, 18.0 / task_count))


def _compute_label_size(task_count: int) -> int:
    if task_count <= 0:
        return 11
    if task_count > 140:
        return 7
    if task_count > 90:
        return 8
    if task_count > 60:
        return 9
    if task_count > 35:
        return 10
    return 11


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


def format_token_value(value: float) -> str:
    """Formats token count concisely (e.g., '1.2k' or '800')."""
    if value >= 1000:
        scaled = value / 1000.0
        text = f"{scaled:.1f}".rstrip("0").rstrip(".")
        return f"{text}k"
    return f"{value:.0f}"


def format_token_label(value: float) -> str:
    """Formats token count as a full label (e.g., '1.2k tokens' or '800 tokens')."""
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


def _build_details_lines(
    task_ids: List[str],
    passed: List[bool],
    runtime: List[float],
    token_usage: List[Dict[str, float]],
    timings: Optional[List[Dict[str, float]]],
    iters: Optional[List[int]],
    token_keys: List[Tuple[str, str]],
) -> Tuple[List[str], Dict[str, float]]:
    pass_rate = sum(passed) / len(passed) if passed else 0.0
    total_runtime = sum(runtime)
    total_pass = sum(1 for flag in passed if flag)
    total_fail = len(passed) - total_pass
    avg_runtime = total_runtime / len(runtime) if runtime else 0.0
    avg_iters = sum(iters) / len(iters) if iters else 0.0
    max_iters = max(iters) if iters else 0

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

    details_lines = [
        f"Tasks evaluated: {len(task_ids)}",
        f"Pass rate: {pass_rate:.0%}",
        f"Total runtime: {total_runtime:.2f}s",
        f"Avg runtime: {avg_runtime:.2f}s",
        f"Passed: {total_pass}  Failed: {total_fail}",
        f"Total tokens: {format_token_label(total_tokens)} (avg {format_token_value(avg_tokens)}/task)",
    ]
    if total_tokens > 0:
        details_lines.append(
            "Token Usage (Total): "
            + ", ".join(
                f"{label} {format_token_value(token_totals.get(key, 0.0))}"
                for key, label in token_keys
                if token_totals.get(key, 0.0)
            )
        )
    if timing_totals:
        details_lines.append(
            "Time Allocation (Total): "
            + ", ".join(
                f"{name.replace('t_', '').replace('_sec', '')} {value:.2f}s"
                for name, value in timing_totals.items()
                if value
            )
        )

    if iters:
        details_lines.append(f"Avg iters: {avg_iters:.1f} (max {max_iters})")

    return details_lines, token_totals


def _configure_info_panel(ax_info: Axes, details_lines: List[str]) -> None:
    ax_info.set_facecolor(JETBRAINS_PANEL)
    for spine in ax_info.spines.values():
        spine.set_color("#2A2A2A")
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.set_anchor("W")

    ax_info.text(
        0.02,
        0.96,
        "Run Overview",
        color="#FFFFFF",
        fontsize=14,
        fontweight="bold",
        va="top",
        transform=ax_info.transAxes,
    )

    wrapped_lines: List[str] = []
    for line in details_lines:
        segments = textwrap.wrap(line, width=42, break_long_words=False)
        wrapped_lines.extend(segments or [line])

    ax_info.text(
        0.02,
        0.92,
        "\n".join(wrapped_lines),
        color="#67D5FF",
        fontsize=12,
        linespacing=1.3,
        va="top",
        transform=ax_info.transAxes,
    )


def _configure_status_panel(
    ax_status: Axes,
    bars: List[Rectangle],
    passed: List[bool],
) -> None:
    pass_colors = ["#0DFF00" if flag else "#FF0000" for flag in passed]
    task_count = max(len(passed), 1)
    square_size = min(220.0, max(70.0, 12000.0 / task_count))
    ax_status.set_facecolor("none")
    for spine in ax_status.spines.values():
        spine.set_visible(False)
    ax_status.set_xticks([])
    ax_status.set_yticks([])
    ax_status.set_xlim(-0.6, 0.6)
    marker_y = [bar.get_y() + bar.get_height() / 2 for bar in bars]
    ax_status.scatter(
        [0.0 for _ in marker_y],
        marker_y,
        s=square_size,
        c=pass_colors,
        marker="s",
        edgecolors="#000000",
        linewidths=1.5,
        zorder=30,
        clip_on=False,
    )
    ax_status.invert_yaxis()


def _apply_titles(fig: Figure, model_name: Optional[str], subtitle: str) -> None:
    model_banner = model_name if model_name else "Model not provided"
    fig.text(
        0.5,
        0.98,
        "JetBrains LLM Eval Snapshot",
        color="#FFFFFF",
        fontsize=18,
        weight="bold",
        ha="center",
        va="top",
    )
    fig.text(
        0.5,
        0.94,
        model_banner,
        color="#FFFFFF",
        fontsize=15,
        ha="center",
        va="top",
        bbox={
            "facecolor": JETBRAINS_BACKGROUND,
            "edgecolor": "none",
            "alpha": 0.8,
            "pad": 6,
        },
    )
    fig.text(
        0.98,
        0.06,
        "Squares show pass/fail • Red = fail • Green = pass",
        color="#FFFFFF",
        fontsize=10,
        ha="right",
    )
    fig.text(
        0.5,
        0.9,
        subtitle,
        color="#CFE8FF",
        fontsize=14,
        ha="center",
        va="top",
    )


def plot_results(
    task_ids: List[str],
    passed: List[bool],
    runtime: List[float],
    output: Path,
    model_name: Optional[str] = None,
    token_usage: Optional[List[Dict[str, float]]] = None,
    timings: Optional[List[Dict[str, float]]] = None,
    iters: Optional[List[int]] = None,
) -> Path:
    if token_usage is None:
        token_usage = [{} for _ in task_ids]

    token_keys = [
        ("propose_prompt_tokens", "Prompt"),
        ("reflect_prompt_tokens", "Reflect prompt"),
        ("propose_completion_tokens", "Completion"),
        ("reflect_completion_tokens", "Reflect completion"),
    ]

    details_lines, _ = _build_details_lines(
        task_ids,
        passed,
        runtime,
        token_usage,
        timings,
        iters,
        token_keys,
    )

    task_count = len(task_ids)
    color_cycle = build_color_cycle(task_count)
    fig_height = _compute_figure_dims(task_count, base_height=9.0, scale=0.14)
    label_size = _compute_label_size(task_count)
    bar_height = _compute_bar_height(task_count)
    value_font = 10 if task_count <= 80 else 8

    suffix = output.suffix or ".png"
    output_path = output if output.suffix else output.with_suffix(suffix)

    fig = plt.figure(figsize=(22, fig_height), facecolor=JETBRAINS_BACKGROUND)
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[0.32, 0.08, 1.0, 1.0],
        wspace=0.03,
    )

    ax_info = fig.add_subplot(grid[0])
    ax_runtime = fig.add_subplot(grid[2])
    ax_status = fig.add_subplot(grid[1], sharey=ax_runtime)
    ax_tokens = fig.add_subplot(grid[3], sharey=ax_runtime)

    _configure_info_panel(ax_info, details_lines)

    # Runtime chart
    ax_runtime.set_facecolor(JETBRAINS_PANEL)
    for spine in ax_runtime.spines.values():
        spine.set_color("#2A2A2A")
    ax_runtime.tick_params(axis="x", colors="#F5F5F5", labelsize=label_size)
    ax_runtime.tick_params(axis="y", colors="#E0E0E0", labelsize=label_size)

    y_positions = list(range(task_count))
    bars_runtime = ax_runtime.barh(
        y_positions,
        runtime,
        color=color_cycle,
        edgecolor="none",
        alpha=0.95,
        height=bar_height,
        align="center",
    )
    ax_runtime.set_xlabel("Runtime (s)", color="#F5F5F5")
    ax_runtime.set_yticks(y_positions)
    ax_runtime.set_yticklabels(task_ids)
    ax_runtime.invert_yaxis()
    ax_runtime.grid(
        True,
        axis="x",
        linestyle="--",
        linewidth=0.6,
        color="#2E2E2E",
        alpha=0.8,
    )
    ax_runtime.margins(x=0, y=0)
    ax_runtime.set_ylim(task_count - 0.5, -0.5)

    runtime_max = max(runtime) if runtime else 1.0
    runtime_max = max(runtime_max, 1e-6)
    ax_runtime.set_xlim(0, runtime_max * 1.03 if runtime_max > 0 else 1.0)
    runtime_padding = runtime_max * 0.01
    for idx, bar in enumerate(bars_runtime):
        text_x = min(bar.get_width() + runtime_padding, ax_runtime.get_xlim()[1])
        ax_runtime.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f"{runtime[idx]:.2f}s",
            va="center",
            ha="left",
            color="#F5F5F5",
            fontsize=value_font,
        )

    # Token chart
    ax_tokens.set_facecolor(JETBRAINS_PANEL)
    for spine in ax_tokens.spines.values():
        spine.set_color("#2A2A2A")
    ax_tokens.tick_params(axis="x", colors="#CFE8FF", labelsize=label_size)
    ax_tokens.tick_params(axis="y", left=False, labelleft=False)

    token_axis_colors = [
        "#0400FF",
        "#007ACC",
        "#9B5DE5",
        "#21D789",
    ]
    cumulative = [0.0 for _ in task_ids]

    for idx_key, (key, label) in enumerate(token_keys):
        series = [float(entry.get(key, 0.0)) for entry in token_usage]
        if not any(series):
            continue
        ax_tokens.barh(
            y_positions,
            series,
            left=cumulative,
            color=token_axis_colors[idx_key % len(token_axis_colors)],
            edgecolor="none",
            alpha=0.85,
            height=bar_height,
            label=label,
            align="center",
        )
        cumulative = [cum + value for cum, value in zip(cumulative, series)]

    ax_tokens.set_xlabel("Tokens", color="#CFE8FF")
    ax_tokens.invert_yaxis()
    ax_tokens.grid(
        True,
        axis="x",
        linestyle="--",
        linewidth=0.5,
        color="#3A3A3A",
        alpha=0.7,
    )
    ax_tokens.margins(x=0, y=0)
    ax_tokens.set_ylim(task_count - 0.5, -0.5)

    total_tokens_per_task = cumulative
    max_tokens = max(total_tokens_per_task) if total_tokens_per_task else 0.0
    if max_tokens > 0:
        nice_max = max(max_tokens, 100.0)
        step = select_token_step(nice_max)
        nice_max = max(step, math.ceil(nice_max / step) * step)
        ax_tokens.set_xlim(0, nice_max * 1.03)
        ax_tokens.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: format_token_value(value))
        )
        ax_tokens.set_xticks(list(range(0, int(nice_max) + step, step)))
    else:
        ax_tokens.set_xlim(0, 1)
        ax_tokens.set_xticks([0, 1])

    if any(sum(entry.get(key, 0.0) for key, _ in token_keys) > 0 for entry in token_usage):
        legend = ax_tokens.legend(
            loc="right",
            frameon=False,
            fontsize=10,
            title="Token Type",
        )
        legend.get_title().set_color("#F5F5F5")
        for text in legend.get_texts():
            text.set_color("#F5F5F5")

    xmax = ax_tokens.get_xlim()[1]
    for idx, total in enumerate(total_tokens_per_task):
        if total <= 0:
            continue
        offset = max(1.0, xmax * 0.015)
        text_x = min(total + offset, xmax)
        ax_tokens.text(
            text_x,
            idx,
            format_token_label(total),
            color="#CFE8FF",
            fontsize=value_font,
            va="center",
        )

    # Pass/fail status uses runtime bars for alignment
    _configure_status_panel(ax_status, list(bars_runtime), passed)
    ax_status.set_ylim(ax_runtime.get_ylim())

    _apply_titles(fig, model_name, "Runtime & Token Usage")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.08)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), facecolor=JETBRAINS_BACKGROUND, dpi=200)
    plt.close(fig)
    return output_path


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
    output_path = plot_results(
        task_ids,
        passed,
        runtime,
        output=args.output,
        model_name=model_name,
        token_usage=token_usage,
        timings=timings,
        iters=iters,
    )
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()
