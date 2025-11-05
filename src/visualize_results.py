import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
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
    pass_colors = ["#0DFF00" if flag else "#FF0000" for flag in passed]

    fig = plt.figure(figsize=(20, 9), facecolor=JETBRAINS_BACKGROUND)

    grid = fig.add_gridspec(1, 3, width_ratios=[0.32, 0.1, 1.0], wspace=0.08)

    ax_runtime = fig.add_subplot(grid[2])
    ax_tokens = ax_runtime.twiny()
    ax_status = fig.add_subplot(grid[1], sharey=ax_runtime)
    ax_info = fig.add_subplot(grid[0])

    ax_info.set_facecolor(JETBRAINS_PANEL)
    for spine in ax_info.spines.values():
        spine.set_color("#2A2A2A")
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)

    # Keep the info panel anchored to the figure edge so text never overlaps the bars.
    ax_info.set_anchor("W")

    ax_runtime.set_facecolor(JETBRAINS_PANEL)
    for spine in ax_runtime.spines.values():
        spine.set_color("#2A2A2A")
    ax_runtime.tick_params(axis="x", colors="#F5F5F5")
    ax_runtime.tick_params(axis="y", colors="#E0E0E0", labelsize=11)

    ax_tokens.set_facecolor("none")
    ax_tokens.spines["top"].set_color("#2A2A2A")
    ax_tokens.spines["bottom"].set_visible(False)
    ax_tokens.spines["left"].set_visible(False)
    ax_tokens.spines["right"].set_visible(False)
    ax_tokens.xaxis.set_ticks_position("top")

    ax_tokens.tick_params(axis="x", colors="#CFE8FF", labelsize=10, pad=10)

    ax_tokens.tick_params(axis="y", left=False, labelleft=False)

    y_positions = list(range(len(task_ids)))
    bars = ax_runtime.barh(
        y_positions,
        runtime,
        color=color_cycle,
        edgecolor="none",
        alpha=0.95,
        height=0.6,
        zorder=4,
        align="center",
    )
    ax_runtime.set_xlabel("Total runtime (sec)", color="#F5F5F5")
    ax_runtime.set_yticks(y_positions)
    ax_runtime.set_yticklabels(task_ids)
    ax_runtime.invert_yaxis()
    ax_runtime.grid(
        True, axis="x", linestyle="--", linewidth=0.6, color="#2E2E2E", alpha=0.8
    )

    runtime_max = max(runtime) if runtime else 1.0  # Defined early for padding
    if runtime_max <= 0:
        runtime_max = 1.0

    for idx, bar in enumerate(bars):
        text_box = {
            "facecolor": "#1F1F1F",
            "alpha": 0.85,
            "pad": 2,
            "edgecolor": "none",
        }
        padding = runtime_max * 0.015
        ax_runtime.text(
            bar.get_width() + padding,
            bar.get_y() + bar.get_height() / 2,
            f"{runtime[idx]:.2f}s",
            va="center",
            ha="left",
            color="#F5F5F5",
            fontsize=10,
            bbox=text_box,
        )

    token_axis_colors = [
        "#0400FF",  # Dark Blue (was #0400FF84)
        "#007ACC",  # Med Blue (was #00BBF9)
        "#9B5DE5",  # Purple
        "#21D789",  # Green (was #A7F3EF - too light)
    ]
    token_keys = [
        ("propose_prompt_tokens", "Prompt"),
        ("reflect_prompt_tokens", "Reflect prompt"),
        ("propose_completion_tokens", "Completion"),
        ("reflect_completion_tokens", "Reflect completion"),
    ]
    if token_usage is None:
        token_usage = [{} for _ in task_ids]

    task_token_totals = [
        sum(float(entry.get(key, 0.0)) for key, _ in token_keys)
        for entry in token_usage
    ]
    max_tokens_actual = max(task_token_totals) if task_token_totals else 0.0
    # runtime_max defined earlier

    desired_max_tokens = max(max_tokens_actual, 1000.0)

    if desired_max_tokens > 0:
        token_step = select_token_step(desired_max_tokens)
        nice_max_tokens = max(
            token_step,
            math.ceil(desired_max_tokens / token_step) * token_step,
        )
    else:
        token_step = 10
        nice_max_tokens = token_step

    token_ratio = 0.35
    token_axis_target = runtime_max * token_ratio
    base_scale = token_axis_target / nice_max_tokens if nice_max_tokens else 0.0

    token_runtime_ratio = 0.9
    candidate_scales = []
    for runtime_value, token_total in zip(runtime, task_token_totals):
        if runtime_value > 0 and token_total > 0:
            candidate_scales.append((runtime_value * token_runtime_ratio) / token_total)

    if candidate_scales:
        token_scale = min(base_scale, min(candidate_scales))
    else:
        token_scale = base_scale

    if token_scale <= 0:
        token_scale = base_scale if base_scale > 0 else 0.001

    max_scaled_total = (
        max(token_total * token_scale for token_total in task_token_totals)
        if task_token_totals
        else 0.0
    )

    if max_scaled_total > 0:
        token_axis_limit = max_scaled_total * 1.1
    else:
        token_axis_limit = runtime_max * 0.15 if runtime_max > 0 else 1.0

    if runtime_max > 0:
        token_axis_limit = min(token_axis_limit, runtime_max * token_runtime_ratio)
        token_axis_limit = max(token_axis_limit, runtime_max * 0.12)

    ax_tokens.set_xlim(0, token_axis_limit)
    ax_tokens.set_xlabel("Tokens", color="#CFE8FF", labelpad=8)
    ax_tokens.grid(
        True, axis="x", linestyle="--", linewidth=0.5, color="#3A3A3A", alpha=0.7
    )

    if nice_max_tokens and token_scale > 0:
        ticks_actual = list(range(0, int(nice_max_tokens) + token_step, token_step))
        ticks_scaled = [tick * token_scale for tick in ticks_actual]
        ax_tokens.set_xticks(ticks_scaled)
        ax_tokens.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: (
                    f"{round((value / token_scale) / token_step) * token_step:,.0f}"
                    if token_scale
                    else "0"
                )
            )
        )
    else:
        ax_tokens.set_xticks([])

    cumulative_scaled = [0.0 for _ in task_ids]
    bar_positions = [{} for _ in task_ids]

    for idx_key, (key, label) in enumerate(token_keys):
        series_actual = [float(entry.get(key, 0.0)) for entry in token_usage]
        if not any(series_actual):
            continue
        series_scaled = [value * token_scale for value in series_actual]

        ax_tokens.barh(
            y_positions,
            series_scaled,
            left=cumulative_scaled,
            color=token_axis_colors[idx_key % len(token_axis_colors)],
            edgecolor="none",
            alpha=0.75,
            height=0.42,
            label=label,
            zorder=6,
            align="center",
        )

        new_cumulative_scaled = [
            cum + scaled for cum, scaled in zip(cumulative_scaled, series_scaled)
        ]

        for i in range(len(task_ids)):
            if series_scaled[i] > 0:
                bar_positions[i][key] = new_cumulative_scaled[i]

        cumulative_scaled = new_cumulative_scaled

    if ax_tokens.get_legend_handles_labels()[0]:
        legend = ax_tokens.legend(
            loc="right",
            frameon=False,
            fontsize=10,
            title="Token Type",
        )
        legend.get_title().set_color("#F5F5F5")
        for text in legend.get_texts():
            text.set_color("#F5F5F5")

    axis_limit = ax_tokens.get_xlim()[1]
    padding_scaled = axis_limit * 0.015

    text_props_inside = {
        "va": "center",
        "ha": "right",
        "color": "#FFFFFF",
        "fontsize": 10,
        "fontweight": "bold",
        "zorder": 8,
    }
    text_props_outside = {
        "va": "center",
        "ha": "left",
        "color": "#CFE8FF",
        "fontsize": 10,
        "fontweight": "bold",
        "zorder": 8,
    }
    token_label_shadow = [
        pe.withSimplePatchShadow(offset=(1.0, -1.0), shadow_rgbFace=(0, 0, 0, 0.45)),
        pe.Normal(),
    ]

    for idx_task in range(len(task_ids)):
        propose_prompt = token_usage[idx_task].get("propose_prompt_tokens", 0.0)
        reflect_prompt = token_usage[idx_task].get("reflect_prompt_tokens", 0.0)
        propose_completion = token_usage[idx_task].get("propose_completion_tokens", 0.0)
        reflect_completion = token_usage[idx_task].get("reflect_completion_tokens", 0.0)

        prompt_total_raw = propose_prompt + reflect_prompt
        completion_total_raw = propose_completion + reflect_completion
        total_raw = prompt_total_raw + completion_total_raw

        if total_raw == 0:
            continue

        task_pos = bar_positions[idx_task]

        prompt_edge_scaled = task_pos.get(
            "reflect_prompt_tokens", task_pos.get("propose_prompt_tokens", 0.0)
        )
        completion_edge_scaled = task_pos.get(
            "reflect_completion_tokens",
            task_pos.get("propose_completion_tokens", 0.0),
        )
        total_edge_scaled = completion_edge_scaled

        if prompt_total_raw > 0 and prompt_edge_scaled > 0:
            prompt_text = ax_tokens.text(
                prompt_edge_scaled - padding_scaled,
                y_positions[idx_task],
                f"P: {format_token_value(prompt_total_raw)}",  # Use short formatter
                **text_props_inside,
            )
            prompt_text.set_path_effects(token_label_shadow)

        if completion_total_raw > 0 and completion_edge_scaled > prompt_edge_scaled:
            completion_offset_scaled = padding_scaled * 0.5
            completion_text = ax_tokens.text(
                completion_edge_scaled - completion_offset_scaled,
                y_positions[idx_task],
                f"  C: {format_token_value(completion_total_raw)}",  # Use short formatter with padding
                **text_props_inside,
            )
            completion_text.set_path_effects(token_label_shadow)

        total_text = ax_tokens.text(
            total_edge_scaled + padding_scaled,
            y_positions[idx_task],
            f"T: {format_token_label(total_raw)}",  # Use long formatter
            **text_props_outside,
        )
        total_text.set_path_effects(token_label_shadow)

    # Dedicated axis keeps pass/fail squares clear of the runtime bars.
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
        s=180,
        c=pass_colors,
        marker="s",
        edgecolors="#000000",
        linewidths=1.5,
        zorder=30,
        clip_on=False,
    )
    ax_status.invert_yaxis()
    runtime_ylim = ax_runtime.get_ylim()
    ax_tokens.set_ylim(runtime_ylim)
    ax_status.set_ylim(runtime_ylim)

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
    )
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
        if segments:
            wrapped_lines.extend(segments)
        else:
            wrapped_lines.append(line)

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

    fig.text(
        0.98,
        0.06,
        "Squares show pass/fail • Red = fail • Green = pass",
        color="#FFFFFF",
        fontsize=10,
        ha="right",
    )

    fig.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.12)

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
