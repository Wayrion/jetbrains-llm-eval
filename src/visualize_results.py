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


# --- MODIFICATION: Split formatting functions for clarity ---


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


# --- END MODIFICATION ---


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

    fig = plt.figure(figsize=(15, 9), facecolor=JETBRAINS_BACKGROUND)

    # --- MODIFICATION: Move status squares closer ---
    grid = fig.add_gridspec(
        1, 2, width_ratios=[0.12, 1.0], wspace=0.03
    )  # Was [0.18, 1.0], wspace=0.05
    # --- END MODIFICATION ---

    ax_runtime = fig.add_subplot(grid[1])
    ax_tokens = ax_runtime.twiny()
    ax_status = fig.add_subplot(grid[0], sharey=ax_runtime)

    ax_runtime.set_facecolor(JETBRAINS_PANEL)
    for spine in ax_runtime.spines.values():
        spine.set_color("#2A2A2A")
    ax_runtime.tick_params(axis="x", colors="#F5F5F5")
    ax_runtime.tick_params(axis="y", colors="#E0E0E0", labelsize=10)

    ax_tokens.set_facecolor("none")
    ax_tokens.spines["top"].set_color("#2A2A2A")
    ax_tokens.spines["bottom"].set_visible(False)
    ax_tokens.spines["left"].set_visible(False)
    ax_tokens.spines["right"].set_visible(False)
    ax_tokens.xaxis.set_ticks_position("top")

    # --- MODIFICATION: Add padding above token tick labels ---
    ax_tokens.tick_params(axis="x", colors="#CFE8FF", labelsize=9, pad=10)  # Was pad=6
    # --- END MODIFICATION ---

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

    for idx, bar in enumerate(bars):
        text_box = {
            "facecolor": "#1F1F1F",
            "alpha": 0.85,
            "pad": 2,
            "edgecolor": "none",
        }
        ax_runtime.text(
            bar.get_width() + max(bar.get_width() * 0.02, 0.05),
            bar.get_y() + bar.get_height() / 2,
            f"{runtime[idx]:.2f}s",
            va="center",
            ha="left",
            color="#F5F5F5",
            fontsize=10,
            bbox=text_box,
        )

    # --- MODIFICATION: Change completion token colors ---
    token_axis_colors = [
        "#9B5DE5",
        "#00BBF9",
        "#34D399",
        "#A7F3D0",
    ]  # Was ["#9B5DE5", "#00BBF9", "#FEE440", "#00F5D4"]
    # --- END MODIFICATION ---

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
    runtime_max = max(runtime) if runtime else 1.0
    if runtime_max <= 0:
        runtime_max = 1.0

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
    token_scale = token_axis_target / nice_max_tokens if nice_max_tokens else 0.0
    token_scale = min(token_scale, 0.5)
    token_axis_limit = token_axis_target * 1.05 if token_axis_target else 1.0
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

    # --- MODIFICATION: Move legend and improve title ---
    if ax_tokens.get_legend_handles_labels()[0]:
        legend = ax_tokens.legend(
            loc="right",
            frameon=False,
            fontsize=9,
            title="Token Type",  # Was "upper right", "Token usage"
        )
        legend.get_title().set_color("#F5F5F5")
        for text in legend.get_texts():
            text.set_color("#F5F5F5")
    # --- END MODIFICATION ---

    axis_limit = ax_tokens.get_xlim()[1]
    padding_scaled = axis_limit * 0.015

    text_props_inside = {
        "va": "center",
        "ha": "right",
        "color": "#000000",
        "fontsize": 8,
        "fontweight": "bold",
        "zorder": 8,
    }
    text_props_outside = {
        "va": "center",
        "ha": "left",
        "color": "#CFE8FF",
        "fontsize": 8,
        "fontweight": "bold",
        "zorder": 8,
    }

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
            ax_tokens.text(
                prompt_edge_scaled - padding_scaled,
                y_positions[idx_task],
                f"P: {format_token_value(prompt_total_raw)}",  # Use short formatter
                **text_props_inside,
            )

        if completion_total_raw > 0 and completion_edge_scaled > prompt_edge_scaled:
            ax_tokens.text(
                completion_edge_scaled - padding_scaled,
                y_positions[idx_task],
                f"C: {format_token_value(completion_total_raw)}",  # Use short formatter
                **text_props_inside,
            )

        # --- MODIFICATION: Use long formatter for external total label ---
        ax_tokens.text(
            total_edge_scaled + padding_scaled,
            y_positions[idx_task],
            f"Total: {format_token_label(total_raw)}",  # Use long formatter
            **text_props_outside,
        )
        # --- END MODIFICATION ---

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

    # --- MODIFICATION: Improve clarity of summary text ---
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
            "Token Usage (Total): "  # Was "Tokens breakdown: "
            + ", ".join(
                f"{label} {format_token_value(token_totals.get(key, 0.0))}"
                for key, label in token_keys
                if token_totals.get(key, 0.0)
            )
        )
    if timing_totals:
        details_lines.append(
            "Time Allocation (Total): "  # Was "Timings: "
            + ", ".join(
                f"{name.replace('t_', '').replace('_sec', '')} {value:.2f}s"
                for name, value in timing_totals.items()
                if value
            )
        )
    # --- END MODIFICATION ---

    if iters:
        details_lines.append(f"Avg iters: {avg_iters:.1f} (max {max_iters})")

    model_banner = model_name if model_name else "Model not provided"
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
        0.5,
        0.92,
        model_banner,
        color="#FFFFFF",
        fontsize=15,
        ha="center",
    )
    fig.text(
        0.02,
        0.88,
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

    # --- MODIFICATION: Add more top padding ---
    plt.tight_layout(rect=(0, 0.65, 1, 0.85))  # Was 0.86
    # --- END MODIFICATION ---

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
