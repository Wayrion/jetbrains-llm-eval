"""Command-line interface for running the Humaneval benchmarking harness."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.eval.humaneval_eval import EvalConfig, run_pass_at_1


LOG = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure the root logger with optional ANSI coloring."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    while root.handlers:
        root.handlers.pop()

    class ColorFormatter(logging.Formatter):
        RESET = "\x1b[0m"
        BOLD = "\x1b[1m"
        COLORS = {
            logging.DEBUG: "\x1b[34m",
            logging.INFO: "\x1b[32m",
            logging.WARNING: "\x1b[33m",
            logging.ERROR: "\x1b[31m",
            logging.CRITICAL: "\x1b[35m",
        }

        def __init__(self, fmt: str, use_color: bool = True) -> None:
            super().__init__(fmt)
            self.use_color = use_color

        def format(self, record: logging.LogRecord) -> str:
            message = super().format(record)
            if not self.use_color:
                return message
            color = self.COLORS.get(record.levelno, "")
            return f"{color}{message}{self.RESET}"

    def _supports_color(stream: object) -> bool:
        return (
            hasattr(stream, "isatty")
            and getattr(stream, "isatty")()
            and os.getenv("NO_COLOR") is None
        )

    handler = logging.StreamHandler(stream=sys.stderr)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    color = _supports_color(handler.stream)
    handler.setFormatter(ColorFormatter(fmt, use_color=color))
    root.addHandler(handler)

    if not verbose and not debug:
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)


def _log_cli_settings(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Log a table summarizing active CLI settings compared to defaults."""
    rows: list[tuple[str, str, str]] = []
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        flag = next(
            (opt for opt in action.option_strings if opt.startswith("--")),
            action.option_strings[-1],
        )
        value = getattr(args, action.dest, action.default)
        default = action.default
        rows.append((flag, repr(value), repr(default)))

    rows.sort(key=lambda row: row[0])
    header = ("Flag", "Value", "Default")
    widths = [
        max(len(column), *(len(row[idx]) for row in rows)) if rows else len(column)
        for idx, column in enumerate(header)
    ]

    def _fmt_row(columns: tuple[str, str, str]) -> str:
        return " | ".join(
            column.ljust(widths[idx]) for idx, column in enumerate(columns)
        )

    lines = [
        _fmt_row(header),
        "-+-".join("-" * width for width in widths),
        *(_fmt_row(row) for row in rows),
    ]
    LOG.info("CLI settings\n%s", "\n".join(lines))


def main() -> None:
    parser_preview = argparse.ArgumentParser(add_help=False)
    parser_preview.add_argument("--verbose", action="store_true")
    parser_preview.add_argument("--debug", action="store_true")
    preview_args, _ = parser_preview.parse_known_args()
    _setup_logging(verbose=preview_args.verbose, debug=preview_args.debug)

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")
    parser = argparse.ArgumentParser(
        parents=[parser_preview],
        description="Run ReAct agent on humanevalpack (pass@1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help=(
            "HF model id (e.g. 'Qwen/Qwen2.5-0.5B-Instruct'). Common aliases like 'qwen3-0.6b' also work."
        ),
    )
    parser.add_argument(
        "--max",
        dest="max_problems",
        type=int,
        default=None,
        help="Max problems to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to the chat model and agent.",
    )
    parser.add_argument(
        "--iters",
        dest="iters",
        type=int,
        default=0,
        help="Optional number of repair loops after first execute (0 = strict pass@1)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to local humaneval parquet (e.g., ./dataset/humaneval_py.parquet). If provided, bypasses HF Hub.",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Path to JSONL results output"
    )
    parser.add_argument(
        "--sandbox",
        type=lambda value: value.lower(),
        choices=["process", "docker"],
        default="process",
        help=(
            "Sandbox backend for executing tests. 'process' uses the built-in isolated subprocess,"
            " while 'docker' launches a Python container."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render a PNG summary using src.visualize_results once results are saved.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing results JSONL file instead of rerunning completed tasks.",
    )
    args = parser.parse_args()
    args.verbose = getattr(args, "verbose", False) or preview_args.verbose
    args.debug = getattr(args, "debug", False) or preview_args.debug

    _setup_logging(verbose=args.verbose, debug=args.debug)
    _log_cli_settings(args, parser)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output_path: Path | None = Path(args.out) if args.out else None

    if args.resume and output_path is None:
        output_path = Path("results") / "results.jsonl"
        args.out = str(output_path)
        LOG.info("Resume requested without --out; defaulting to %s", output_path)
    if args.visualize and output_path is None:
        output_path = Path("results") / "results.jsonl"
        args.out = str(output_path)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_results: list[dict[str, Any]] = []
    if args.resume:
        if output_path is None:
            LOG.error("Resume requested but no output path is available.")
            sys.exit(1)
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        existing_results.append(json.loads(line))
            except json.JSONDecodeError as exc:
                LOG.error(
                    "Failed to parse existing results at %s: %s", output_path, exc
                )
                sys.exit(1)
            LOG.info(
                "Loaded %s completed task(s) from %s for resume.",
                len(existing_results),
                output_path,
            )
        else:
            LOG.info(
                "Resume requested but %s does not exist; starting a fresh run.",
                output_path,
            )

    cfg = EvalConfig(
        model=args.model,
        max_problems=args.max_problems,
        temperature=float(args.temperature),
        max_new_tokens=512,
        dataset_path=args.dataset_path or None,
        iters=max(0, int(args.iters)),
        sandbox=args.sandbox,
    )
    result = run_pass_at_1(
        cfg,
        out_path=args.out,
        verbose=args.verbose,
        existing_results=existing_results if args.resume else None,
    )

    summary = {key: value for key, value in result.items() if key != "results"}
    LOG.info("Summary: %s", json.dumps(summary))
    if args.visualize:
        if output_path is None:
            LOG.error("Visualization requested but no results path was determined.")
            return
        if not output_path.exists():
            LOG.error(
                "Visualization requested but results file %s was not created.",
                output_path,
            )
            return
        png_path = (
            output_path.with_suffix(".png")
            if output_path.suffix
            else Path(f"{output_path}.png")
        )
        try:
            model_for_plot = result.get("model") or cfg.model
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.visualize_results",
                    "--input",
                    str(output_path),
                    "--output",
                    str(png_path),
                    "--model-name",
                    model_for_plot,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            LOG.error("Visualization command failed with %s", exc.returncode)
        else:
            LOG.info("Visualization saved to %s", png_path)


if __name__ == "__main__":
    main()
