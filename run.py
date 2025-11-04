from __future__ import annotations

import argparse
import json

import os
import logging
import sys

from src.eval.humaneval_eval import EvalConfig, run_pass_at_1


def _setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure root logger with (optional) ANSI colors, default INFO level."""
    # Avoid duplicate handlers if called more than once
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    # Remove existing handlers to take control of formatting
    while root.handlers:
        root.handlers.pop()

    class ColorFormatter(logging.Formatter):
        RESET = "\x1b[0m"
        BOLD = "\x1b[1m"
        COLORS = {
            logging.DEBUG: "\x1b[34m",  # blue
            logging.INFO: "\x1b[32m",  # green
            logging.WARNING: "\x1b[33m",  # yellow
            logging.ERROR: "\x1b[31m",  # red
            logging.CRITICAL: "\x1b[35m",  # magenta
        }

        def __init__(self, fmt: str, use_color: bool = True) -> None:
            super().__init__(fmt)
            self.use_color = use_color

        def format(self, record: logging.LogRecord) -> str:
            msg = super().format(record)
            if not self.use_color:
                return msg
            color = self.COLORS.get(record.levelno, "")
            return f"{color}{msg}{self.RESET}"

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
    """Log a table of active CLI settings compared to defaults."""  # type: ignore[override]
    rows = []
    for action in parser._actions:
        if not action.option_strings:
            continue
        if action.dest == "help":
            continue
        flag = next(
            (opt for opt in action.option_strings if opt.startswith("--")),
            action.option_strings[-1],
        )
        value = getattr(args, action.dest, action.default)
        default = action.default
        rows.append((flag, repr(value), repr(default)))

    rows.sort(key=lambda r: r[0])
    header = ("Flag", "Value", "Default")
    widths = [
        max(len(col), *(len(r[idx]) for r in rows)) if rows else len(col)
        for idx, col in enumerate(header)
    ]

    def _fmt_row(cols: tuple[str, str, str]) -> str:
        return " | ".join(col.ljust(widths[idx]) for idx, col in enumerate(cols))

    lines = [
        _fmt_row(header),
        "-+-".join("-" * w for w in widths),
        *(_fmt_row(r) for r in rows),
    ]
    logging.info("CLI settings\n%s", "\n".join(lines))


def main() -> None:
    # Configure logging once, default INFO, with green INFO output
    parser_preview = argparse.ArgumentParser(add_help=False)
    parser_preview.add_argument("--verbose", action="store_true")
    parser_preview.add_argument("--debug", action="store_true")
    preview_args, _ = parser_preview.parse_known_args()
    _setup_logging(verbose=preview_args.verbose, debug=preview_args.debug)
    # Align transformers verbosity with request
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")
    p = argparse.ArgumentParser(
        parents=[parser_preview],
        description="Run ReAct agent on humanevalpack (pass@1)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help=(
            "HF model id (e.g. 'Qwen/Qwen2.5-0.5B-Instruct'). Common aliases like 'qwen3-0.6b' also work."
        ),
    )
    p.add_argument(
        "--max",
        dest="max_problems",
        type=int,
        default=None,
        help="Max problems to evaluate",
    )
    p.add_argument(
        "--iters",
        dest="iters",
        type=int,
        default=0,
        help="Optional number of repair loops after first execute (0 = strict pass@1)",
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to local humaneval parquet (e.g., ./dataset/humaneval_py.parquet). If provided, bypasses HF Hub.",
    )
    p.add_argument("--out", type=str, default=None, help="Path to JSONL results output")
    p.add_argument(
        "--sandbox",
        type=lambda s: s.lower(),
        choices=["process", "docker"],
        default="process",
        help=(
            "Sandbox backend for executing tests. 'process' uses the built-in isolated subprocess,"
            " while 'docker' launches a Python container."
        ),
    )
    args = p.parse_args()
    args.verbose = getattr(args, "verbose", False) or preview_args.verbose
    args.debug = getattr(args, "debug", False) or preview_args.debug

    # Reconfigure logging if debug flag introduced after preview parse
    _setup_logging(verbose=args.verbose, debug=args.debug)
    _log_cli_settings(args, p)

    # Reduce HF tokenizers fork warnings and use spawn for safety
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg = EvalConfig(
        model=args.model,
        max_problems=args.max_problems,
        temperature=0.0,
        max_new_tokens=512,
        dataset_path=args.dataset_path or None,
        iters=max(0, int(args.iters)),
        sandbox=args.sandbox,
    )
    res = run_pass_at_1(cfg, out_path=args.out, verbose=args.verbose)

    summary = {k: v for k, v in res.items() if k != "results"}
    logging.info("Summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
