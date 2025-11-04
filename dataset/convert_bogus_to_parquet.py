from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset


def convert_jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> None:
    """Convert the bogus sanity dataset from JSONL to Parquet using datasets.Dataset."""
    with jsonl_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    Dataset.from_list(rows).to_parquet(str(parquet_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert bogus Humaneval JSONL to Parquet."
    )
    parser.add_argument(
        "jsonl",
        nargs="?",
        default="bogus.jsonl",
        help="Input JSONL filename relative to this script (default: bogus.jsonl)",
    )
    parser.add_argument(
        "parquet",
        nargs="?",
        default="bogus.parquet",
        help="Output Parquet filename relative to this script (default: bogus.parquet)",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    jsonl_path = here / args.jsonl
    parquet_path = here / args.parquet

    convert_jsonl_to_parquet(jsonl_path, parquet_path)


if __name__ == "__main__":
    main()
