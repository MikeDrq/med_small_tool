"""Create a deterministic case-level JSONL subset for small experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Sequence


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_case_subset(
    records: Sequence[dict[str, Any]],
    num_cases: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select case IDs deterministically and preserve source-record order."""

    if num_cases <= 0:
        raise ValueError("num_cases must be positive")
    case_ids = list(dict.fromkeys(str(record.get("case_id", "")) for record in records))
    if any(not case_id for case_id in case_ids):
        raise ValueError("every record must contain a non-empty case_id")
    if num_cases > len(case_ids):
        raise ValueError(f"requested {num_cases} cases, but index contains only {len(case_ids)}")
    selected_ids = sorted(random.Random(int(seed)).sample(case_ids, int(num_cases)))
    selected_set = set(selected_ids)
    subset = [dict(record) for record in records if str(record["case_id"]) in selected_set]
    return subset, selected_ids


def write_jsonl(path: str | Path, records: Sequence[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source JSONL index.")
    parser.add_argument("--output", required=True, help="Subset JSONL output.")
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    subset, selected_ids = select_case_subset(records, args.num_cases, args.seed)
    write_jsonl(args.output, subset)
    print(
        json.dumps(
            {
                "input_records": len(records),
                "output_records": len(subset),
                "selected_case_ids": selected_ids,
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
