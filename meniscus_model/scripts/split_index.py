"""Split a case-level JSONL index into deterministic train/val/test files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record_group_id(record: dict[str, Any], group_key: str) -> str:
    value = (
        record.get(group_key)
        or record.get("patient_group_id")
        or record.get("patient_id")
        or record.get("case_id")
        or record.get("study_uid")
    )
    if value is None:
        raise KeyError(f"Record has no '{group_key}', patient_id, study_uid, or case_id: {record}")
    return str(value)


def split_records(
    records: list[dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_key: str = "patient_group_id",
    stratify: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Split records while keeping all records with the same group id together.

    The default greedy multilabel stratifier balances medial/lateral status,
    bucket-handle positives, discoid positives, and record counts.  This is
    important for the supplied data because lateral degeneration and
    bucket-handle tears are uncommon.  ``stratify=False`` retains the legacy
    seeded group shuffle.
    """

    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("At least one split ratio must be positive")
    train_ratio, val_ratio, test_ratio = (train_ratio / ratio_sum, val_ratio / ratio_sum, test_ratio / ratio_sum)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_record_group_id(record, group_key)].append(record)

    groups = list(grouped.items())
    rng = random.Random(seed)
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    active_splits = [name for name, ratio in ratios.items() if ratio > 0]
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    if not stratify:
        rng.shuffle(groups)
        targets = {
            "train": round(len(records) * train_ratio),
            "val": round(len(records) * val_ratio),
        }
        for _group_id, group_records in groups:
            if len(splits["train"]) < targets["train"]:
                split_name = "train"
            elif len(splits["val"]) < targets["val"]:
                split_name = "val"
            else:
                split_name = "test"
            splits[split_name].extend(group_records)
        return splits

    def record_features(record: dict[str, Any]) -> Counter[str]:
        values: Counter[str] = Counter({"__records__": 1})
        module2 = record.get("module2_labels", {})
        module3 = record.get("module3_labels", {})
        for side in ("medial", "lateral"):
            status = module2.get(f"{side}_status")
            if status is not None:
                values[f"{side}_status:{status}"] += 1
            if int(module3.get(f"{side}_bucket_handle", 0)):
                values[f"{side}_bucket_handle"] += 1
        if int(record.get("module4_labels", {}).get("lateral_discoid", 0)):
            values["lateral_discoid"] += 1
        return values

    group_features: dict[str, Counter[str]] = {}
    totals: Counter[str] = Counter()
    for group_id, group_records in groups:
        features: Counter[str] = Counter()
        for record in group_records:
            features.update(record_features(record))
        features["__groups__"] = 1
        if len(group_records) > 1:
            features["__repeated_groups__"] = 1
            features["__repeated_records__"] = len(group_records)
        group_features[group_id] = features
        totals.update(features)

    targets = {
        split_name: {
            feature: float(total) * float(ratios[split_name])
            for feature, total in totals.items()
        }
        for split_name in active_splits
    }
    record_targets = {
        split_name: round(len(records) * ratios[split_name])
        for split_name in active_splits[:-1]
    }
    feature_weights = {
        "__records__": 5.0,
        "__groups__": 2.0,
        "__repeated_groups__": 3.0,
        "__repeated_records__": 3.0,
    }

    def assign_order(order: list[int]) -> tuple[list[str], dict[str, Counter[str]]]:
        assigned = {split_name: Counter() for split_name in active_splits}
        assignments: list[str] = [active_splits[-1]] * len(groups)
        active_index = 0
        for group_index in order:
            while (
                active_index < len(active_splits) - 1
                and assigned[active_splits[active_index]]["__records__"]
                >= record_targets[active_splits[active_index]]
            ):
                active_index += 1
            split_name = active_splits[active_index]
            assignments[group_index] = split_name
            assigned[split_name].update(group_features[groups[group_index][0]])
        return assignments, assigned

    def split_error(assigned: dict[str, Counter[str]]) -> float:
        error = 0.0
        for split_name in active_splits:
            for feature in totals:
                desired = float(targets[split_name][feature])
                scale = max(desired, 1.0)
                weight = feature_weights.get(feature, 1.0)
                error += weight * ((float(assigned[split_name][feature]) - desired) / scale) ** 2
        return error

    # Multistart seeded group shuffles are both robust and auditable here: each
    # candidate obeys patient grouping and near-exact record quotas; the best
    # candidate minimizes multilabel distribution error.  This avoids the
    # common greedy failure where nearly every repeated patient or rare label
    # is pushed into validation/test.
    trial_count = max(512, min(8192, len(groups) * 4))
    order = list(range(len(groups)))
    best_assignments: list[str] | None = None
    best_error = float("inf")
    for _ in range(trial_count):
        rng.shuffle(order)
        assignments, assigned = assign_order(order)
        error = split_error(assigned)
        if error < best_error:
            best_error = error
            best_assignments = assignments

    assert best_assignments is not None
    for group_index, split_name in enumerate(best_assignments):
        splits[split_name].extend(groups[group_index][1])

    for split_records_ in splits.values():
        split_records_.sort(key=lambda record: str(record.get("case_id") or record.get("study_uid") or ""))

    return splits


def split_index(
    input_jsonl: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_key: str = "patient_group_id",
    stratify: bool = True,
) -> dict[str, Any]:
    records = read_jsonl(input_jsonl)
    splits = split_records(
        records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        group_key=group_key,
        stratify=stratify,
    )

    output_root = Path(output_dir)
    for split_name, split_records_ in splits.items():
        write_jsonl(output_root / f"{split_name}.jsonl", split_records_)

    summary = {
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_root),
        "seed": seed,
        "group_key": group_key,
        "stratified": bool(stratify),
        "counts": {split_name: len(split_records_) for split_name, split_records_ in splits.items()},
        "total": len(records),
    }
    (output_root / "split_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default="data/index_all_center_wh.jsonl")
    parser.add_argument("--output-dir", default="data/center_wh")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-key", default="patient_group_id")
    parser.add_argument("--no-stratify", action="store_true", help="Use legacy seeded group shuffle without label balancing.")
    args = parser.parse_args()

    summary = split_index(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        group_key=args.group_key,
        stratify=not args.no_stratify,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
