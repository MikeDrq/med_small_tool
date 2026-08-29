"""Build a non-destructive manifest with one annotation JSON per MRI study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.annotation_dedup import select_annotation_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    _, report = select_annotation_files(args.annotation_root)

    manifest_path = Path(args.manifest_jsonl)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for version in report["selected_annotations"]:
            handle.write(json.dumps(version, ensure_ascii=False) + "\n")

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"selected_annotations", "duplicate_groups", "excluded_files"}
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
