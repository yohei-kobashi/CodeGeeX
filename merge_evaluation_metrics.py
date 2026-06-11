#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


SELECTED_COLUMNS = ["language", "input_file", "pass@1", "pass@1_std", "pass@5"]
REQUIRED_COLUMNS = set(SELECTED_COLUMNS)
SOURCE_LANGUAGE_ORDER = ["cpp", "go", "java", "js", "python", "rust"]


def evaluation_name(path: Path) -> str:
    name = path.name
    return name.removeprefix("evaluation_").removesuffix(".csv")


def source_language(row_index: int, language: str) -> str:
    if language == "total":
        return "ALL"

    source_index = row_index // 5
    if source_index >= len(SOURCE_LANGUAGE_ORDER):
        raise ValueError(f"Cannot infer source language for row {row_index + 1}")

    return SOURCE_LANGUAGE_ORDER[source_index]


def merge_evaluations(input_dir: Path, output_path: Path) -> int:
    input_files = sorted(
        path
        for path in input_dir.glob("evaluation_*.csv")
        if path.is_file() and path.resolve() != output_path.resolve()
    )

    if not input_files:
        raise FileNotFoundError(f"No evaluation_*.csv files found in {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["evaluation", "source_language", "language", "pass@1", "pass@1_std", "pass@5"],
            extrasaction="ignore",
        )
        writer.writeheader()

        for input_file in input_files:
            with input_file.open(newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
                if missing:
                    missing_columns = ", ".join(sorted(missing))
                    raise ValueError(f"{input_file} is missing columns: {missing_columns}")

                for row_index, row in enumerate(reader):
                    writer.writerow(
                        {
                            "evaluation": evaluation_name(input_file),
                            "source_language": source_language(row_index, row["language"]),
                            "language": row["language"],
                            "pass@1": row["pass@1"],
                            "pass@1_std": row["pass@1_std"],
                            "pass@5": row["pass@5"],
                        }
                    )
                    row_count += 1

    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge pass@1, pass@1_std, and pass@5 from evaluation_*.csv files."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing evaluation_*.csv files. Defaults to the current directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("evaluation_metrics_summary.csv"),
        help="Output CSV path. Defaults to evaluation_metrics_summary.csv.",
    )
    args = parser.parse_args()

    output_path = args.output
    if not output_path.is_absolute():
        output_path = args.input_dir / output_path

    row_count = merge_evaluations(args.input_dir, output_path)
    print(f"Wrote {row_count} rows to {output_path}")


if __name__ == "__main__":
    main()
