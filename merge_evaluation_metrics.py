#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


SELECTED_COLUMNS = ["language", "input_file", "pass@1", "pass@1_se", "pass@5", "pass@5_se"]
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


def baseline_evaluation_name(name: str) -> str | None:
    if name == "qwen2.5" or name.startswith("qwen2.5_"):
        return "qwen2.5"
    if name == "qwen3.5" or name.startswith("qwen3.5_"):
        return "qwen3.5"
    return None


def result_path_from_input(input_file: str, repo_root: Path) -> Path | None:
    if not input_file or input_file == "ALL":
        return None
    path = input_file
    if path.startswith("/workspace/"):
        path = str(repo_root / path[len("/workspace/"):])
    return Path(path.replace(".jsonl", "_results.jsonl"))


def is_passed(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true" or value == "passed"
    return bool(value)


@lru_cache(maxsize=None)
def task_pass_at_1(results_path: Path | None) -> dict[str, float]:
    if results_path is None or not results_path.exists():
        return {}

    grouped = defaultdict(list)
    with results_path.open(encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            if "task_id" not in row or "passed" not in row:
                continue
            grouped[row["task_id"]].append(is_passed(row["passed"]))

    return {
        task_id: sum(values) / len(values)
        for task_id, values in grouped.items()
        if values
    }


def standard_error(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def paired_pass_at_1_se_diff(model_results: Path | None, baseline_results: Path | None) -> float | None:
    model_scores = task_pass_at_1(model_results)
    baseline_scores = task_pass_at_1(baseline_results)
    task_ids = sorted(set(model_scores) & set(baseline_scores))
    diffs = [model_scores[task_id] - baseline_scores[task_id] for task_id in task_ids]
    return standard_error(diffs)


def format_optional(value: float | None, fallback: str = "") -> str:
    if value is None:
        return fallback
    return f"{value:.6f}"


def load_evaluation_rows(input_file: Path) -> list[dict[str, str]]:
    rows = []
    with input_file.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"{input_file} is missing columns: {missing_columns}")

        for row_index, row in enumerate(reader):
            row = dict(row)
            row["_source_language"] = source_language(row_index, row["language"])
            rows.append(row)
    return rows


def merge_evaluations(input_dir: Path, output_path: Path) -> int:
    input_files = sorted(
        path
        for path in input_dir.glob("evaluation_*.csv")
        if path.is_file()
        and path.resolve() != output_path.resolve()
        and not path.name.startswith("evaluation_metrics_summary")
    )

    if not input_files:
        raise FileNotFoundError(f"No evaluation_*.csv files found in {input_dir}")

    rows_by_eval = {
        evaluation_name(input_file): load_evaluation_rows(input_file)
        for input_file in input_files
    }
    row_index_by_eval = {
        name: {
            (row["_source_language"], row["language"]): row
            for row in rows
        }
        for name, rows in rows_by_eval.items()
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["evaluation", "source_language", "language", "pass@1", "pass@1_se", "pass@5", "pass@5_se"],
            extrasaction="ignore",
        )
        writer.writeheader()

        for input_file in input_files:
            eval_name = evaluation_name(input_file)
            baseline_name = baseline_evaluation_name(eval_name)
            baseline_rows = row_index_by_eval.get(baseline_name or "", {})
            all_diffs = []

            for row in rows_by_eval[eval_name]:
                baseline_row = baseline_rows.get((row["_source_language"], row["language"]))
                se_diff = None
                if baseline_row is not None:
                    if eval_name == baseline_name:
                        se_diff = 0.0
                    elif row["language"] == "total":
                        se_diff = None
                    else:
                        model_results = result_path_from_input(row["input_file"], input_dir)
                        baseline_results = result_path_from_input(baseline_row["input_file"], input_dir)
                        se_diff = paired_pass_at_1_se_diff(model_results, baseline_results)
                        model_scores = task_pass_at_1(model_results)
                        baseline_scores = task_pass_at_1(baseline_results)
                        all_diffs.extend(
                            model_scores[task_id] - baseline_scores[task_id]
                            for task_id in sorted(set(model_scores) & set(baseline_scores))
                        )

                if row["language"] == "total" and baseline_name is not None:
                    se_diff = standard_error(all_diffs) if eval_name != baseline_name else 0.0

                writer.writerow(
                    {
                        "evaluation": eval_name,
                        "source_language": row["_source_language"],
                        "language": row["language"],
                        "pass@1": row["pass@1"],
                        "pass@1_se": format_optional(se_diff, row["pass@1_se"]),
                        "pass@5": row["pass@5"],
                        "pass@5_se": row["pass@5_se"],
                    }
                )
                row_count += 1

    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge evaluation_*.csv files and replace pass@1_se with paired SE_diff vs qwen2.5/qwen3.5 baselines when results JSONL files are available."
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
