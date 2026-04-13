"""Performance analysis by reference function type for FormulaSPIN.

Usage:
    cd /root/formulaspin/formulaspin
    CUDA_VISIBLE_DEVICES=0 python tools/analyze_function_type_performance.py \
      --model /path/to/base_model \
      --adapter_name_or_path /path/to/adapter \
      --test_data dataset/testset.json \
      --output_dir outputs/train/some_run/analysis/function_type_performance \
      --levels simple medium complex \
      --max_per_level 300 \
      --max_new_tokens 256

Inputs:
    --model: Base model path or model id.
    --adapter_name_or_path: Adapter checkpoint being analyzed.
    --test_data: Raw JSON test set in NL2Formula-style format.
    --output_dir: Directory where the analysis files will be written.
    --levels: Difficulty buckets to sample from before aggregating by function type.
    --max_per_level: Maximum number of samples drawn from each difficulty bucket.
    --seed: Random seed for stratified sampling.
    --max_new_tokens: Maximum generation length for greedy decoding.
    --min_top_function_support: Minimum support required before a top function is shown.
    --min_family_support: Minimum support required before a function family is shown.

Outputs:
    - function_type_performance_summary.json
    - function_type_performance_details.json
    - function_type_performance_report.md
    - Console progress bar and aggregate statistics

What this script does:
    It runs greedy decoding on a stratified subset, then groups results by reference
    top-level function and broader function family so you can see which function types
    have the largest execution or exact-match gaps.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence
import sys

from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from .analyze_level_errors import (
        LEVELS,
        categorize_executable_miss,
        categorize_non_executable,
        choose_best_reference,
        first_function,
        generate_prediction,
        load_test_samples,
        normalize_level,
        serialize_value,
        stratified_subset,
    )
    from ..evaluate import normalize_references
    from ..execution_engine import FormulaExecutor, ExecutionStatus
    from ..model_utils import load_causal_lm, load_tokenizer
except ImportError:
    from analyze_level_errors import (
        LEVELS,
        categorize_executable_miss,
        categorize_non_executable,
        choose_best_reference,
        first_function,
        generate_prediction,
        load_test_samples,
        normalize_level,
        serialize_value,
        stratified_subset,
    )
    from evaluate import normalize_references
    from execution_engine import FormulaExecutor, ExecutionStatus
    from model_utils import load_causal_lm, load_tokenizer


AGGREGATION_FUNCTIONS = {
    "SUM", "AVERAGE", "MIN", "MAX", "SUMIFS", "AVERAGEIFS", "MINIFS", "MAXIFS",
    "SUMX", "AVERAGEX", "MINX", "MAXX", "MEDIANX",
}
COUNT_FUNCTIONS = {"ROWS", "COUNT", "COUNTA", "COUNTIFS", "DCOUNTX"}
LOOKUP_PROJECTION_FUNCTIONS = {"FILTER", "CHOOSECOLS", "XLOOKUP", "VLOOKUP", "HLOOKUP", "INDEX", "MATCH", "XMATCH"}
SORT_RANK_FUNCTIONS = {"SORT", "SORTBY", "TAKE", "LARGE", "SMALL", "RANK"}
SHAPE_FUNCTIONS = {"UNIQUE", "HSTACK", "VSTACK", "CHOOSE"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze performance by function type")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter_name_or_path", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--levels", nargs="+", default=list(LEVELS))
    parser.add_argument("--max_per_level", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--min_top_function_support", type=int, default=10)
    parser.add_argument("--min_family_support", type=int, default=10)
    return parser.parse_args()


def function_family(function_name: str) -> str:
    if function_name in AGGREGATION_FUNCTIONS:
        return "aggregation"
    if function_name in COUNT_FUNCTIONS:
        return "counting"
    if function_name in LOOKUP_PROJECTION_FUNCTIONS:
        return "lookup_projection"
    if function_name in SORT_RANK_FUNCTIONS:
        return "sorting_ranking"
    if function_name in SHAPE_FUNCTIONS:
        return "shape_projection"
    if function_name == "LET":
        return "let_pipeline"
    return "other"


def normalize_sample_level(level: str) -> str:
    normalized = (level or "").strip().lower()
    if normalized in LEVELS:
        return normalized
    return normalize_level(level)


def init_bucket() -> Dict[str, Any]:
    return {
        "total": 0,
        "exact_match": 0,
        "execution_match": 0,
        "execution_success": 0,
        "executor_artifact_count": 0,
        "levels": Counter(),
        "predicted_top_functions_on_error": Counter(),
        "error_categories": Counter(),
    }


def update_bucket(bucket: Dict[str, Any], record: Dict[str, Any]) -> None:
    bucket["total"] += 1
    bucket["exact_match"] += int(record["exact_match"])
    bucket["execution_match"] += int(record["execution_match"])
    bucket["execution_success"] += int(record["execution_success"])
    bucket["levels"][record["level"]] += 1
    if record["executor_artifact"]:
        bucket["executor_artifact_count"] += 1
    if not record["execution_match"]:
        bucket["predicted_top_functions_on_error"][record["prediction_top_function"]] += 1
        bucket["error_categories"][record["error_category"]] += 1


def finalize_bucket(name: str, bucket: Dict[str, Any]) -> Dict[str, Any]:
    total = bucket["total"]
    return {
        "name": name,
        "total": total,
        "exact_match_rate": 100.0 * bucket["exact_match"] / total if total else 0.0,
        "execution_accuracy_rate": 100.0 * bucket["execution_match"] / total if total else 0.0,
        "execution_success_rate": 100.0 * bucket["execution_success"] / total if total else 0.0,
        "error_count": total - bucket["execution_match"],
        "executor_artifact_count": bucket["executor_artifact_count"],
        "level_distribution": dict(bucket["levels"]),
        "top_predicted_functions_on_error": bucket["predicted_top_functions_on_error"].most_common(8),
        "top_error_categories": bucket["error_categories"].most_common(8),
    }


def analyze_samples(
    model,
    tokenizer,
    executor: FormulaExecutor,
    samples: Sequence[Dict[str, Any]],
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for index, sample in enumerate(tqdm(samples, desc="Running function-type analysis"), start=1):
        prediction = generate_prediction(model, tokenizer, sample["query"], sample["table"], max_new_tokens)
        references = normalize_references(sample.get("references", [sample.get("formula", "")]))
        best_reference = choose_best_reference(prediction, references)
        pred_result = executor.execute_formula(prediction, sample["table"])
        ref_results = [executor.execute_formula(reference, sample["table"]) for reference in references]
        execution_match = any(executor.compare_results(pred_result, ref_result) for ref_result in ref_results)
        exact_match = any(prediction.strip() == reference.strip() for reference in references)
        executor_artifact = bool(exact_match and not execution_match and pred_result.status != ExecutionStatus.SUCCESS)
        error_category = None
        if not execution_match:
            error_category = (
                categorize_non_executable(pred_result.error)
                if pred_result.status != ExecutionStatus.SUCCESS
                else categorize_executable_miss(prediction, best_reference)
            )

        ref_top = first_function(best_reference)
        pred_top = first_function(prediction)

        records.append(
            {
                "sample_index": index,
                "table_name": sample.get("table_name", ""),
                "query": sample["query"],
                "level": normalize_sample_level(sample.get("level", "")),
                "references": references,
                "best_reference": best_reference,
                "reference_top_function": ref_top,
                "reference_function_family": function_family(ref_top),
                "prediction": prediction,
                "prediction_top_function": pred_top,
                "prediction_function_family": function_family(pred_top),
                "exact_match": exact_match,
                "execution_match": execution_match,
                "execution_success": pred_result.status == ExecutionStatus.SUCCESS,
                "executor_artifact": executor_artifact,
                "error_category": error_category,
                "prediction_status": pred_result.status.value,
                "prediction_value": serialize_value(pred_result.value),
                "prediction_error": pred_result.error,
            }
        )

    return records


def summarize_records(records: Sequence[Dict[str, Any]], min_top_function_support: int, min_family_support: int) -> Dict[str, Any]:
    by_top_function: Dict[str, Dict[str, Any]] = defaultdict(init_bucket)
    by_family: Dict[str, Dict[str, Any]] = defaultdict(init_bucket)

    for record in records:
        update_bucket(by_top_function[record["reference_top_function"]], record)
        update_bucket(by_family[record["reference_function_family"]], record)

    top_functions = [
        finalize_bucket(name, bucket)
        for name, bucket in by_top_function.items()
        if bucket["total"] >= min_top_function_support
    ]
    top_functions.sort(key=lambda item: (-item["total"], item["name"]))

    hardest_top_functions = sorted(
        top_functions,
        key=lambda item: (item["execution_accuracy_rate"], -item["total"], item["name"]),
    )

    families = [
        finalize_bucket(name, bucket)
        for name, bucket in by_family.items()
        if bucket["total"] >= min_family_support
    ]
    families.sort(key=lambda item: (-item["total"], item["name"]))

    hardest_families = sorted(
        families,
        key=lambda item: (item["execution_accuracy_rate"], -item["total"], item["name"]),
    )

    return {
        "overall": {
            "total_samples": len(records),
            "exact_match_rate": 100.0 * sum(int(record["exact_match"]) for record in records) / len(records) if records else 0.0,
            "execution_accuracy_rate": 100.0 * sum(int(record["execution_match"]) for record in records) / len(records) if records else 0.0,
            "execution_success_rate": 100.0 * sum(int(record["execution_success"]) for record in records) / len(records) if records else 0.0,
        },
        "top_functions": top_functions,
        "hardest_top_functions": hardest_top_functions[:12],
        "function_families": families,
        "hardest_function_families": hardest_families,
    }


def render_summary_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Name | Support | EM | EA | ESR | Errors | Executor Artifacts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['total']} | {row['exact_match_rate']:.2f}% | {row['execution_accuracy_rate']:.2f}% | {row['execution_success_rate']:.2f}% | {row['error_count']} | {row['executor_artifact_count']} |"
        )
    return lines


def render_detail_block(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for row in rows:
        lines.append(f"### {row['name']}")
        lines.append(f"- Support: {row['total']}")
        lines.append(f"- EM / EA / ESR: {row['exact_match_rate']:.2f}% / {row['execution_accuracy_rate']:.2f}% / {row['execution_success_rate']:.2f}%")
        lines.append(f"- Errors: {row['error_count']}")
        lines.append(f"- Executor artifacts: {row['executor_artifact_count']}")
        lines.append(f"- Level distribution: {row['level_distribution']}")
        if row['top_error_categories']:
            lines.append("- Top error categories:")
            for category, count in row['top_error_categories']:
                lines.append(f"  - {category}: {count}")
        if row['top_predicted_functions_on_error']:
            lines.append("- Top predicted functions on error:")
            for function_name, count in row['top_predicted_functions_on_error']:
                lines.append(f"  - {function_name}: {count}")
        lines.append("")
    return lines


def render_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# Performance by Function Type",
        "",
        "This report groups the sampled evaluation set by the reference formula's top-level function and broader function family.",
        "The sample uses the same stratified setting as the recent level analysis: simple / medium / complex, 300 each.",
        "",
        "## Overall",
        "",
        f"- Samples analyzed: {summary['overall']['total_samples']}",
        f"- Exact match: {summary['overall']['exact_match_rate']:.2f}%",
        f"- Execution accuracy: {summary['overall']['execution_accuracy_rate']:.2f}%",
        f"- Execution success: {summary['overall']['execution_success_rate']:.2f}%",
        "",
        "## Hardest Top Functions",
        "",
    ]
    lines.extend(render_summary_table(summary["hardest_top_functions"]))
    lines.extend([
        "",
        "## Function Families",
        "",
    ])
    lines.extend(render_summary_table(summary["function_families"]))
    lines.extend([
        "",
        "## Hardest Families",
        "",
    ])
    lines.extend(render_summary_table(summary["hardest_function_families"]))
    lines.extend([
        "",
        "## Detailed Top-Function Notes",
        "",
    ])
    lines.extend(render_detail_block(summary["hardest_top_functions"][:8]))
    lines.extend([
        "## Detailed Family Notes",
        "",
    ])
    lines.extend(render_detail_block(summary["hardest_function_families"]))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_arguments()

    samples = load_test_samples(args.test_data)
    selected = stratified_subset(samples, args.levels, args.max_per_level, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(samples)} raw test samples")
    print(f"Analyzing {len(selected)} samples across levels: {', '.join(args.levels)}")

    model = load_causal_lm(
        base_model_name_or_path=args.model,
        adapter_name_or_path=args.adapter_name_or_path,
        torch_dtype=args.torch_dtype,
        device_map="auto",
    )
    tokenizer = load_tokenizer(args.model)
    executor = FormulaExecutor(use_xlwings=False)

    records = analyze_samples(model, tokenizer, executor, selected, args.max_new_tokens)
    summary = summarize_records(records, args.min_top_function_support, args.min_family_support)

    details_payload = {
        "model": args.model,
        "adapter_name_or_path": args.adapter_name_or_path,
        "test_data": args.test_data,
        "levels": args.levels,
        "max_per_level": args.max_per_level,
        "summary": summary,
        "records": records,
    }

    summary_path = output_dir / "function_type_performance_summary.json"
    details_path = output_dir / "function_type_performance_details.json"
    report_path = output_dir / "function_type_performance_report.md"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    with open(details_path, "w", encoding="utf-8") as file:
        json.dump(details_payload, file, indent=2, ensure_ascii=False)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(render_report(summary))

    print(f"Summary written to {summary_path}")
    print(f"Detailed records written to {details_path}")
    print(f"Markdown report written to {report_path}")


if __name__ == "__main__":
    main()