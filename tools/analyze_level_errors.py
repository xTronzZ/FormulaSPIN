"""Detailed error analysis for FormulaSPIN by difficulty bucket.

Usage:
    cd /root/formulaspin/formulaspin
    CUDA_VISIBLE_DEVICES=0 python tools/analyze_level_errors.py \
      --model /path/to/base_model \
      --adapter_name_or_path /path/to/adapter \
      --test_data dataset/testset.json \
      --output_dir outputs/train/some_run/analysis/simple_medium_complex_error_analysis \
      --levels simple medium complex \
      --max_per_level 300 \
      --max_new_tokens 256

Inputs:
    --model: Base model path or model id.
    --adapter_name_or_path: Adapter checkpoint being analyzed.
    --test_data: Raw JSON test set in NL2Formula-style format.
    --output_dir: Directory where the analysis files will be written.
    --levels: Difficulty buckets to sample from, typically simple/medium/complex.
    --max_per_level: Maximum number of samples drawn from each level.
    --seed: Random seed for stratified sampling.
    --max_new_tokens: Maximum generation length for greedy decoding.
    --examples_per_category: Number of representative examples to include per error type.

Outputs:
    - level_error_analysis_summary.json
    - level_error_analysis_details.json
    - level_error_analysis_report.md
    - Console progress bar and run summary

What this script does:
    It runs greedy decoding on a stratified subset, checks exact match and execution
    behavior, groups errors into executable and non-executable categories, and writes
    both machine-readable JSON and a human-readable Markdown report.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import sys

import torch
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from ..evaluate import extract_formula_sketch, format_table, normalize_references
    from ..execution_engine import FormulaExecutor, ExecutionStatus
    from ..model_utils import load_causal_lm, load_tokenizer
except ImportError:
    from evaluate import extract_formula_sketch, format_table, normalize_references
    from execution_engine import FormulaExecutor, ExecutionStatus
    from model_utils import load_causal_lm, load_tokenizer


LEVELS = ("simple", "medium", "complex")
LOOKUP_FUNCTIONS = {"FILTER", "INDEX", "MATCH", "XMATCH", "XLOOKUP", "VLOOKUP", "HLOOKUP", "CHOOSECOLS"}
AGG_FUNCTIONS = {"MAX", "MIN", "SUM", "AVERAGE", "COUNT", "COUNTA", "LARGE", "SMALL"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze model errors by difficulty bucket")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter_name_or_path", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--levels", nargs="+", default=list(LEVELS))
    parser.add_argument("--max_per_level", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--examples_per_category", type=int, default=3)
    return parser.parse_args()


def normalize_level(level: str) -> str:
    normalized = (level or "").strip().lower()
    if normalized == "easy":
        return "simple"
    if normalized == "medium":
        return "medium"
    if normalized == "hard":
        return "complex"
    if normalized == "calculation":
        return "calculation"
    return "unknown"


def load_test_samples(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as file:
        raw_entries = json.load(file)

    samples: List[Dict[str, Any]] = []
    for table_entry in raw_entries:
        table = table_entry.get("Table", [])
        table_name = table_entry.get("TableName", "")
        for formula_entry in table_entry.get("t5Formulas", []):
            references = []
            for key in ("Formula", "Formula2"):
                value = formula_entry.get(key, "")
                if value and value not in references:
                    references.append(value)

            samples.append(
                {
                    "table_name": table_name,
                    "query": formula_entry.get("Question", ""),
                    "formula": formula_entry.get("Formula", ""),
                    "references": references,
                    "table": table,
                    "level": normalize_level(formula_entry.get("Level", "")),
                }
            )
    return samples


def stratified_subset(
    samples: Sequence[Dict[str, Any]],
    levels: Sequence[str],
    max_per_level: int,
    seed: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    for level in levels:
        bucket = [sample for sample in samples if sample.get("level") == level]
        if max_per_level and len(bucket) > max_per_level:
            selected.extend(rng.sample(bucket, max_per_level))
        else:
            selected.extend(bucket)
    return selected


def first_function(formula: str) -> str:
    match = re.search(r"([A-Z][A-Z0-9_]*)\(", (formula or "").upper())
    return match.group(1) if match else "UNKNOWN"


def extract_functions(formula: str) -> List[str]:
    return re.findall(r"([A-Z][A-Z0-9_]*)\(", (formula or "").upper())


def extract_literals(formula: str) -> List[str]:
    text_literals = re.findall(r'"([^"]*)"', formula or "")
    number_literals = re.findall(r"(?<![A-Z])(\d+(?:\.\d+)?)", formula or "", flags=re.IGNORECASE)
    return text_literals + number_literals


def extract_choosecols_targets(formula: str) -> List[str]:
    matches = re.findall(r"CHOOSECOLS\([^,]+,\s*([^\)]+)\)", (formula or "").upper())
    targets: List[str] = []
    for match in matches:
        targets.extend(re.findall(r"\d+", match))
    return targets


def extract_columns(formula: str) -> List[str]:
    columns = re.findall(r"\b([A-Z]{1,3})(?=\d)", (formula or "").upper())
    return sorted(set(columns))


def condition_count(formula: str) -> int:
    return len(re.findall(r"<>|>=|<=|=|>|<", formula or ""))


def structural_similarity(formula_a: str, formula_b: str) -> int:
    funcs_a = extract_functions(formula_a)
    funcs_b = extract_functions(formula_b)
    score = 0
    if first_function(formula_a) == first_function(formula_b):
        score += 5
    score += 2 * len(set(funcs_a) & set(funcs_b))
    if extract_formula_sketch(formula_a) == extract_formula_sketch(formula_b):
        score += 4
    score -= abs(condition_count(formula_a) - condition_count(formula_b))
    return score


def choose_best_reference(prediction: str, references: Sequence[str]) -> str:
    if not references:
        return ""
    return max(references, key=lambda ref: structural_similarity(prediction, ref))


def categorize_non_executable(error_text: str) -> str:
    normalized = (error_text or "").lower()
    if any(token in normalized for token in ("parse", "syntax", "token", "expected")):
        return "non_executable_parse_or_syntax"
    if any(token in normalized for token in ("unsupported", "unknown function", "not implemented")):
        return "non_executable_unsupported_function"
    if any(token in normalized for token in ("reference", "range", "cell", "address")):
        return "non_executable_bad_reference"
    return "non_executable_other"


def categorize_executable_miss(prediction: str, reference: str) -> str:
    pred_top = first_function(prediction)
    ref_top = first_function(reference)
    pred_targets = extract_choosecols_targets(prediction)
    ref_targets = extract_choosecols_targets(reference)
    pred_columns = extract_columns(prediction)
    ref_columns = extract_columns(reference)
    pred_literals = extract_literals(prediction)
    ref_literals = extract_literals(reference)
    pred_condition_count = condition_count(prediction)
    ref_condition_count = condition_count(reference)

    if pred_top != ref_top:
        if pred_top in AGG_FUNCTIONS and ref_top in AGG_FUNCTIONS:
            return "aggregation_confusion"
        if (pred_top in LOOKUP_FUNCTIONS) != (ref_top in LOOKUP_FUNCTIONS):
            return "lookup_strategy_confusion"
        return "top_function_mismatch"

    if pred_targets and ref_targets and pred_targets != ref_targets:
        return "wrong_return_column"

    if pred_condition_count != ref_condition_count:
        return "missing_or_extra_filter_condition"

    if pred_literals != ref_literals:
        return "wrong_filter_value_or_threshold"

    if pred_columns != ref_columns:
        return "wrong_column_binding"

    if extract_formula_sketch(prediction) == extract_formula_sketch(reference):
        return "same_sketch_wrong_argument_binding"

    return "structural_drift"


def serialize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    return str(value)


def generate_prediction(model, tokenizer, query: str, table: List[List[str]], max_new_tokens: int) -> str:
    table_str = format_table(table)
    prompt = (
        "Generate an Excel formula for the following query:\n"
        f"Query: {query}\n"
        f"Table: {table_str}\n"
        "Formula:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def analyze_samples(
    model,
    tokenizer,
    executor: FormulaExecutor,
    samples: Sequence[Dict[str, Any]],
    max_new_tokens: int,
) -> Dict[str, Any]:
    results_by_level: Dict[str, Dict[str, Any]] = {
        level: {
            "total": 0,
            "exact_match": 0,
            "execution_match": 0,
            "execution_success": 0,
            "errors": [],
            "error_category_counts": Counter(),
            "top_function_confusions": Counter(),
        }
        for level in LEVELS
    }

    for index, sample in enumerate(tqdm(samples, desc="Running level error analysis"), start=1):
        level = sample["level"]
        if level not in results_by_level:
            continue

        prediction = generate_prediction(model, tokenizer, sample["query"], sample["table"], max_new_tokens)
        references = normalize_references(sample.get("references", [sample.get("formula", "")]))
        ref_results = [executor.execute_formula(reference, sample["table"]) for reference in references]
        pred_result = executor.execute_formula(prediction, sample["table"])
        execution_match = any(executor.compare_results(pred_result, ref_result) for ref_result in ref_results)
        exact_match = any(prediction.strip() == reference.strip() for reference in references)

        bucket = results_by_level[level]
        bucket["total"] += 1
        bucket["exact_match"] += int(exact_match)
        bucket["execution_match"] += int(execution_match)
        bucket["execution_success"] += int(pred_result.status == ExecutionStatus.SUCCESS)

        if execution_match:
            continue

        best_reference = choose_best_reference(prediction, references)
        category = (
            categorize_non_executable(pred_result.error)
            if pred_result.status != ExecutionStatus.SUCCESS
            else categorize_executable_miss(prediction, best_reference)
        )

        pred_top = first_function(prediction)
        ref_top = first_function(best_reference)
        bucket["error_category_counts"][category] += 1
        bucket["top_function_confusions"][f"{pred_top} -> {ref_top}"] += 1
        bucket["errors"].append(
            {
                "sample_index": index,
                "table_name": sample.get("table_name", ""),
                "level": level,
                "query": sample["query"],
                "prediction": prediction,
                "references": references,
                "best_reference": best_reference,
                "exact_match": exact_match,
                "execution_match": execution_match,
                "execution_success": pred_result.status == ExecutionStatus.SUCCESS,
                "prediction_status": pred_result.status.value,
                "prediction_value": serialize_value(pred_result.value),
                "prediction_error": pred_result.error,
                "prediction_top_function": pred_top,
                "reference_top_function": ref_top,
                "prediction_sketch": extract_formula_sketch(prediction),
                "reference_sketch": extract_formula_sketch(best_reference),
                "category": category,
            }
        )

    return results_by_level


def make_summary(results_by_level: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"levels": {}}
    for level, bucket in results_by_level.items():
        total = bucket["total"]
        error_total = len(bucket["errors"])
        summary["levels"][level] = {
            "total": total,
            "exact_match_rate": 100.0 * bucket["exact_match"] / total if total else 0.0,
            "execution_accuracy_rate": 100.0 * bucket["execution_match"] / total if total else 0.0,
            "execution_success_rate": 100.0 * bucket["execution_success"] / total if total else 0.0,
            "error_count": error_total,
            "error_rate": 100.0 * error_total / total if total else 0.0,
            "top_error_categories": bucket["error_category_counts"].most_common(8),
            "top_function_confusions": bucket["top_function_confusions"].most_common(8),
        }
    return summary


def render_examples(errors: Sequence[Dict[str, Any]], examples_per_category: int) -> List[str]:
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for error in errors:
        if len(by_category[error["category"]]) < examples_per_category:
            by_category[error["category"]].append(error)

    lines: List[str] = []
    for category, examples in sorted(by_category.items()):
        lines.append(f"### {category}")
        for example in examples:
            lines.append(f"- Query: {example['query']}")
            lines.append(f"  Table: {example['table_name']}")
            lines.append(f"  Pred: {example['prediction']}")
            lines.append(f"  Ref: {example['best_reference']}")
            lines.append(f"  Status: {example['prediction_status']}")
            if example["prediction_error"]:
                lines.append(f"  Error: {example['prediction_error']}")
    return lines


def render_report(summary: Dict[str, Any], results_by_level: Dict[str, Dict[str, Any]], examples_per_category: int) -> str:
    lines = [
        "# Error Analysis by Difficulty Bucket",
        "",
        "This report focuses on model mistakes in the simple, medium, and complex buckets.",
        "Calculation is intentionally excluded.",
        "",
    ]

    for level in LEVELS:
        level_summary = summary["levels"].get(level)
        if not level_summary or not level_summary["total"]:
            continue

        lines.extend(
            [
                f"## {level}",
                "",
                f"- Samples analyzed: {level_summary['total']}",
                f"- Exact match: {level_summary['exact_match_rate']:.2f}%",
                f"- Execution accuracy: {level_summary['execution_accuracy_rate']:.2f}%",
                f"- Execution success: {level_summary['execution_success_rate']:.2f}%",
                f"- Errors analyzed: {level_summary['error_count']} ({level_summary['error_rate']:.2f}%)",
                "",
                "### Top error categories",
            ]
        )
        for category, count in level_summary["top_error_categories"]:
            ratio = 100.0 * count / max(level_summary["error_count"], 1)
            lines.append(f"- {category}: {count} ({ratio:.2f}% of errors)")

        lines.append("")
        lines.append("### Top function confusions")
        for confusion, count in level_summary["top_function_confusions"]:
            ratio = 100.0 * count / max(level_summary["error_count"], 1)
            lines.append(f"- {confusion}: {count} ({ratio:.2f}% of errors)")

        lines.append("")
        lines.extend(render_examples(results_by_level[level]["errors"], examples_per_category))
        lines.append("")

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

    results_by_level = analyze_samples(model, tokenizer, executor, selected, args.max_new_tokens)
    summary = make_summary(results_by_level)

    summary_path = output_dir / "level_error_analysis_summary.json"
    details_path = output_dir / "level_error_analysis_details.json"
    report_path = output_dir / "level_error_analysis_report.md"

    details_payload = {
        "model": args.model,
        "adapter_name_or_path": args.adapter_name_or_path,
        "test_data": args.test_data,
        "levels": args.levels,
        "max_per_level": args.max_per_level,
        "summary": summary,
        "details": results_by_level,
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    with open(details_path, "w", encoding="utf-8") as file:
        json.dump(details_payload, file, indent=2, ensure_ascii=False)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(render_report(summary, results_by_level, args.examples_per_category))

    print(f"Summary written to {summary_path}")
    print(f"Detailed records written to {details_path}")
    print(f"Markdown report written to {report_path}")


if __name__ == "__main__":
    main()