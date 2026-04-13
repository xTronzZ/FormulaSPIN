"""
Evaluation Script for FormulaSPIN

Evaluates formula generation models on metrics like Exact Match (EM),
Execution Accuracy (EA), and Formula Sketch Match (FSM).
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import re

import torch
from datasets import load_dataset

try:
    from .execution_engine import FormulaExecutor, ExecutionStatus
    from .consensus_polling import ConsensusPoller
    from .model_utils import load_causal_lm, load_tokenizer
except ImportError:
    from execution_engine import FormulaExecutor, ExecutionStatus
    from consensus_polling import ConsensusPoller
    from model_utils import load_causal_lm, load_tokenizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate FormulaSPIN model")
    parser.add_argument('--model', type=str, required=True,
                       help='Path or name of the base or merged model to evaluate')
    parser.add_argument('--adapter_name_or_path', type=str, default=None,
                       help='Optional SFT/LoRA adapter to apply on top of --model for evaluation')
    parser.add_argument('--torch_dtype', type=str, default='bfloat16',
                       choices=['float16', 'bfloat16', 'float32'],
                       help='Precision used when loading model weights')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test dataset')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Evaluation batch size')
    parser.add_argument('--use_consensus', action='store_true',
                       help='Use consensus polling for generation')
    parser.add_argument('--num_candidates', type=int, default=10,
                       help='Number of candidates for consensus polling')
    parser.add_argument('--temperature', type=float, default=1.2,
                       help='Temperature for consensus polling')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    return parser.parse_args()


def extract_formula_sketch(formula: str) -> str:
    """
    Extract function-level sketch from formula (ignoring cell references).

    Example: "SUM(FILTER(E1, C1<4))" -> "SUM(FILTER(Cell, Cell<Num))"
    """
    # Replace cell references with "Cell"
    sketch = re.sub(r'[A-Z]+\d+', 'Cell', formula)

    # Replace numbers with "Num"
    sketch = re.sub(r'\b\d+\b', 'Num', sketch)

    return sketch


def format_table(table_data: List[List[str]], max_rows: int = 20) -> str:
    """
    Format table data as a string for prompt.
    """
    if not table_data:
        return ""

    # Limit to max_rows
    table_data = table_data[:max_rows]

    # Format as rows
    rows = []
    for row in table_data:
        rows.append(" | ".join(str(cell) for cell in row))

    return "\n".join(rows)


def normalize_references(references: Any) -> List[str]:
    """Normalize one or many ground-truth formulas into a deduplicated list."""
    if isinstance(references, str):
        return [references] if references else []

    normalized = []
    for reference in references or []:
        if reference and reference not in normalized:
            normalized.append(reference)
    return normalized


def compute_metrics(
    predictions: List[str],
    references: List[Any],
    tables: List[List[List[str]]],
    executor: FormulaExecutor
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: List of predicted formulas
        references: List of ground truth formulas
        tables: List of table data
        executor: Formula executor

    Returns:
        Dictionary of metrics
    """
    exact_matches = 0
    execution_matches = 0
    sketch_matches = 0
    execution_success = 0
    total = len(predictions)

    for pred, refs_raw, table in zip(predictions, references, tables):
        refs = normalize_references(refs_raw)
        if not refs:
            continue

        # Exact Match (EM)
        if any(pred.strip() == ref.strip() for ref in refs):
            exact_matches += 1

        pred_result = executor.execute_formula(pred, table)

        # Execution Success Rate (ESR)
        if pred_result.status == ExecutionStatus.SUCCESS:
            execution_success += 1

        # Execution Accuracy (EA)
        ref_results = [executor.execute_formula(ref, table) for ref in refs]
        if any(executor.compare_results(pred_result, ref_result) for ref_result in ref_results):
            execution_matches += 1

        # Formula Sketch Match (FSM)
        pred_sketch = extract_formula_sketch(pred)
        ref_sketches = [extract_formula_sketch(ref) for ref in refs]
        if pred_sketch in ref_sketches:
            sketch_matches += 1

    metrics = {
        'exact_match': 100.0 * exact_matches / total,
        'execution_accuracy': 100.0 * execution_matches / total,
        'execution_success_rate': 100.0 * execution_success / total,
        'sketch_match': 100.0 * sketch_matches / total,
        'total_samples': total,
    }

    return metrics


def evaluate_greedy(model, tokenizer, test_data: List[Dict], executor: FormulaExecutor) -> Tuple[List[str], Dict]:
    """Evaluate with greedy decoding."""
    predictions = []

    for sample in tqdm(test_data, desc="Evaluating (greedy)"):
        query = sample['query']
        table = sample.get('table', [])

        # Create prompt
        table_str = format_table(table)
        prompt = f"Generate an Excel formula for the following query:\nQuery: {query}\nTable: {table_str}\nFormula:"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predictions.append(generated.strip())

    # Compute metrics
    references = [sample.get('references', [sample['formula']]) for sample in test_data]
    tables = [sample.get('table', []) for sample in test_data]
    metrics = compute_metrics(predictions, references, tables, executor)

    return predictions, metrics


def evaluate_consensus(
    model,
    tokenizer,
    test_data: List[Dict],
    executor: FormulaExecutor,
    num_candidates: int,
    temperature: float
) -> Tuple[List[str], Dict]:
    """Evaluate with consensus polling."""
    poller = ConsensusPoller(executor=executor)
    predictions = []

    for sample in tqdm(test_data, desc=f"Evaluating (consensus K={num_candidates})"):
        query = sample['query']
        table = sample.get('table', [])

        table_str = format_table(table)
        prompt = (
            f"Generate an Excel formula for the following query:\n"
            f"Query: {query}\n"
            f"Table: {table_str}\n"
            "Formula:"
        )

        # Generate with consensus polling
        result = poller.poll(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            table_data=table,
            num_candidates=num_candidates,
            temperature=temperature,
            max_length=256,
        )

        predictions.append(result.formula)

    # Compute metrics
    references = [sample.get('references', [sample['formula']]) for sample in test_data]
    tables = [sample.get('table', []) for sample in test_data]
    metrics = compute_metrics(predictions, references, tables, executor)

    # Add consensus-specific metrics
    metrics['num_candidates'] = num_candidates
    metrics['temperature'] = temperature

    return predictions, metrics


def main():
    args = parse_arguments()

    print("=" * 60)
    print("FormulaSPIN Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Test data: {args.test_data}")
    print(f"Use consensus: {args.use_consensus}")
    if args.use_consensus:
        print(f"Num candidates: {args.num_candidates}")
        print(f"Temperature: {args.temperature}")
    print("=" * 60)

    # Load model and tokenizer
    print("\nLoading model...")
    model = load_causal_lm(
        base_model_name_or_path=args.model,
        adapter_name_or_path=args.adapter_name_or_path,
        torch_dtype=args.torch_dtype,
        device_map="auto",
    )
    tokenizer = load_tokenizer(args.model)

    # Load test data
    print("Loading test data...")
    if args.test_data.endswith('.json'):
        with open(args.test_data, 'r', encoding='utf-8') as f:
            test_data_raw = json.load(f)

        # Convert to list of samples
        test_data = []
        for table_entry in test_data_raw:
            table = table_entry.get('Table', [])
            for formula_entry in table_entry.get('t5Formulas', []):
                references = []
                for key in ('Formula', 'Formula2'):
                    value = formula_entry.get(key, '')
                    if value and value not in references:
                        references.append(value)

                test_data.append({
                    'query': formula_entry.get('Question', ''),
                    'formula': formula_entry.get('Formula', ''),
                    'references': references,
                    'table': table,
                })
    else:
        dataset = load_dataset(args.test_data, split='test')
        test_data = list(dataset)

    # Limit samples if specified
    if args.max_samples:
        test_data = test_data[:args.max_samples]

    print(f"Evaluating on {len(test_data)} samples")

    # Initialize executor
    executor = FormulaExecutor(use_xlwings=False)

    # Evaluate
    if args.use_consensus:
        predictions, metrics = evaluate_consensus(
            model, tokenizer, test_data, executor,
            args.num_candidates, args.temperature
        )
    else:
        predictions, metrics = evaluate_greedy(
            model, tokenizer, test_data, executor
        )

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name}: {value:.2f}%")
        else:
            print(f"{metric_name}: {value}")
    print("=" * 60)

    # Save results
    results = {
        'metrics': metrics,
        'model': args.model,
        'test_data': args.test_data,
        'use_consensus': args.use_consensus,
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
