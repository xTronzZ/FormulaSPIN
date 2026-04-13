"""Benchmark execution-based voting on a small test subset.

Usage:
    cd /root/formulaspin/formulaspin
    CUDA_VISIBLE_DEVICES=0 python tools/benchmark_execution_voting.py \
      --model /path/to/base_model \
      --adapter_name_or_path /path/to/adapter \
      --test_data dataset/testset.json \
      --max_samples 128 \
      --ks 1 5 10 20 \
      --temperature 1.2 \
      --max_new_tokens 256 \
      --output_file outputs/benchmarks/testset_smallbatch_consensus_k_sweep.json

Inputs:
    --model: Base model path or model id.
    --adapter_name_or_path: Adapter checkpoint to evaluate.
    --test_data: Raw JSON test set in NL2Formula-style format.
    --max_samples: Maximum number of samples to benchmark.
    --ks: Candidate counts used for execution-voting sweeps.
    --temperature: Sampling temperature used when generating voting candidates.
    --max_new_tokens: Maximum generation length for each candidate.
    --output_file: JSON file where the benchmark summary is saved.

Outputs:
    - A JSON summary containing one metrics record per K value
    - Console progress logs during generation and benchmarking
    - Metrics including exact match, execution accuracy, execution success rate,
      total inference time, and seconds per sample

When to use:
    Use this to compare different execution-voting candidate counts before choosing
    a default K for evaluation or deployment.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from ..evaluate import compute_metrics, format_table
    from ..consensus_polling import ConsensusPoller
    from ..execution_engine import FormulaExecutor
    from ..model_utils import load_causal_lm, load_tokenizer
except ImportError:
    from evaluate import compute_metrics, format_table
    from consensus_polling import ConsensusPoller
    from execution_engine import FormulaExecutor
    from model_utils import load_causal_lm, load_tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark execution-based voting on a test subset")
    parser.add_argument('--model', type=str, required=True, help='Base model path')
    parser.add_argument('--adapter_name_or_path', type=str, required=True, help='Adapter path')
    parser.add_argument('--test_data', type=str, required=True, help='Path to testset JSON')
    parser.add_argument('--max_samples', type=int, default=128, help='Number of test samples to benchmark')
    parser.add_argument('--temperature', type=float, default=1.2, help='Sampling temperature for voting')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Max new tokens for generation')
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 5, 10, 20], help='Candidate counts to benchmark')
    parser.add_argument('--torch_dtype', type=str, default='bfloat16', help='Torch dtype for model loading')
    parser.add_argument('--output_file', type=str, required=True, help='Where to save benchmark JSON')
    return parser.parse_args()


def load_test_samples(test_data_path: str, max_samples: int):
    with open(test_data_path, 'r', encoding='utf-8') as file:
        raw_data = json.load(file)

    samples = []
    for table_entry in raw_data:
        table = table_entry.get('Table', [])
        for formula_entry in table_entry.get('t5Formulas', []):
            references = []
            for key in ('Formula', 'Formula2'):
                value = formula_entry.get(key, '')
                if value and value not in references:
                    references.append(value)

            samples.append({
                'query': formula_entry.get('Question', ''),
                'formula': formula_entry.get('Formula', ''),
                'references': references,
                'table': table,
            })

            if len(samples) >= max_samples:
                return samples

    return samples


def build_prompt(sample) -> str:
    table_str = format_table(sample.get('table', []))
    return (
        f"Generate an Excel formula for the following query:\n"
        f"Query: {sample['query']}\n"
        f"Table: {table_str}\n"
        "Formula:"
    )


def main():
    args = parse_arguments()

    test_samples = load_test_samples(args.test_data, args.max_samples)
    print(f"Loaded {len(test_samples)} test samples from {args.test_data}", flush=True)

    model = load_causal_lm(
        base_model_name_or_path=args.model,
        adapter_name_or_path=args.adapter_name_or_path,
        torch_dtype=args.torch_dtype,
        device_map='auto',
    )
    tokenizer = load_tokenizer(args.model)
    executor = FormulaExecutor(use_xlwings=False)
    poller = ConsensusPoller(executor=executor)

    references = [sample.get('references', [sample['formula']]) for sample in test_samples]
    tables = [sample.get('table', []) for sample in test_samples]

    summary = {
        'model': args.model,
        'adapter_name_or_path': args.adapter_name_or_path,
        'test_data': args.test_data,
        'max_samples': len(test_samples),
        'temperature': args.temperature,
        'max_new_tokens': args.max_new_tokens,
        'ks': args.ks,
        'results': [],
    }

    for k in args.ks:
        predictions: List[str] = []
        started_at = time.perf_counter()

        for index, sample in enumerate(test_samples, start=1):
            result = poller.poll(
                model=model,
                tokenizer=tokenizer,
                prompt=build_prompt(sample),
                table_data=sample.get('table', []),
                num_candidates=k,
                temperature=args.temperature,
                max_length=args.max_new_tokens,
            )
            predictions.append(result.formula)

            if index % 25 == 0 or index == len(test_samples):
                print(f"K={k}: processed {index}/{len(test_samples)}", flush=True)

        elapsed = time.perf_counter() - started_at
        metrics = compute_metrics(predictions, references, tables, executor)

        record = {
            'k': k,
            'exact_match': metrics['exact_match'],
            'execution_accuracy': metrics['execution_accuracy'],
            'execution_success_rate': metrics['execution_success_rate'],
            'sketch_match': metrics['sketch_match'],
            'total_samples': metrics['total_samples'],
            'inference_time_seconds': elapsed,
            'seconds_per_sample': elapsed / max(len(test_samples), 1),
        }
        summary['results'].append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Saved benchmark summary to {output_path}", flush=True)


if __name__ == '__main__':
    main()