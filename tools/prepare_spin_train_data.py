"""Prepare a parquet training dataset from generated SPIN opponent JSONL files.

Usage:
    cd /root/formulaspin/formulaspin
    python tools/prepare_spin_train_data.py \
      --input_dir opponent_dataset/adapter-instruct_2epoch_generation_greedy \
      --output_dir opponent_dataset/spin_adapter_instruct2epoch_iter0_mixed_data \
      --num_fracs 1 \
      --split train

Inputs:
    --input_dir: Directory containing generated JSONL files such as generated_0.jsonl
        or generated_0_rank*.jsonl.
    --output_dir: Directory where the converted parquet dataset will be written.
    --num_fracs: Number of generated shards/fractions to load.
    --split: train writes train_prefs-*.parquet; test writes eval_prefs-*.parquet.

Outputs:
    - A parquet file under output_dir:
        train_prefs-00000-of-00001.parquet or eval_prefs-00000-of-00001.parquet
    - A README.md metadata file beside the parquet export
    - Console summary with input path, output path, and sample count

When to use:
    Use this as a convenience wrapper after generation, when you want a ready-to-train
    parquet dataset directory without calling convert_data.py manually.
"""

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from ..convert_data import load_jsonl_files, convert_to_parquet, write_conversion_readme
except ImportError:
    from convert_data import load_jsonl_files, convert_to_parquet, write_conversion_readme


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert generated SPIN opponent JSONL files into a parquet dataset directory."
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='opponent_dataset/adapter-instruct_2epoch_generation_greedy',
        help='Directory containing generated JSONL files.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='opponent_dataset/spin_adapter_instruct2epoch_iter0_mixed_data',
        help='Directory where the parquet dataset will be written.',
    )
    parser.add_argument(
        '--num_fracs',
        type=int,
        default=1,
        help='Number of generated shards/fractions to load.',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'test'],
        help='Whether to export a training or test-style parquet dataset.',
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_jsonl_files(str(input_dir), args.num_fracs, args.split)
    if not data:
        raise ValueError(f'No data loaded from {input_dir}')

    output_name = 'train_prefs-00000-of-00001.parquet' if args.split == 'train' else 'eval_prefs-00000-of-00001.parquet'
    output_path = output_dir / output_name

    convert_to_parquet(data, str(output_path))
    write_conversion_readme(output_dir, input_dir, args.split, len(data), output_path)

    print('\nPrepared parquet dataset:')
    print(f'  Input: {input_dir}')
    print(f'  Output: {output_path}')
    print(f'  Samples: {len(data)}')


if __name__ == '__main__':
    main()