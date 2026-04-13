"""
Data Conversion Script

Converts generated JSONL files to Parquet format for training.
Implements Step 1.5 from the SPIN pipeline.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from datasets import Dataset


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert generated data to Parquet format")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing generated JSONL files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save converted Parquet files')
    parser.add_argument('--num_fracs', type=int, required=True,
                       help='Number of fraction files to combine')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split (train or test)')
    return parser.parse_args()


def load_jsonl_files(input_dir: str, num_fracs: int, split: str = 'train') -> List[Dict[str, Any]]:
    """
    Load and combine multiple JSONL files.

    Args:
        input_dir: Directory containing JSONL files
        num_fracs: Number of fraction files
        split: Dataset split

    Returns:
        List of data samples
    """
    all_data = []
    input_path = Path(input_dir)

    for frac_idx in range(num_fracs):
        if split == 'test':
            filename = input_path / f"generated_{frac_idx}_test.jsonl"
            shard_pattern = f"generated_{frac_idx}_rank*_test.jsonl"
        else:
            filename = input_path / f"generated_{frac_idx}.jsonl"
            shard_pattern = f"generated_{frac_idx}_rank*.jsonl"

        if filename.exists():
            files_to_load = [filename]
        else:
            files_to_load = sorted(input_path.glob(shard_pattern))

        if not files_to_load:
            print(f"Warning: File {filename} not found and no shards matched {shard_pattern}, skipping...")
            continue

        for file_path in files_to_load:
            print(f"Loading {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        all_data.append(data)

    print(f"Loaded {len(all_data)} total samples")
    return all_data


def convert_to_parquet(data: List[Dict[str, Any]], output_path: str):
    """
    Convert data to Parquet format using HuggingFace datasets.

    Args:
        data: List of data samples
        output_path: Path to save Parquet file
    """
    # Create dataset
    dataset = Dataset.from_list(data)

    # Save as Parquet
    dataset.to_parquet(output_path)
    print(f"Saved {len(data)} samples to {output_path}")


def parse_source_readme(input_dir: Path) -> Dict[str, str]:
    """Read generation metadata when the source directory already carries a README."""
    readme_path = input_dir / 'README.md'
    if not readme_path.exists():
        return {}

    metadata: Dict[str, str] = {}
    for line in readme_path.read_text(encoding='utf-8').splitlines():
        if not line.startswith('- '):
            continue
        key, separator, value = line[2:].partition(':')
        if separator:
            metadata[key.strip().lower()] = value.strip()
    return metadata


def infer_dataset_kind(input_dir: str, split: str, source_metadata: Dict[str, str]) -> str:
    """Infer whether the converted directory represents valid, test, or training data."""
    source_kind = source_metadata.get('dataset kind')
    if source_kind:
        return source_kind

    marker = f"{input_dir} {split}".lower()
    if 'valid' in marker:
        return 'validset'
    if 'test' in marker:
        return 'testset'
    return 'training dataset'


def write_conversion_readme(output_dir: Path, input_dir: Path, split: str, sample_count: int, output_file: Path):
    """Persist provenance metadata alongside converted parquet datasets."""
    source_metadata = parse_source_readme(input_dir)
    dataset_kind = infer_dataset_kind(str(input_dir), split, source_metadata)

    content = """# Opponent Data Metadata

- Data type: converted opponent data
- Dataset kind: {dataset_kind}
- Source generation directory: {source_generation_directory}
- Base model: {base_model}
- Adapter: {adapter}
- Input split: {split}
- Output file: {output_file}
- Total samples: {sample_count}
""".format(
        dataset_kind=dataset_kind,
        source_generation_directory=input_dir,
        base_model=source_metadata.get('base model', '(unknown)'),
        adapter=source_metadata.get('adapter', '(unknown)'),
        split=split,
        output_file=output_file.name,
        sample_count=sample_count,
    )

    (output_dir / 'README.md').write_text(content, encoding='utf-8')


def main():
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all JSONL files
    data = load_jsonl_files(args.input_dir, args.num_fracs, args.split)

    if len(data) == 0:
        print("Error: No data loaded!")
        return

    # Convert and save
    if args.split == 'train':
        # Save the full self-play dataset as training preferences.
        train_path = output_dir / "train_prefs-00000-of-00001.parquet"
        convert_to_parquet(data, str(train_path))
        write_conversion_readme(output_dir, Path(args.input_dir), args.split, len(data), train_path)

        print(f"\nConversion complete:")
        print(f"  Train samples: {len(data)}")
        print("  Validation: use the raw validset via train_formulaspin.py --valid_data")

    else:
        # Optional held-out preference export for external analysis.
        eval_path = output_dir / "eval_prefs-00000-of-00001.parquet"
        convert_to_parquet(data, str(eval_path))
        write_conversion_readme(output_dir, Path(args.input_dir), args.split, len(data), eval_path)

        print(f"\nConversion complete:")
        print(f"  Eval samples: {len(data)}")


if __name__ == "__main__":
    main()
