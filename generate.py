"""
Formula Generation Script

Generates synthetic formulas using the opponent model for self-play training.
Implements Step 1 from the SPIN pipeline.
"""

import argparse
import json
import os
from pathlib import Path
import time
from typing import List, Dict, Any
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore")

try:
    from .model_utils import load_causal_lm, load_tokenizer
except ImportError:
    from model_utils import load_causal_lm, load_tokenizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate formulas for FormulaSPIN")
    parser.add_argument('--model', type=str, required=True,
                       help='Path or name of the base or merged model to use for generation')
    parser.add_argument('--adapter_name_or_path', type=str, default=None,
                       help='Optional SFT/LoRA adapter to apply on top of --model for generation')
    parser.add_argument('--torch_dtype', type=str, default='bfloat16',
                       choices=['float16', 'bfloat16', 'float32'],
                       help='Precision used when loading model weights')
    parser.add_argument('--input_data', type=str, required=True,
                       help='Path to input dataset (JSON file or HuggingFace dataset name)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save generated formulas')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for generation')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature for generation')
    parser.add_argument('--do_sample', action='store_true',
                       help='Enable sampling during generation (disabled by default for stricter format adherence)')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                       help='Maximum number of new tokens to generate')
    parser.add_argument('--data_frac', type=int, default=0,
                       help='Data fraction index for parallel generation')
    parser.add_argument('--frac_len', type=int, default=0,
                       help='Length of each data fraction (0 = use all data)')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use')
    parser.add_argument('--prompt_template', type=str,
                       default=(
                           "Generate one valid Microsoft Excel formula for the following query. "
                           "Return only the formula itself. Do not include any explanation, notes, Markdown, quotes, backticks, labels, or extra text.\n"
                           "Query: {query}\n"
                           "Table: {table}\n"
                           "Microsoft Excel formula:"
                       ),
                       help='Prompt template for formula generation')
    parser.add_argument('--prompt_template_file', type=str, default=None,
                       help='Optional path to a text file containing the prompt template')
    return parser.parse_args()


def load_prompt_template(args) -> str:
    """Load the prompt template from a file when provided, otherwise use the inline template."""
    if args.prompt_template_file:
        with open(args.prompt_template_file, 'r', encoding='utf-8') as file:
            return file.read().strip()
    return args.prompt_template


def format_table(table_data: List[List[str]], max_rows: int = 30) -> str:
    """
    Format table data as a string for prompt.

    Args:
        table_data: 2D list of table data
        max_rows: Maximum number of rows to include

    Returns:
        Formatted table string
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


def create_prompt(query: str, table_data: List[List[str]], template: str) -> str:
    """
    Create generation prompt from query and table data.

    Args:
        query: Natural language query
        table_data: Table data
        template: Prompt template

    Returns:
        Formatted prompt string
    """
    table_str = format_table(table_data)
    prompt = template.format(query=query, table=table_str)
    return prompt


def prepare_prompts(prompts: List[str], tokenizer, batch_size: int = 16):
    """
    Prepare prompts for batched tokenization.

    Args:
        prompts: List of prompt strings
        tokenizer: Tokenizer
        batch_size: Batch size

    Returns:
        List of batched tokenized prompts
    """
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    batches_tok = []

    tokenizer.padding_side = "left"
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding='longest',
                truncation=False,
                pad_to_multiple_of=8,
                add_special_tokens=True
            )
        )
    tokenizer.padding_side = "right"

    return batches_tok


def load_nl2formula_dataset(data_path: str, split: str = 'train') -> List[Dict[str, Any]]:
    """
    Load NL2FORMULA dataset.

    Args:
        data_path: Path to JSON file
        split: Dataset split

    Returns:
        List of data samples
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process NL2FORMULA format
    samples = []
    for table_entry in data:
        table_data = table_entry.get('Table', [])
        formulas = table_entry.get('t5Formulas', [])

        for formula_entry in formulas:
            references = []
            for key in ('Formula', 'Formula2'):
                value = formula_entry.get(key, '')
                if value and value not in references:
                    references.append(value)

            sample = {
                'query': formula_entry.get('Question', ''),
                'formula': formula_entry.get('Formula', ''),
                'references': references,
                'table': table_data,
                'level': formula_entry.get('Level', ''),
            }
            samples.append(sample)

    return samples


def get_output_filename(output_dir: Path, split: str, data_frac: int) -> Path:
    """Return the final output filename for a generation run."""
    if split == 'test':
        return output_dir / f"generated_{data_frac}_test.jsonl"
    return output_dir / f"generated_{data_frac}.jsonl"


def get_shard_filename(output_dir: Path, split: str, data_frac: int, process_index: int) -> Path:
    """Return the per-process shard filename used during streaming generation."""
    if split == 'test':
        return output_dir / f"generated_{data_frac}_rank{process_index}_test.jsonl"
    return output_dir / f"generated_{data_frac}_rank{process_index}.jsonl"


def infer_dataset_kind(input_data: str, split: str) -> str:
    """Infer whether the generated directory is for train, valid, or test data."""
    marker = f"{input_data} {split}".lower()
    if 'valid' in marker or split.lower() in {'valid', 'validation'}:
        return 'validset'
    if 'test' in marker:
        return 'testset'
    if 'train' in marker:
        return 'training dataset'
    return f'{split} split dataset'


def write_generation_readme(output_dir: Path, args, sample_count: int, output_file: Path):
    """Persist provenance metadata alongside generated opponent data."""
    dataset_kind = infer_dataset_kind(args.input_data, args.split)
    decoding = 'sampled' if args.do_sample else 'greedy'
    temperature = str(args.temperature) if args.do_sample else 'N/A (greedy decoding)'
    prompt_template_source = args.prompt_template_file or '(inline prompt template)'
    adapter_name_or_path = args.adapter_name_or_path or '(none)'

    content = """# Opponent Data Metadata

- Data type: generated opponent data
- Dataset kind: {dataset_kind}
- Base model: {base_model}
- Adapter: {adapter}
- Input data: {input_data}
- Split: {split}
- Decoding: {decoding}
- Temperature: {temperature}
- Max new tokens: {max_new_tokens}
- Prompt template source: {prompt_template_source}
- Output file: {output_file}
- Total samples: {sample_count}
""".format(
        dataset_kind=dataset_kind,
        base_model=args.model,
        adapter=adapter_name_or_path,
        input_data=args.input_data,
        split=args.split,
        decoding=decoding,
        temperature=temperature,
        max_new_tokens=args.max_new_tokens,
        prompt_template_source=prompt_template_source,
        output_file=output_file.name,
        sample_count=sample_count,
    )

    (output_dir / 'README.md').write_text(content, encoding='utf-8')


def main():
    args = parse_arguments()
    prompt_template = load_prompt_template(args)

    if not args.do_sample and args.temperature != 0.8:
        warnings.warn(
            "--temperature has no effect unless --do_sample is enabled; generation will run greedily.",
            stacklevel=2,
        )

    # Initialize accelerator for multi-GPU generation
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    if accelerator.is_local_main_process:
        print(f"Loading model from {args.model}...")

    model = load_causal_lm(
        base_model_name_or_path=args.model,
        adapter_name_or_path=args.adapter_name_or_path,
        torch_dtype=args.torch_dtype,
        device_map={"": accelerator.process_index},
    )
    tokenizer = load_tokenizer(args.model)

    # Load dataset
    if accelerator.is_local_main_process:
        print(f"Loading dataset from {args.input_data}...")

    if args.input_data.endswith('.json'):
        # Load from local JSON file
        data_samples = load_nl2formula_dataset(args.input_data, args.split)
    else:
        # Try loading from HuggingFace
        dataset = load_dataset(args.input_data, split=args.split)
        data_samples = list(dataset)

    # Apply data fraction if specified
    if args.frac_len > 0:
        start_idx = args.frac_len * args.data_frac
        end_idx = min(args.frac_len * (args.data_frac + 1), len(data_samples))
        data_samples = data_samples[start_idx:end_idx]

        if accelerator.is_local_main_process:
            print(f"Using data fraction {args.data_frac}: samples {start_idx} to {end_idx}")

    # Create prompts
    records = []

    for sample in data_samples:
        query = sample.get('query', sample.get('Question', ''))
        table = sample.get('table', sample.get('Table', []))
        formula_gt = sample.get('formula', sample.get('Formula', ''))

        prompt = create_prompt(query, table, prompt_template)
        records.append({
            'prompt': prompt,
            'query': query,
            'formula_gt': formula_gt,
            'references': sample.get('references', [formula_gt] if formula_gt else []),
            'table': table,
        })

    # Sync GPUs and start timer
    accelerator.wait_for_everyone()
    start_time = time.time()

    shard_filename = get_shard_filename(
        output_dir, args.split, args.data_frac, accelerator.process_index
    )
    if shard_filename.exists():
        shard_filename.unlink()

    # Split prompts among GPUs
    with accelerator.split_between_processes(records) as records_split:
        prompts_split = [record['prompt'] for record in records_split]
        prompt_batches = prepare_prompts(prompts_split, tokenizer, batch_size=args.batch_size)

        with open(shard_filename, 'a', encoding='utf-8', buffering=1) as shard_file:
            for batch_idx, prompts_tokenized in enumerate(
                tqdm(prompt_batches, disable=not accelerator.is_local_main_process)
            ):
                batch_start = batch_idx * args.batch_size
                batch_records = records_split[batch_start:batch_start + len(prompts_tokenized['input_ids'])]

            # Generate
                prompts_tokenized = prompts_tokenized.to(accelerator.device)
                with torch.inference_mode():
                    generation_kwargs = {
                        'max_new_tokens': args.max_new_tokens,
                        'do_sample': args.do_sample,
                        'pad_token_id': tokenizer.eos_token_id,
                    }
                    if args.do_sample:
                        generation_kwargs['temperature'] = args.temperature

                    outputs_tokenized = model.generate(
                        **prompts_tokenized,
                        **generation_kwargs,
                    )

                # Remove prompt from generated tokens
                outputs_tokenized = [
                    tok_out[len(tok_in):]
                    for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)
                ]

                # Decode and stream results to disk
                outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
                generated_formulas = [output.strip() for output in outputs]

                for record, generated_formula in zip(batch_records, generated_formulas):
                    entry = {
                        "prompt": record['prompt'],
                        "real": [
                            {"role": "user", "content": record['query']},
                            {"role": "assistant", "content": record['formula_gt']}
                        ],
                        "references": record.get('references', [record['formula_gt']] if record['formula_gt'] else []),
                        "generated": [
                            {"role": "user", "content": record['query']},
                            {"role": "assistant", "content": generated_formula.strip()}
                        ],
                        "table": record['table']
                    }
                    shard_file.write(json.dumps(entry) + '\n')
                shard_file.flush()

    accelerator.wait_for_everyone()

    # Save results on main process
    if accelerator.is_local_main_process:
        elapsed_time = time.time() - start_time
        print(f"Generation completed in {elapsed_time:.2f} seconds")
        print(f"Generated {len(records)} formulas")

        filename = get_output_filename(output_dir, args.split, args.data_frac)
        with open(filename, 'w', encoding='utf-8') as outfile:
            for process_index in range(accelerator.num_processes):
                shard_path = get_shard_filename(output_dir, args.split, args.data_frac, process_index)
                if not shard_path.exists():
                    continue
                with open(shard_path, 'r', encoding='utf-8') as shard_file:
                    for line in shard_file:
                        outfile.write(line)
                if shard_path != filename:
                    shard_path.unlink()

        write_generation_readme(output_dir, args, len(records), filename)

        print(f"Saved results to {filename}")


if __name__ == "__main__":
    main()
