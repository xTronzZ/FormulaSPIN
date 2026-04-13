"""
FormulaSPIN Training Script

Main script for running FormulaSPIN self-play fine-tuning.
Implements the complete training pipeline from Algorithm 1.
"""

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import torch
import yaml
from transformers import (
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator

try:
    from .formula_spin_trainer import FormulaSPINTrainer, FormulaSPINConfig, FormulaSPINDataCollator
    from .execution_engine import FormulaExecutor, ExecutionStatus
    from .model_utils import (
        DEFAULT_POLICY_ADAPTER_NAME,
        DEFAULT_REFERENCE_ADAPTER_NAME,
        load_causal_lm,
        load_shared_reference_policy_model,
        set_peft_base_model_name_or_path,
        load_tokenizer,
    )
except ImportError:
    from formula_spin_trainer import FormulaSPINTrainer, FormulaSPINConfig, FormulaSPINDataCollator
    from execution_engine import FormulaExecutor, ExecutionStatus
    from model_utils import (
        DEFAULT_POLICY_ADAPTER_NAME,
        DEFAULT_REFERENCE_ADAPTER_NAME,
        load_causal_lm,
        load_shared_reference_policy_model,
        set_peft_base_model_name_or_path,
        load_tokenizer,
    )


DEFAULT_PROMPT_TEMPLATE = (
    "Generate one valid Microsoft Excel formula for the following query. "
    "Return only the formula itself. Do not include any explanation, notes, Markdown, quotes, backticks, labels, or extra text.\n"
    "Query: {query}\n"
    "Table: {table}\n"
    "Microsoft Excel formula:"
)

OUTPUTS_ROOT = Path(__file__).resolve().parent / 'outputs'
DEFAULT_RUNTIME_LOG_PATH = str(OUTPUTS_ROOT / 'logs' / 'train_runtime.log')
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / 'configs' / 'training_config.yaml'


def parse_bool_arg(value):
    """Parse flexible boolean CLI/config values."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


class TeeStream:
    """Mirror writes to the original stream and a line-buffered log file."""

    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], 'encoding', 'utf-8')
        self.errors = getattr(streams[0], 'errors', 'strict')

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)

    def fileno(self):
        return self.streams[0].fileno()


def start_runtime_log(log_file: str):
    """Tee stdout/stderr to a persistent runtime log file."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, 'a', encoding='utf-8', buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)
    print(f"\n===== FormulaSPIN run started {datetime.now().isoformat(timespec='seconds')} pid={os.getpid()} =====")
    print(f"Runtime log: {log_path}")
    return log_handle, original_stdout, original_stderr


def resolve_runtime_log_path(log_file: str, global_rank: int, local_rank: int) -> str:
    """Derive a per-rank runtime log path from the configured base path."""
    log_path = Path(log_file)
    if log_path.exists() and log_path.is_dir():
        return str(log_path / f'train_runtime.rank{global_rank}.gpu{local_rank}.log')

    suffix = log_path.suffix or '.log'
    stem = log_path.stem if log_path.suffix else log_path.name
    ranked_name = f"{stem}.rank{global_rank}.gpu{local_rank}{suffix}"
    return str(log_path.with_name(ranked_name))


def stop_runtime_log(log_handle, original_stdout, original_stderr):
    """Restore stdout/stderr after tee logging."""
    if log_handle is None:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.close()


def configure_nccl_for_rtx40():
    """Set conservative NCCL defaults for RTX 40-series cards when not provided."""
    os.environ.setdefault('NCCL_P2P_DISABLE', '1')
    os.environ.setdefault('NCCL_IB_DISABLE', '1')


def patch_deepspeed_optimizer_compat():
    """Add no-op train/eval methods expected by newer transformers training loops."""
    if not hasattr(torch.optim.Optimizer, 'train'):
        torch.optim.Optimizer.train = lambda self: None
    if not hasattr(torch.optim.Optimizer, 'eval'):
        torch.optim.Optimizer.eval = lambda self: None

    try:
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    except Exception:
        DeepSpeedZeroOptimizer = None

    try:
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
    except Exception:
        DeepSpeedZeroOptimizer_Stage3 = None

    for optimizer_cls in (DeepSpeedZeroOptimizer, DeepSpeedZeroOptimizer_Stage3):
        if optimizer_cls is None:
            continue
        if not hasattr(optimizer_cls, 'train'):
            optimizer_cls.train = lambda self: None
        if not hasattr(optimizer_cls, 'eval'):
            optimizer_cls.eval = lambda self: None


def prepare_model_for_gradient_checkpointing(model):
    """Enable input grads for LoRA-style training under gradient checkpointing."""
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, model_input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if hasattr(model, 'config'):
        model.config.use_cache = False


def save_model_without_hub_lookup(model, output_dir: str, base_model_name_or_path: str, is_main_process: bool, selected_adapters=None):
    """Save PEFT adapters without PEFT auto-probing the Hub for base config metadata."""
    set_peft_base_model_name_or_path(model, base_model_name_or_path)
    save_kwargs = {
        'is_main_process': is_main_process,
        'save_embedding_layers': False,
    }
    if selected_adapters is not None:
        save_kwargs['selected_adapters'] = selected_adapters
    model.save_pretrained(output_dir, **save_kwargs)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load a nested YAML config file and flatten it into argparse-style keys."""
    resolved_path = Path(config_path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = Path.cwd() / resolved_path

    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with open(resolved_path, 'r', encoding='utf-8') as file:
        raw_config = yaml.safe_load(file) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {resolved_path}")

    flattened: Dict[str, Any] = {}

    def _flatten(mapping: Dict[str, Any]):
        for key, value in mapping.items():
            if isinstance(value, dict):
                _flatten(value)
            else:
                flattened[key] = value

    _flatten(raw_config)
    if 'lambda_reg' in flattened and 'spin_logit_scale' not in flattened:
        flattened['spin_logit_scale'] = flattened['lambda_reg']
    return flattened


def normalize_optional_config_values(args):
    """Map explicit config placeholders back to runtime defaults."""
    optional_string_fields = (
        'adapter_name_or_path',
        'ref_model_name_or_path',
        'ref_adapter_name_or_path',
        'eval_data',
        'dataset_mixer',
        'prompt_template_file',
        'deepspeed',
        'log_file',
    )
    for field_name in optional_string_fields:
        value = getattr(args, field_name, None)
        if isinstance(value, str) and value.strip().lower() in {'', 'none', 'null'}:
            setattr(args, field_name, None)

    optional_int_fields = (
        'preprocessing_num_workers',
        'eval_accumulation_steps',
    )
    for field_name in optional_int_fields:
        value = getattr(args, field_name, None)
        if isinstance(value, int) and value < 0:
            setattr(args, field_name, None)

    return args


def build_argument_parser():
    """Build the argument parser for FormulaSPIN training."""
    parser = argparse.ArgumentParser(description="Train FormulaSPIN model")
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH),
                       help='YAML config file loaded before CLI overrides')

    # Model arguments
    parser.add_argument('--model_name_or_path', type=str, default=None,
                       help='Path or name of the base model')
    parser.add_argument('--adapter_name_or_path', type=str, default=None,
                       help='Optional SFT/LoRA adapter to merge into the base model before training')
    parser.add_argument('--ref_model_name_or_path', type=str, default=None,
                       help='Path to reference model (if None, uses copy of base model)')
    parser.add_argument('--use_shared_reference_adapter', action='store_true',
                       help='Share one base model between trainable policy and frozen reference adapters')
    parser.add_argument('--ref_adapter_name_or_path', type=str, default=None,
                       help='Optional reference adapter path for shared-base training (defaults to --adapter_name_or_path)')
    parser.add_argument('--policy_adapter_name', type=str, default=DEFAULT_POLICY_ADAPTER_NAME,
                       help='Internal adapter name used for the trainable policy adapter')
    parser.add_argument('--reference_adapter_name', type=str, default=DEFAULT_REFERENCE_ADAPTER_NAME,
                       help='Internal adapter name used for the frozen reference adapter')
    parser.add_argument('--torch_dtype', type=str, default='bfloat16',
                       choices=['float16', 'bfloat16', 'float32'],
                       help='Precision used when loading the model weights')
    parser.add_argument('--use_flash_attention_2', type=parse_bool_arg, default=False,
                       help='Enable flash_attention_2 when the model/backend supports it')

    # Data arguments
    parser.add_argument('--train_data', type=str, default=None,
                       help='Path to training data (Parquet or dataset name)')
    parser.add_argument('--eval_data', type=str, default=None,
                       help='Path to evaluation data')
    parser.add_argument('--dataset_mixer', type=str, default=None,
                       help='JSON string of dataset mixer (e.g., \'{"dataset1": 0.5, "dataset2": 0.5}\')')
    parser.add_argument('--prompt_template', type=str, default=DEFAULT_PROMPT_TEMPLATE,
                       help='Prompt template used to generate the opponent formulas when the dataset lacks a stored prompt')
    parser.add_argument('--prompt_template_file', type=str, default=None,
                       help='Optional text file containing the prompt template used during generation')
    parser.add_argument('--preprocessing_num_workers', type=int, default=None,
                       help='Number of worker processes used during dataset preprocessing/tokenization')

    # Training arguments
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for model checkpoints')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1,
                       help='Training batch size per device')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1,
                       help='Evaluation batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.05,
                       help='Warmup ratio')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                       help='Learning rate scheduler name passed to TrainingArguments')
    parser.add_argument('--max_length', type=int, default=640,
                       help='Maximum sequence length')
    parser.add_argument('--max_prompt_length', type=int, default=448,
                       help='Maximum prompt length')
    parser.add_argument('--adam_beta1', type=float, default=0.9,
                       help='Adam/RMSProp beta1 passed to TrainingArguments when supported')
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                       help='Adam beta2 passed to TrainingArguments when supported')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                       help='Optimizer epsilon passed to TrainingArguments when supported')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Gradient clipping norm')

    # FormulaSPIN specific arguments
    parser.add_argument('--beta_max', type=float, default=0.25,
                       help='Maximum beta value for adaptive curriculum')
    parser.add_argument('--spin_logit_scale', type=float, default=1.0,
                       help='Scale coefficient applied to SPIN preference logits inside the loss')
    parser.add_argument('--lambda_reg', dest='spin_logit_scale', type=float, default=None,
                       help=argparse.SUPPRESS)
    parser.add_argument('--loss_type', type=str, default='sigmoid',
                       choices=['sigmoid', 'hinge'],
                       help='Loss type for SPIN')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature for generation')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--bf16', action='store_true',
                       help='Use bfloat16 precision')
    parser.add_argument('--fp16', action='store_true',
                       help='Use float16 precision')
    parser.add_argument('--logging_steps', type=int, default=20,
                       help='Logging frequency')
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='Checkpoint save frequency')
    parser.add_argument('--eval_steps', type=int, default=1000,
                       help='Evaluation frequency')
    parser.add_argument('--save_total_limit', type=int, default=2,
                       help='Maximum number of checkpoints to keep')
    parser.add_argument('--evaluation_strategy', type=str, default='no',
                       help='Evaluation scheduling strategy passed to TrainingArguments')
    parser.add_argument('--eval_accumulation_steps', type=int, default=None,
                       help='Eval accumulation steps passed to TrainingArguments')
    parser.add_argument('--valid_data', type=str, default='formulaspin/dataset/testset.json',
                       help='Raw test set used for periodic generation-based evaluation')
    parser.add_argument('--valid_steps', type=int, default=500,
                       help='Run a small validation sweep every N optimizer steps (0 disables it)')
    parser.add_argument('--valid_subset_size', type=int, default=1024,
                       help='Number of validation samples in the fixed periodic validation subset')
    parser.add_argument('--valid_batch_size', type=int, default=8,
                       help='Batch size used during periodic validation generation')
    parser.add_argument('--valid_max_new_tokens', type=int, default=256,
                       help='Maximum new tokens used during periodic validation generation')
    parser.add_argument('--deepspeed', type=str, default=None,
                       help='Path to a DeepSpeed config file')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable activation checkpointing during training')
    parser.add_argument('--tf32', action='store_true',
                       help='Allow TF32 matmul kernels on Ampere/Hopper GPUs')
    parser.add_argument('--optim', type=str, default='adamw_torch',
                       help='Optimizer name passed to Hugging Face TrainingArguments')
    parser.add_argument('--report_to', type=str, default='none',
                       help='TrainingArguments report_to setting')
    parser.add_argument('--remove_unused_columns', type=parse_bool_arg, default=False,
                       help='TrainingArguments remove_unused_columns setting')
    parser.add_argument('--log_file', type=str, default=DEFAULT_RUNTIME_LOG_PATH,
                       help='Path to the realtime training log file (rank 0 only)')

    return parser


def parse_arguments():
    """Parse command line arguments with YAML defaults and CLI overrides."""
    parser = build_argument_parser()

    config_probe_parser = argparse.ArgumentParser(add_help=False)
    config_probe_parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH))
    config_probe_args, _ = config_probe_parser.parse_known_args()

    config_values = load_yaml_config(config_probe_args.config)
    config_defaults: Dict[str, Any] = {}
    recognized_keys = {action.dest for action in parser._actions}
    for key, value in config_values.items():
        if key not in recognized_keys or value is None:
            continue
        if key == 'dataset_mixer' and isinstance(value, dict):
            config_defaults[key] = json.dumps(value)
        else:
            config_defaults[key] = value

    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    args = normalize_optional_config_values(args)

    missing_required = [
        field_name
        for field_name in ('model_name_or_path', 'train_data', 'output_dir')
        if getattr(args, field_name) in (None, '')
    ]
    if missing_required:
        parser.error(
            'Missing required settings after applying config and CLI overrides: '
            + ', '.join(missing_required)
        )

    return args


def load_preference_dataset(data_path: str, split: str):
    """Load a dataset from a dataset name, parquet file, or converted parquet directory."""
    path = Path(data_path)

    if path.is_dir():
        split_prefix = 'train' if split == 'train' else 'test'
        parquet_files = sorted(str(file_path) for file_path in path.glob(f'{split_prefix}*.parquet'))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files matching '{split_prefix}*.parquet' were found in {data_path}"
            )
        return load_dataset('parquet', data_files=parquet_files, split='train')

    if path.suffix == '.parquet':
        return load_dataset('parquet', data_files=data_path, split='train')

    return load_dataset(data_path, split=split)


def load_prompt_template(args) -> str:
    """Load the prompt template from disk when provided, otherwise use the inline default."""
    if args.prompt_template_file:
        with open(args.prompt_template_file, 'r', encoding='utf-8') as file:
            return file.read().strip()
    return args.prompt_template


def format_table(table_data: List[List[str]], max_rows: int = 30) -> str:
    """Serialize table rows into the prompt format used during generation."""
    if not table_data:
        return ""

    rows = []
    for row in table_data[:max_rows]:
        rows.append(" | ".join(str(cell) for cell in row))
    return "\n".join(rows)


def create_prompt(query: str, table_data: List[List[str]], prompt_template: str) -> str:
    """Reconstruct the generation prompt when it was not stored in the dataset."""
    return prompt_template.format(query=query, table=format_table(table_data))


def _extract_query(messages: Any) -> str:
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, dict):
            return first.get('content', '')
    return ""


def _ensure_non_empty_token_ids(token_ids: List[int], fallback_token_id: int) -> List[int]:
    return token_ids if token_ids else [fallback_token_id]


def load_datasets_from_mixer(dataset_mixer: Dict[str, float], split: str = 'train'):
    """
    Load and mix multiple datasets.

    Args:
        dataset_mixer: Dictionary mapping dataset names to mixing ratios
        split: Dataset split to load

    Returns:
        Combined dataset
    """
    from datasets import concatenate_datasets

    datasets = []
    for dataset_name, weight in dataset_mixer.items():
        try:
            ds = load_preference_dataset(dataset_name, split=split)
            # Sample according to weight
            if weight < 1.0:
                sample_size = int(len(ds) * weight)
                ds = ds.shuffle(seed=42).select(range(sample_size))
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: Could not load dataset {dataset_name}: {e}")

    if len(datasets) == 0:
        raise ValueError("No datasets loaded successfully!")

    return concatenate_datasets(datasets).shuffle(seed=42)


def normalize_periodic_eval_level(level: str) -> str:
    """Map dataset difficulty labels to the periodic-eval buckets used in logs."""
    normalized = (level or '').strip().lower()
    if normalized == 'easy':
        return 'simple'
    if normalized == 'medium':
        return 'medium'
    if normalized == 'hard':
        return 'complex'
    if normalized == 'calculation':
        return 'calculation'
    return 'unknown'


def load_raw_formula_dataset(data_path: str) -> List[Dict[str, Any]]:
    """Load NL2Formula-style JSON data into a flat list of periodic evaluation samples."""
    with open(data_path, 'r', encoding='utf-8') as file:
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
                'level': normalize_periodic_eval_level(formula_entry.get('Level', '')),
            })

    return samples


def extract_formula_sketch(formula: str) -> str:
    """Extract a function-level sketch from a formula."""
    import re

    sketch = re.sub(r'[A-Z]+\d+', 'Cell', formula)
    sketch = re.sub(r'\b\d+\b', 'Num', sketch)
    return sketch


def normalize_references(references: Any) -> List[str]:
    """Normalize one or many references into a deduplicated list."""
    if isinstance(references, str):
        return [references] if references else []

    normalized = []
    for reference in references or []:
        if reference and reference not in normalized:
            normalized.append(reference)
    return normalized


def compute_generation_metrics(
    predictions: List[str],
    references: List[Any],
    tables: List[List[List[str]]],
    levels: List[str],
    executor: FormulaExecutor,
) -> Dict[str, float]:
    """Compute EM / EA / ESR / FSM and per-level execution match for periodic evaluation."""
    exact_matches = 0
    execution_matches = 0
    sketch_matches = 0
    execution_success = 0
    total = len(predictions)
    execution_match_by_level = {
        'simple': {'matches': 0, 'total': 0},
        'medium': {'matches': 0, 'total': 0},
        'complex': {'matches': 0, 'total': 0},
        'calculation': {'matches': 0, 'total': 0},
    }

    for pred, refs_raw, table, level in zip(predictions, references, tables, levels):
        refs = normalize_references(refs_raw)
        if not refs:
            continue

        if level in execution_match_by_level:
            execution_match_by_level[level]['total'] += 1

        if any(pred.strip() == ref.strip() for ref in refs):
            exact_matches += 1

        pred_result = executor.execute_formula(pred, table)
        if pred_result.status == ExecutionStatus.SUCCESS:
            execution_success += 1

        ref_results = [executor.execute_formula(ref, table) for ref in refs]
        if any(executor.compare_results(pred_result, ref_result) for ref_result in ref_results):
            execution_matches += 1
            if level in execution_match_by_level:
                execution_match_by_level[level]['matches'] += 1

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

    for level_name, bucket in execution_match_by_level.items():
        total_for_level = bucket['total']
        metrics[f'execution_match_{level_name}'] = (
            100.0 * bucket['matches'] / total_for_level if total_for_level else 0.0
        )
        metrics[f'execution_match_{level_name}_count'] = bucket['matches']
        metrics[f'execution_match_{level_name}_total'] = total_for_level

    return metrics


class PeriodicValidationCallback(TrainerCallback):
    """Run small-batch generation evaluation against a fixed raw test subset."""

    def __init__(
        self,
        valid_samples: List[Dict[str, Any]],
        tokenizer,
        executor: FormulaExecutor,
        prompt_template: str,
        output_dir: str,
        valid_steps: int,
        subset_size: int,
        batch_size: int,
        max_new_tokens: int,
        seed: int,
        base_model_name_or_path: str,
        use_shared_reference_adapter: bool,
        ref_model=None,
        policy_adapter_name: str = DEFAULT_POLICY_ADAPTER_NAME,
        reference_adapter_name: str = DEFAULT_REFERENCE_ADAPTER_NAME,
    ):
        self.valid_samples = valid_samples
        self.tokenizer = tokenizer
        self.executor = executor
        self.prompt_template = prompt_template
        self.output_dir = output_dir
        self.valid_steps = valid_steps
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.base_model_name_or_path = base_model_name_or_path
        self.use_shared_reference_adapter = use_shared_reference_adapter
        self.ref_model = ref_model
        self.policy_adapter_name = policy_adapter_name
        self.reference_adapter_name = reference_adapter_name
        self.metrics_path = Path(output_dir) / 'periodic_test_metrics.jsonl'
        self.game_log_path = Path(output_dir) / 'periodic_test_game_log.jsonl'
        self.best_checkpoint_dir = Path(output_dir) / 'best_checkpoint'
        self.fixed_subset_path = Path(output_dir) / 'fixed_test_subset.jsonl'
        self.best_record_path = Path(output_dir) / 'best_periodic_test.json'
        fixed_subset_size = min(self.subset_size, len(self.valid_samples))
        fixed_subset_rng = random.Random(self.seed)
        self.fixed_valid_subset = fixed_subset_rng.sample(self.valid_samples, fixed_subset_size)
        self.best_metric_tuple = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control

        self.fixed_subset_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.fixed_subset_path, 'w', encoding='utf-8') as file:
            for sample in self.fixed_valid_subset:
                file.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(
            f"Fixed periodic evaluation subset prepared: {len(self.fixed_valid_subset)} samples "
            f"written to {self.fixed_subset_path}"
        )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.valid_steps <= 0:
            return control
        if state.global_step == 0 or state.global_step % self.valid_steps != 0:
            return control
        if not state.is_world_process_zero:
            return control

        model = kwargs.get('model')
        if model is None:
            return control

        metrics = self._run_validation(model, state.global_step)
        print(
            f"\n[periodic-eval] step={state.global_step} "
            f"samples={int(metrics['total_samples'])} "
            f"EM={metrics['exact_match']:.2f} "
            f"EX={metrics['execution_accuracy']:.2f} "
            f"ES={metrics['execution_success_rate']:.2f} "
            f"EX[s/m/c/calc]="
            f"{metrics['execution_match_simple']:.2f}/"
            f"{metrics['execution_match_medium']:.2f}/"
            f"{metrics['execution_match_complex']:.2f}/"
            f"{metrics['execution_match_calculation']:.2f}"
        )

        record = {'step': state.global_step, **metrics}
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(record) + '\n')

        if self._is_better(metrics):
            self._save_best_checkpoint(model, state.global_step, metrics)

        for sample_record in self._sample_game_log(model, state.global_step):
            with open(self.game_log_path, 'a', encoding='utf-8') as file:
                file.write(json.dumps(sample_record, ensure_ascii=False) + '\n')

        return control

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, 'module') else model

    def _uses_shared_reference_adapter(self, model) -> bool:
        if not self.policy_adapter_name or not self.reference_adapter_name:
            return False

        peft_config = getattr(model, 'peft_config', None)
        if peft_config is None:
            return False

        return (
            self.policy_adapter_name in peft_config
            and self.reference_adapter_name in peft_config
        )

    def _set_active_adapter(self, model, adapter_name: str):
        if adapter_name and hasattr(model, 'set_adapter'):
            model.set_adapter(adapter_name)

    def _generate_text_batch(self, model, prompts: List[str], do_sample: bool) -> List[str]:
        model_device = next(model.parameters()).device
        encoded = self.tokenizer(prompts, return_tensors='pt', padding=True)
        encoded = {key: value.to(model_device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[:, encoded['input_ids'].shape[1]:]
        return [
            text.strip()
            for text in self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        ]

    def _sample_game_log(self, model, global_step: int) -> List[Dict[str, Any]]:
        sample_count = min(self.batch_size, len(self.fixed_valid_subset))
        if sample_count == 0:
            return []

        rng = random.Random((self.seed * 1009) + global_step)
        sampled = rng.sample(self.fixed_valid_subset, sample_count)
        prompts = [
            create_prompt(sample['query'], sample.get('table', []), self.prompt_template)
            for sample in sampled
        ]

        policy_model = self._unwrap_model(model)
        policy_was_training = policy_model.training
        policy_model.eval()

        self._set_active_adapter(policy_model, self.policy_adapter_name)
        policy_outputs = self._generate_text_batch(policy_model, prompts, do_sample=True)

        reference_outputs = [""] * len(prompts)
        if self._uses_shared_reference_adapter(policy_model):
            self._set_active_adapter(policy_model, self.reference_adapter_name)
            try:
                reference_outputs = self._generate_text_batch(policy_model, prompts, do_sample=True)
            finally:
                self._set_active_adapter(policy_model, self.policy_adapter_name)
        elif self.ref_model is not None:
            reference_model = self._unwrap_model(self.ref_model)
            reference_was_training = reference_model.training
            reference_model.eval()
            reference_outputs = self._generate_text_batch(reference_model, prompts, do_sample=True)
            if reference_was_training:
                reference_model.train()

        if policy_was_training:
            policy_model.train()

        return [
            {
                'step': global_step,
                'prompt': prompt,
                'policy_sample': policy_text,
                'reference_sample': reference_text,
            }
            for prompt, policy_text, reference_text in zip(prompts, policy_outputs, reference_outputs)
        ]

    def _run_validation(self, model, global_step: int) -> Dict[str, float]:
        sampled = self.fixed_valid_subset

        prompts = [
            create_prompt(sample['query'], sample.get('table', []), self.prompt_template)
            for sample in sampled
        ]
        predictions = []
        was_training = model.training
        model.eval()

        for start_idx in range(0, len(prompts), self.batch_size):
            prompt_batch = prompts[start_idx:start_idx + self.batch_size]
            encoded = self.tokenizer(prompt_batch, return_tensors='pt', padding=True)
            encoded = {key: value.to(model.device) for key, value in encoded.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_tokens = outputs[:, encoded['input_ids'].shape[1]:]
            predictions.extend(
                text.strip()
                for text in self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            )

        if was_training:
            model.train()

        return compute_generation_metrics(
            predictions=predictions,
            references=[sample.get('references', [sample.get('formula', '')]) for sample in sampled],
            tables=[sample.get('table', []) for sample in sampled],
            levels=[sample.get('level', 'unknown') for sample in sampled],
            executor=self.executor,
        )

    @staticmethod
    def _metric_tuple(metrics: Dict[str, float]):
        return (
            float(metrics['execution_accuracy']),
            float(metrics['exact_match']),
            float(metrics['execution_success_rate']),
            float(metrics['sketch_match']),
        )

    def _is_better(self, metrics: Dict[str, float]) -> bool:
        metric_tuple = self._metric_tuple(metrics)
        if self.best_metric_tuple is None or metric_tuple > self.best_metric_tuple:
            self.best_metric_tuple = metric_tuple
            return True
        return False

    def _save_best_checkpoint(self, model, global_step: int, metrics: Dict[str, float]):
        unwrapped_model = self._unwrap_model(model)
        if self.best_checkpoint_dir.exists():
            shutil.rmtree(self.best_checkpoint_dir)
        self.best_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(unwrapped_model, 'peft_config'):
            selected_adapters = [self.policy_adapter_name] if self.use_shared_reference_adapter else None
            save_model_without_hub_lookup(
                unwrapped_model,
                str(self.best_checkpoint_dir),
                self.base_model_name_or_path,
                is_main_process=True,
                selected_adapters=selected_adapters,
            )
        else:
            unwrapped_model.save_pretrained(self.best_checkpoint_dir)

        self.tokenizer.save_pretrained(self.best_checkpoint_dir)

        best_record = {
            'step': global_step,
            **metrics,
            'selection_metric_priority': [
                'execution_accuracy',
                'exact_match',
                'execution_success_rate',
                'sketch_match',
            ],
        }
        with open(self.best_record_path, 'w', encoding='utf-8') as file:
            json.dump(best_record, file, ensure_ascii=False, indent=2)

        print(
            f"[periodic-eval] new best checkpoint saved at step={global_step} "
            f"EX={metrics['execution_accuracy']:.2f} EM={metrics['exact_match']:.2f} "
            f"ES={metrics['execution_success_rate']:.2f} "
            f"EX[s/m/c/calc]={metrics['execution_match_simple']:.2f}/"
            f"{metrics['execution_match_medium']:.2f}/"
            f"{metrics['execution_match_complex']:.2f}/"
            f"{metrics['execution_match_calculation']:.2f}"
        )


def preprocess_dataset(examples, executor: FormulaExecutor):
    """
    Preprocess dataset examples.

    This function categorizes generated formulas using execution filtering.
    """
    # Extract data
    real_formulas = [msg[1]['content'] for msg in examples['real']]
    generated_formulas = [msg[1]['content'] for msg in examples['generated']]
    tables = examples.get('table', [[] for _ in real_formulas])

    # Categorize samples
    granularities = []
    for formula_gt, formula_gen, table_data in zip(real_formulas, generated_formulas, tables):
        granularity = executor.categorize_sample(formula_gt, formula_gen, table_data)
        granularities.append(granularity.value)

    examples['granularity'] = granularities
    examples['formula_gt'] = real_formulas
    examples['formula_gen'] = generated_formulas
    examples['table_data'] = tables

    return examples


def tokenize_preference_dataset(
    examples,
    tokenizer,
    max_prompt_length: int,
    max_length: int,
    prompt_template: str,
):
    """Tokenize prompts and paired formulas into the tensors required by FormulaSPIN loss."""
    formula_gt = examples.get('formula_gt') or [msg[1]['content'] for msg in examples['real']]
    formula_gen = examples.get('formula_gen') or [msg[1]['content'] for msg in examples['generated']]
    tables = examples.get('table_data') or examples.get('table', [[] for _ in formula_gt])
    prompts = examples.get('prompt', [])

    prompt_texts = []
    for idx, formula in enumerate(formula_gt):
        prompt_text = prompts[idx] if idx < len(prompts) else None
        if not prompt_text:
            query = _extract_query(examples['real'][idx])
            prompt_text = create_prompt(query, tables[idx], prompt_template)
        prompt_texts.append(prompt_text)

    target_max_length = max(1, max_length - max_prompt_length)
    prompt_tokens = tokenizer(
        prompt_texts,
        add_special_tokens=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    real_tokens = tokenizer(
        formula_gt,
        add_special_tokens=False,
        truncation=True,
        max_length=target_max_length,
    )
    generated_tokens = tokenizer(
        formula_gen,
        add_special_tokens=False,
        truncation=True,
        max_length=target_max_length,
    )

    fallback_token_id = tokenizer.eos_token_id
    if fallback_token_id is None:
        fallback_token_id = tokenizer.pad_token_id
    if fallback_token_id is None:
        raise ValueError("Tokenizer must define an eos_token_id or pad_token_id.")

    return {
        'prompt_input_ids': [
            _ensure_non_empty_token_ids(token_ids, fallback_token_id)
            for token_ids in prompt_tokens['input_ids']
        ],
        'real_input_ids': [
            _ensure_non_empty_token_ids(token_ids, fallback_token_id)
            for token_ids in real_tokens['input_ids']
        ],
        'generated_input_ids': [
            _ensure_non_empty_token_ids(token_ids, fallback_token_id)
            for token_ids in generated_tokens['input_ids']
        ],
        'formula_gt': formula_gt,
        'formula_gen': formula_gen,
        'table_data': tables,
        'granularity': examples.get('granularity', []),
    }


def main():
    args = parse_arguments()
    log_handle = None
    original_stdout = None
    original_stderr = None

    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    global_rank = int(os.environ.get('RANK', str(local_rank)))
    if args.log_file:
        resolved_log_file = resolve_runtime_log_path(args.log_file, global_rank, local_rank)
        log_handle, original_stdout, original_stderr = start_runtime_log(resolved_log_file)

    try:
        prompt_template = load_prompt_template(args)

        configure_nccl_for_rtx40()
        patch_deepspeed_optimizer_compat()

        if not args.bf16 and not args.fp16:
            if args.torch_dtype == 'bfloat16':
                args.bf16 = True
            elif args.torch_dtype == 'float16':
                args.fp16 = True

        if args.tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Set seed
        set_seed(args.seed)

        # Initialize accelerator
        accelerator = Accelerator()

        # Print configuration
        if accelerator.is_local_main_process:
            print("=" * 60)
            print("FormulaSPIN Training Configuration")
            print("=" * 60)
            print(f"Config file: {args.config}")
            print(f"Base model: {args.model_name_or_path}")
            print(f"Adapter: {args.adapter_name_or_path or 'None'}")
            if args.use_shared_reference_adapter:
                print(f"Shared reference adapter: {args.ref_adapter_name_or_path or args.adapter_name_or_path}")
            print(f"Torch dtype: {args.torch_dtype}")
            print(f"Output directory: {args.output_dir}")
            print(f"Beta max: {args.beta_max}")
            print(f"Spin logit scale: {args.spin_logit_scale}")
            print(f"Temperature: {args.temperature}")
            print(f"Learning rate: {args.learning_rate}")
            print(f"Epochs: {args.num_train_epochs}")
            print("=" * 60)

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load tokenizer and model
        tokenizer = load_tokenizer(args.model_name_or_path)
        ref_model = None
        if args.use_shared_reference_adapter:
            if not args.adapter_name_or_path:
                raise ValueError("--use_shared_reference_adapter requires --adapter_name_or_path.")
            if args.ref_model_name_or_path:
                raise ValueError(
                    "--ref_model_name_or_path cannot be combined with --use_shared_reference_adapter. "
                    "Use --ref_adapter_name_or_path instead."
                )

            model = load_shared_reference_policy_model(
                base_model_name_or_path=args.model_name_or_path,
                policy_adapter_name_or_path=args.adapter_name_or_path,
                reference_adapter_name_or_path=args.ref_adapter_name_or_path,
                torch_dtype=args.torch_dtype,
                policy_adapter_name=args.policy_adapter_name,
                reference_adapter_name=args.reference_adapter_name,
                use_flash_attention_2=args.use_flash_attention_2,
            )
        else:
            model = load_causal_lm(
                base_model_name_or_path=args.model_name_or_path,
                adapter_name_or_path=args.adapter_name_or_path,
                torch_dtype=args.torch_dtype,
                merge_adapter=bool(args.adapter_name_or_path),
                use_flash_attention_2=args.use_flash_attention_2,
            )
            if args.ref_model_name_or_path:
                ref_model = load_causal_lm(
                    base_model_name_or_path=args.ref_model_name_or_path,
                    torch_dtype=args.torch_dtype,
                    use_flash_attention_2=args.use_flash_attention_2,
                )

        if args.gradient_checkpointing:
            prepare_model_for_gradient_checkpointing(model)
            if ref_model is not None:
                prepare_model_for_gradient_checkpointing(ref_model)

        # Load datasets
        if args.dataset_mixer:
            dataset_mixer = json.loads(args.dataset_mixer)
            train_dataset = load_datasets_from_mixer(dataset_mixer, split='train')
            if args.eval_data:
                eval_dataset = load_preference_dataset(args.eval_data, split='test')
            else:
                eval_dataset = None
        else:
            train_dataset = load_preference_dataset(args.train_data, split='train')
            if args.eval_data:
                eval_dataset = load_preference_dataset(args.eval_data, split='test')
            else:
                eval_dataset = None

        # Initialize formula executor
        executor = FormulaExecutor(use_xlwings=False)

        valid_samples = None
        if args.valid_steps > 0:
            valid_path = Path(args.valid_data)
            if not valid_path.exists():
                raise FileNotFoundError(f"Periodic evaluation dataset not found: {args.valid_data}")
            valid_samples = load_raw_formula_dataset(args.valid_data)
            if accelerator.is_local_main_process:
                print(
                    f"Loaded {len(valid_samples)} raw periodic-eval samples from {args.valid_data}; "
                    f"periodic evaluation uses a fixed subset of {min(args.valid_subset_size, len(valid_samples))} samples every {args.valid_steps} steps."
                )

        # Preprocess datasets with execution filtering
        if accelerator.is_local_main_process:
            print("\nApplying execution filtering to training data...")

        train_dataset = train_dataset.map(
            lambda x: preprocess_dataset(x, executor),
            batched=True,
            batch_size=100,
            num_proc=args.preprocessing_num_workers,
            desc="Execution filtering (train)"
        )

        if accelerator.is_local_main_process:
            print("Tokenizing FormulaSPIN training pairs...")

        train_columns = list(train_dataset.column_names)
        train_dataset = train_dataset.map(
            lambda x: tokenize_preference_dataset(
                x,
                tokenizer=tokenizer,
                max_prompt_length=args.max_prompt_length,
                max_length=args.max_length,
                prompt_template=prompt_template,
            ),
            batched=True,
            batch_size=100,
            num_proc=args.preprocessing_num_workers,
            remove_columns=train_columns,
            desc="Tokenization (train)"
        )

        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                lambda x: preprocess_dataset(x, executor),
                batched=True,
                batch_size=100,
                num_proc=args.preprocessing_num_workers,
                desc="Execution filtering (eval)"
            )

            eval_columns = list(eval_dataset.column_names)
            eval_dataset = eval_dataset.map(
                lambda x: tokenize_preference_dataset(
                    x,
                    tokenizer=tokenizer,
                    max_prompt_length=args.max_prompt_length,
                    max_length=args.max_length,
                    prompt_template=prompt_template,
                ),
                batched=True,
                batch_size=100,
                num_proc=args.preprocessing_num_workers,
                remove_columns=eval_columns,
                desc="Tokenization (eval)"
            )

        # Configure FormulaSPIN
        formulaspin_config = FormulaSPINConfig(
            beta_max=args.beta_max,
            spin_logit_scale=args.spin_logit_scale,
            loss_type=args.loss_type,
            temperature=args.temperature,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            optim=args.optim,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_epsilon,
            max_grad_norm=args.max_grad_norm,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_total_limit=args.save_total_limit,
            evaluation_strategy=args.evaluation_strategy,
            eval_accumulation_steps=args.eval_accumulation_steps,
            bf16=args.bf16,
            fp16=args.fp16,
            tf32=args.tf32,
            deepspeed=args.deepspeed,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={'use_reentrant': False} if args.gradient_checkpointing else None,
            remove_unused_columns=args.remove_unused_columns,
            ddp_find_unused_parameters=False if args.use_shared_reference_adapter else None,
            report_to=args.report_to,
            seed=args.seed,
        )
        training_args.label_names = []

        # Initialize trainer
        trainer = FormulaSPINTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            config=formulaspin_config,
            policy_adapter_name=args.policy_adapter_name if args.use_shared_reference_adapter else None,
            reference_adapter_name=args.reference_adapter_name if args.use_shared_reference_adapter else None,
            executor=executor,
            data_collator=FormulaSPINDataCollator(tokenizer),
        )

        if valid_samples is not None:
            trainer.add_callback(
                PeriodicValidationCallback(
                    valid_samples=valid_samples,
                    tokenizer=tokenizer,
                    executor=executor,
                    prompt_template=prompt_template,
                    output_dir=args.output_dir,
                    valid_steps=args.valid_steps,
                    subset_size=args.valid_subset_size,
                    batch_size=args.valid_batch_size,
                    max_new_tokens=args.valid_max_new_tokens,
                    seed=args.seed,
                    base_model_name_or_path=args.model_name_or_path,
                    use_shared_reference_adapter=args.use_shared_reference_adapter,
                    ref_model=trainer.ref_model,
                    policy_adapter_name=args.policy_adapter_name,
                    reference_adapter_name=args.reference_adapter_name,
                )
            )

        # Train
        if accelerator.is_local_main_process:
            print("\nStarting training...")

        train_result = trainer.train()

        # Log statistics
        trainer.log_iteration_stats()

        # Save final model
        if accelerator.is_local_main_process:
            print(f"\nSaving model to {args.output_dir}...")

        if args.use_shared_reference_adapter:
            save_model_without_hub_lookup(
                trainer.model,
                args.output_dir,
                args.model_name_or_path,
                accelerator.is_local_main_process,
                selected_adapters=[args.policy_adapter_name],
            )
        elif hasattr(trainer.model, 'peft_config'):
            save_model_without_hub_lookup(
                trainer.model,
                args.output_dir,
                args.model_name_or_path,
                accelerator.is_local_main_process,
            )
        else:
            trainer.save_model(args.output_dir)

        if accelerator.is_local_main_process:
            tokenizer.save_pretrained(args.output_dir)

        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        if accelerator.is_local_main_process:
            print("\nTraining complete!")
    finally:
        stop_runtime_log(log_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
