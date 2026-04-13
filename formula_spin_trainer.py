"""
FormulaSPIN Trainer

Implements the Formula-Aware Self-Play Fine-Tuning trainer with:
- Execution-based filtering
- Multi-granularity curriculum learning
- Adaptive beta weighting
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from datasets import Dataset
from torch.utils.data import DataLoader
import warnings
from tqdm import tqdm

try:
    from .execution_engine import FormulaExecutor, SampleGranularity
except ImportError:
    from execution_engine import FormulaExecutor, SampleGranularity


@dataclass
class FormulaSPINConfig:
    """Configuration for FormulaSPIN training"""
    beta_max: float = 0.25  # Maximum weight for Fine samples
    spin_logit_scale: float = 1.0  # Scale applied to preference logits inside the SPIN loss
    loss_type: str = "sigmoid"  # Loss type: sigmoid or hinge
    temperature: float = 0.8  # Sampling temperature for generation
    max_length: int = 512
    max_prompt_length: int = 256


@dataclass
class FormulaSPINDataCollator:
    """Build prompt-target sequences and labels for batched FormulaSPIN training."""

    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            return {}

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define a pad_token_id or eos_token_id.")

        real_sequences = []
        real_labels = []
        generated_sequences = []
        generated_labels = []

        for feature in features:
            prompt_ids = feature['prompt_input_ids']
            real_target_ids = feature['real_input_ids']
            generated_target_ids = feature['generated_input_ids']

            real_sequences.append(prompt_ids + real_target_ids)
            generated_sequences.append(prompt_ids + generated_target_ids)

            real_labels.append(
                [self.label_pad_token_id] * len(prompt_ids) + real_target_ids
            )
            generated_labels.append(
                [self.label_pad_token_id] * len(prompt_ids) + generated_target_ids
            )

        return {
            'real_input_ids': self._pad_token_sequences(
                real_sequences,
                pad_token_id,
                left_pad=False,
            ),
            'real_labels': self._pad_token_sequences(
                real_labels,
                self.label_pad_token_id,
                left_pad=False,
            ),
            'generated_input_ids': self._pad_token_sequences(
                generated_sequences,
                pad_token_id,
                left_pad=False,
            ),
            'generated_labels': self._pad_token_sequences(
                generated_labels,
                self.label_pad_token_id,
                left_pad=False,
            ),
            'formula_gt': [feature['formula_gt'] for feature in features],
            'formula_gen': [feature['formula_gen'] for feature in features],
            'table_data': [feature['table_data'] for feature in features],
            'granularity': [feature['granularity'] for feature in features],
        }

    @staticmethod
    def _pad_token_sequences(
        sequences: List[List[int]],
        pad_token_id: int,
        left_pad: bool,
    ) -> torch.Tensor:
        max_length = max(len(sequence) for sequence in sequences)
        padded = torch.full((len(sequences), max_length), pad_token_id, dtype=torch.long)

        for row_idx, sequence in enumerate(sequences):
            seq_tensor = torch.tensor(sequence, dtype=torch.long)
            if left_pad:
                padded[row_idx, max_length - len(sequence):] = seq_tensor
            else:
                padded[row_idx, :len(sequence)] = seq_tensor

        return padded


class FormulaSPINTrainer(Trainer):
    """
    FormulaSPIN Trainer implementing formula-aware self-play fine-tuning.

    This trainer extends the HuggingFace Trainer with:
    1. Execution Filtering: Categorizes samples into Trivial/Coarse/Fine
    2. Adaptive Curriculum: Adjusts focus from semantics to style
    3. Formula-Aware Self-Play Loss: Weighs samples by error type
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str],
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        config: FormulaSPINConfig = None,
        policy_adapter_name: Optional[str] = None,
        reference_adapter_name: Optional[str] = None,
        executor: Optional[FormulaExecutor] = None,
        **kwargs
    ):
        """
        Initialize FormulaSPIN Trainer.

        Args:
            model: Main player model to be optimized
            ref_model: Reference model (opponent player), if None uses model's initial weights
            args: Training arguments
            train_dataset: Training dataset with 'real' and 'generated' columns
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer for the model
            config: FormulaSPIN configuration
            executor: Formula execution engine
        """
        # Initialize parent trainer
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            **kwargs
        )

        # Set config
        self.config = config or FormulaSPINConfig()
        self.policy_adapter_name = policy_adapter_name
        self.reference_adapter_name = reference_adapter_name
        self.uses_shared_reference_adapter = self._uses_shared_reference_adapter()

        # Set reference model
        if self.uses_shared_reference_adapter:
            self.ref_model = None
            self._activate_policy_adapter(self.model)
        else:
            if ref_model is None:
                # Create reference model as a copy of the initial model
                self.ref_model = self._create_reference_model()
            else:
                self.ref_model = ref_model

        # Freeze reference model
        if self.ref_model is not None:
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()

        # Initialize formula executor
        self.executor = executor or FormulaExecutor(use_xlwings=False)

        # Track statistics for adaptive curriculum
        self.iteration_stats = {
            'fine_count': 0,
            'coarse_count': 0,
            'trivial_count': 0,
            'beta_med': 0.0
        }
        self.fixed_beta_med = None
        if train_dataset is not None and 'granularity' in getattr(train_dataset, 'column_names', []):
            dataset_granularity = [
                self._normalize_granularity(value)
                for value in train_dataset['granularity']
            ]
            self.fixed_beta_med = self._compute_adaptive_beta(dataset_granularity)

    def _uses_shared_reference_adapter(self) -> bool:
        """Detect whether policy and reference adapters live on the same base model."""
        if not self.policy_adapter_name or not self.reference_adapter_name:
            return False

        peft_config = getattr(self.model, 'peft_config', None)
        if peft_config is None:
            return False

        return (
            self.policy_adapter_name in peft_config
            and self.reference_adapter_name in peft_config
        )

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """Return the underlying module when Trainer wraps the model for DDP/FSDP."""
        return model.module if hasattr(model, 'module') else model

    def _activate_policy_adapter(self, model: nn.Module):
        """Switch the shared PEFT model back to the trainable policy adapter."""
        if not self.uses_shared_reference_adapter:
            return

        model.set_adapter(self.policy_adapter_name)
        if hasattr(model, 'set_requires_grad'):
            model.set_requires_grad(self.policy_adapter_name, True)
            model.set_requires_grad(self.reference_adapter_name, False)

    def _activate_reference_adapter(self, model: nn.Module):
        """Switch the shared PEFT model to the frozen reference adapter."""
        if not self.uses_shared_reference_adapter:
            return

        model.set_adapter(self.reference_adapter_name)
        if hasattr(model, 'set_requires_grad'):
            model.set_requires_grad(self.policy_adapter_name, False)
            model.set_requires_grad(self.reference_adapter_name, False)

    def _sync_shared_reference_adapter(self):
        """Copy policy adapter weights into the frozen reference adapter in-place."""
        if not self.uses_shared_reference_adapter:
            return

        state_dict = self.model.state_dict()
        policy_token = f".{self.policy_adapter_name}."
        reference_token = f".{self.reference_adapter_name}."
        reference_state = {
            key.replace(policy_token, reference_token): value.detach().clone()
            for key, value in state_dict.items()
            if policy_token in key
        }
        if not reference_state:
            warnings.warn(
                "No policy adapter weights were found when synchronizing the reference adapter.",
                stacklevel=2,
            )
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(reference_state, strict=False)
        if unexpected_keys:
            warnings.warn(
                f"Unexpected keys while synchronizing reference adapter: {unexpected_keys}",
                stacklevel=2,
            )
        if missing_keys:
            warnings.warn(
                f"Missing keys while synchronizing reference adapter: {missing_keys}",
                stacklevel=2,
            )

        self._activate_policy_adapter(self.model)

    def _create_reference_model(self) -> PreTrainedModel:
        """Create a reference model as a copy of the main model"""
        from copy import deepcopy
        ref_model = deepcopy(self.model)
        return ref_model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the FormulaSPIN loss with execution filtering and adaptive curriculum.

        Implements Algorithm 1 from the paper.
        """
        # Extract inputs
        real_ids = inputs.get('real_input_ids')
        real_labels = inputs.get('real_labels')
        generated_ids = inputs.get('generated_input_ids')
        generated_labels = inputs.get('generated_labels')
        formula_gt = inputs.get('formula_gt', [])
        formula_gen = inputs.get('formula_gen', [])
        table_data = inputs.get('table_data', [])
        granularity = inputs.get('granularity', [])

        missing_fields = [
            field_name
            for field_name, value in (
                ('real_input_ids', real_ids),
                ('real_labels', real_labels),
                ('generated_input_ids', generated_ids),
                ('generated_labels', generated_labels),
            )
            if value is None
        ]
        if missing_fields:
            raise ValueError(
                "Missing tokenized FormulaSPIN fields in the training batch: "
                f"{', '.join(missing_fields)}. "
                "Tokenize the preference dataset and use FormulaSPINDataCollator before training."
            )

        if len(granularity) > 0:
            granularity = [self._normalize_granularity(value) for value in granularity]

        # If granularity not precomputed, compute it now
        if len(granularity) == 0 and len(formula_gt) > 0:
            granularity = self._batch_categorize_samples(
                formula_gt, formula_gen, table_data
            )

        active_model = model
        unwrapped_model = self._unwrap_model(active_model)
        if self.uses_shared_reference_adapter:
            self._activate_policy_adapter(unwrapped_model)

        # Score real/generated pairs with one policy forward pass.
        logp_real, logp_gen = self.concatenated_forward(active_model, inputs)

        # Score real/generated pairs with one reference forward pass.
        with torch.no_grad():
            if self.uses_shared_reference_adapter:
                self._activate_reference_adapter(unwrapped_model)
                try:
                    logp_real_ref, logp_gen_ref = self.concatenated_forward(active_model, inputs)
                finally:
                    self._activate_policy_adapter(unwrapped_model)
            elif self.ref_model is not None:
                logp_real_ref, logp_gen_ref = self.concatenated_forward(self.ref_model, inputs)
            else:
                logp_real_ref = torch.zeros_like(logp_real)
                logp_gen_ref = torch.zeros_like(logp_gen)

        # Compute relative preferences
        # r(f; θ, θ_t) = log p_θ(f|q,T) / p_θ_t(f|q,T)
        r_real = logp_real - logp_real_ref
        r_gen = logp_gen - logp_gen_ref

        # Compute adaptive beta_med based on Fine/Coarse ratio for the full
        # self-play iteration when available, otherwise fall back to the batch.
        beta_med = self.fixed_beta_med
        if beta_med is None:
            beta_med = self._compute_adaptive_beta(granularity)

        # Compute sample weights based on granularity
        weights = self._compute_sample_weights(granularity, beta_med, device=logp_real.device)

        # Compute SPIN loss: E[w * ℓ(r(f) - r(f'))]
        logits = r_real - r_gen

        spin_beta = getattr(self.config, 'spin_logit_scale', 0.1)

        if self.config.loss_type == "sigmoid":
            # Sigmoid loss (default)
            loss = -F.logsigmoid(spin_beta * logits)
        elif self.config.loss_type == "hinge":
            # Hinge loss
            loss = torch.relu(1 - spin_beta * logits)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        # Apply sample weights
        active_mask = weights > 0
        if active_mask.any():
            weighted_loss = (loss * weights)[active_mask].mean()
        else:
            weighted_loss = logits.sum() * 0.0

        # Update statistics
        self._update_stats(granularity, beta_med)

        if return_outputs:
            return weighted_loss, {'loss': weighted_loss}
        return weighted_loss

    def _pad_to_length(self, tensor: torch.Tensor, length: int, pad_value: int) -> torch.Tensor:
        """Pad a 2D tensor on the sequence dimension to the requested length."""
        if tensor.shape[1] >= length:
            return tensor

        padded = torch.full(
            (tensor.shape[0], length),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded[:, :tensor.shape[1]] = tensor
        return padded

    def concatenated_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Concatenate real/generated sequences to score them in one forward pass."""
        processing_class = self.processing_class
        if processing_class is None:
            raise ValueError("FormulaSPINTrainer requires a tokenizer or processing_class.")

        pad_token_id = processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = processing_class.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define a pad_token_id or eos_token_id.")

        max_length = max(batch['real_input_ids'].shape[1], batch['generated_input_ids'].shape[1])

        real_input_ids = self._pad_to_length(batch['real_input_ids'], max_length, pad_token_id)
        generated_input_ids = self._pad_to_length(batch['generated_input_ids'], max_length, pad_token_id)
        real_labels = self._pad_to_length(batch['real_labels'], max_length, -100)
        generated_labels = self._pad_to_length(batch['generated_labels'], max_length, -100)

        concatenated_input_ids = torch.cat([real_input_ids, generated_input_ids], dim=0)
        concatenated_labels = torch.cat([real_labels, generated_labels], dim=0)

        return {
            'concatenated_input_ids': concatenated_input_ids,
            'concatenated_attention_mask': concatenated_input_ids.ne(pad_token_id).long(),
            'concatenated_labels': concatenated_labels,
        }

    def _get_batch_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        """Compute sequence log probabilities using masked labels on the response tokens only."""
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits and labels must have matching batch and sequence dimensions.")

        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:].clone()
        loss_mask = shifted_labels.ne(-100)
        shifted_labels[~loss_mask] = 0

        token_logps = torch.gather(
            shifted_logits.log_softmax(dim=-1),
            dim=2,
            index=shifted_labels.unsqueeze(2),
        ).squeeze(2)

        return (token_logps * loss_mask).sum(dim=-1)

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run one forward pass over real and generated sequences concatenated on the batch dimension."""
        concatenated_batch = self.concatenated_inputs(batch)
        real_batch_size = batch['real_input_ids'].shape[0]

        outputs = model(
            input_ids=concatenated_batch['concatenated_input_ids'],
            attention_mask=concatenated_batch['concatenated_attention_mask'],
        )
        all_logits = outputs.logits.to(torch.float32)
        all_logps = self._get_batch_logps(all_logits, concatenated_batch['concatenated_labels'])

        return all_logps[:real_batch_size], all_logps[real_batch_size:]

    def _batch_categorize_samples(
        self,
        formulas_gt: List[str],
        formulas_gen: List[str],
        tables_data: List[List[List[str]]]
    ) -> List[SampleGranularity]:
        """
        Categorize a batch of samples using execution filtering.
        """
        granularities = []
        for formula_gt, formula_gen, table_data in zip(formulas_gt, formulas_gen, tables_data):
            granularity = self.executor.categorize_sample(
                formula_gt, formula_gen, table_data
            )
            granularities.append(granularity)
        return granularities

    @staticmethod
    def _normalize_granularity(value: Union[SampleGranularity, str]) -> SampleGranularity:
        """Convert serialized granularity values back into SampleGranularity enums."""
        if isinstance(value, SampleGranularity):
            return value
        if isinstance(value, str):
            return SampleGranularity(value.lower())
        raise ValueError(f"Unsupported granularity value: {value!r}")

    def _compute_adaptive_beta(self, granularities: List[SampleGranularity]) -> float:
        """
        Compute adaptive beta_med based on Fine/Coarse ratio.

        Implements Equation 2 from the paper:
        β_med^(t) = β_max * |S_fine^(t)| / (|S_fine^(t)| + |S_coarse^(t)|)
        """
        fine_count = sum(1 for g in granularities if g == SampleGranularity.FINE)
        coarse_count = sum(1 for g in granularities if g == SampleGranularity.COARSE)

        if fine_count + coarse_count == 0:
            return 0.0

        beta_med = self.config.beta_max * (fine_count / (fine_count + coarse_count))
        return beta_med

    def _compute_sample_weights(
        self,
        granularities: List[SampleGranularity],
        beta_med: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute sample weights based on granularity.

        - Trivial: weight = 0 (filtered out)
        - Coarse: weight = 1.0 (full weight)
        - Fine: weight = beta_med (adaptive)
        """
        weights = []
        for g in granularities:
            if g == SampleGranularity.TRIVIAL:
                weights.append(0.0)
            elif g == SampleGranularity.COARSE:
                weights.append(1.0)
            elif g == SampleGranularity.FINE:
                weights.append(beta_med)
            else:
                weights.append(0.0)

        return torch.tensor(weights, device=device, dtype=torch.float32)

    def _update_stats(self, granularities: List[SampleGranularity], beta_med: float):
        """Update iteration statistics for logging"""
        for g in granularities:
            if g == SampleGranularity.FINE:
                self.iteration_stats['fine_count'] += 1
            elif g == SampleGranularity.COARSE:
                self.iteration_stats['coarse_count'] += 1
            elif g == SampleGranularity.TRIVIAL:
                self.iteration_stats['trivial_count'] += 1

        self.iteration_stats['beta_med'] = beta_med

    def log_iteration_stats(self):
        """Log statistics for the current iteration"""
        total = sum([
            self.iteration_stats['fine_count'],
            self.iteration_stats['coarse_count'],
            self.iteration_stats['trivial_count']
        ])

        if total > 0:
            print(f"\n=== Iteration Statistics ===")
            print(f"Fine samples: {self.iteration_stats['fine_count']} ({100*self.iteration_stats['fine_count']/total:.1f}%)")
            print(f"Coarse samples: {self.iteration_stats['coarse_count']} ({100*self.iteration_stats['coarse_count']/total:.1f}%)")
            print(f"Trivial samples: {self.iteration_stats['trivial_count']} ({100*self.iteration_stats['trivial_count']/total:.1f}%)")
            print(f"Adaptive β_med: {self.iteration_stats['beta_med']:.4f}")
            print(f"===========================\n")

    def update_reference_model(self):
        """Update reference model to current model weights (for next iteration)"""
        if self.uses_shared_reference_adapter:
            self._sync_shared_reference_adapter()
        elif self.ref_model is not None:
            self.ref_model.load_state_dict(self.model.state_dict())
            self.ref_model.eval()

    def save_iteration_checkpoint(self, iteration: int, output_dir: str):
        """Save checkpoint for a specific iteration"""
        iter_output_dir = os.path.join(output_dir, f"iter_{iteration}")
        os.makedirs(iter_output_dir, exist_ok=True)

        # Save model
        self.save_model(iter_output_dir)

        # Save statistics
        import json
        stats_file = os.path.join(iter_output_dir, "iteration_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(self.iteration_stats, f, indent=2)

        print(f"Saved iteration {iteration} checkpoint to {iter_output_dir}")
