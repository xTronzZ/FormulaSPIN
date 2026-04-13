"""
Test-Time Consensus Polling

Implements semantic-level consensus voting for formula generation at inference time.
Samples multiple candidates and selects based on execution result agreement.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass

try:
    from .execution_engine import FormulaExecutor, ExecutionResult, ExecutionStatus
except ImportError:
    from execution_engine import FormulaExecutor, ExecutionResult, ExecutionStatus


@dataclass
class ConsensusPollResult:
    """Result of consensus polling"""
    formula: str
    confidence: float
    execution_result: Any
    num_candidates: int
    num_executable: int
    result_distribution: Dict[Any, int]
    consensus_found: bool
    selection_strategy: str


class ConsensusPoller:
    """
    Implements consensus polling for formula generation.

    Samples K candidate formulas at temperature > 1 and selects the best one
    based on execution result voting, as described in Section 3.3 of the paper.
    """

    def __init__(self, executor: Optional[FormulaExecutor] = None):
        """
        Initialize consensus poller.

        Args:
            executor: Formula execution engine
        """
        self.executor = executor or FormulaExecutor(use_xlwings=False)

    def poll(
        self,
        model,
        tokenizer,
        prompt: str,
        table_data: List[List[str]],
        num_candidates: int = 10,
        temperature: float = 1.2,
        max_length: int = 256,
        **generation_kwargs
    ) -> ConsensusPollResult:
        """
        Generate multiple candidates and select the best via consensus polling.

        Algorithm:
        1. Sample K candidates at temperature > 1
        2. Execute all candidates on the table
        3. Vote over execution results (not surface forms)
        4. Among formulas with winning result, select highest probability

        Args:
            model: Language model for generation
            tokenizer: Tokenizer
            prompt: Natural language query
            table_data: Table data for execution
            num_candidates: Number of candidates to sample (K)
            temperature: Sampling temperature (> 1 for diversity)
            max_length: Maximum generation length
            **generation_kwargs: Additional generation parameters

        Returns:
            ConsensusPollResult with selected formula and metadata
        """
        # Step 1: Sample K candidates
        candidates, log_probs = self._sample_candidates(
            model, tokenizer, prompt, num_candidates,
            temperature, max_length, **generation_kwargs
        )

        # Step 2: Execute all candidates
        execution_results = []
        for formula in candidates:
            result = self.executor.execute_formula(formula, table_data)
            execution_results.append(result)

        # Step 3: Vote over execution results
        selected_formula, result_dist, confidence, consensus_found, selection_strategy = self._vote_and_select(
            candidates, execution_results, log_probs
        )

        # Get execution result for selected formula
        selected_result = None
        for formula, result in zip(candidates, execution_results):
            if formula == selected_formula:
                selected_result = result.value if result.status == ExecutionStatus.SUCCESS else None
                break

        # Count executable formulas
        num_executable = sum(
            1 for r in execution_results if r.status == ExecutionStatus.SUCCESS
        )

        return ConsensusPollResult(
            formula=selected_formula,
            confidence=confidence,
            execution_result=selected_result,
            num_candidates=num_candidates,
            num_executable=num_executable,
            result_distribution=result_dist,
            consensus_found=consensus_found,
            selection_strategy=selection_strategy,
        )

    def _sample_candidates(
        self,
        model,
        tokenizer,
        prompt: str,
        num_candidates: int,
        temperature: float,
        max_length: int,
        **generation_kwargs
    ) -> Tuple[List[str], List[float]]:
        """
        Sample multiple candidate formulas.

        Returns:
            Tuple of (formulas, log_probabilities)
        """
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate candidates
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=num_candidates,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                **generation_kwargs
            )

        # Decode formulas
        generated_ids = outputs.sequences[:, inputs['input_ids'].shape[1]:]
        formulas = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Compute log probabilities
        log_probs = self._compute_log_probs(outputs.scores, generated_ids)

        return formulas, log_probs

    def _compute_log_probs(self, scores: Tuple[torch.Tensor], token_ids: torch.Tensor) -> List[float]:
        """
        Compute log probability of each generated sequence.

        Args:
            scores: Tuple of logits for each generation step
            token_ids: Generated token IDs [batch_size, seq_len]

        Returns:
            List of log probabilities, one per sequence
        """
        log_probs = []

        for batch_idx in range(token_ids.size(0)):
            seq_log_prob = 0.0
            for step_idx, step_scores in enumerate(scores):
                if step_idx >= token_ids.size(1):
                    break
                # Get logits for this sequence
                logits = step_scores[batch_idx]
                # Convert to log probabilities
                log_probs_step = torch.log_softmax(logits, dim=-1)
                # Get log prob of the selected token
                token_id = token_ids[batch_idx, step_idx]
                seq_log_prob += log_probs_step[token_id].item()

            log_probs.append(seq_log_prob)

        return log_probs

    def _vote_and_select(
        self,
        formulas: List[str],
        execution_results: List[ExecutionResult],
        log_probs: List[float]
    ) -> Tuple[str, Dict[Any, int], float, bool, str]:
        """
        Vote over execution results and select the best formula.

        Algorithm:
        1. Discard non-executable formulas
        2. Group executable formulas by execution result equivalence classes
        3. If a unique majority result set exists, return its highest-probability formula
        4. If no consensus exists, return the highest-probability executable formula
        5. Final fallback: if all executions fail, return the highest-probability formula

        Returns:
            Tuple of (selected_formula, result_distribution, confidence, consensus_found, selection_strategy)
        """
        executable_indices = [
            idx for idx, result in enumerate(execution_results)
            if result.status == ExecutionStatus.SUCCESS
        ]

        # If no executable formulas, return highest-probability formula.
        if not executable_indices:
            best_idx = np.argmax(log_probs)
            return formulas[best_idx], {}, 0.0, False, 'top1_fallback_all_failed'

        # Group executable formulas by semantic execution equivalence using the
        # executor's tolerant comparison rather than raw Python equality.
        result_groups = []
        for idx in executable_indices:
            matched_group = None
            for group in result_groups:
                if self.executor.compare_results(execution_results[idx], group['representative_result']):
                    matched_group = group
                    break

            if matched_group is None:
                result_groups.append({
                    'representative_result': execution_results[idx],
                    'members': [idx],
                })
            else:
                matched_group['members'].append(idx)

        result_counts = {
            self._make_hashable(group['representative_result'].value): len(group['members'])
            for group in result_groups
        }

        top_vote_count = max(len(group['members']) for group in result_groups)
        winning_groups = [group for group in result_groups if len(group['members']) == top_vote_count]

        # Only treat it as consensus if there is a unique winning result set with
        # support from more than one executable candidate.
        consensus_found = len(winning_groups) == 1 and top_vote_count > 1

        if consensus_found:
            winning_group = winning_groups[0]
            best_idx = max(winning_group['members'], key=lambda idx: log_probs[idx])
            confidence = top_vote_count / len(executable_indices)
            return formulas[best_idx], result_counts, confidence, True, 'majority_result'

        # No consensus: return the highest-probability executable formula.
        best_idx = max(executable_indices, key=lambda idx: log_probs[idx])
        return formulas[best_idx], result_counts, 0.0, False, 'top_executable_fallback'

    @staticmethod
    def _make_hashable(value: Any) -> Any:
        """Convert value to hashable type for use as dictionary key"""
        if isinstance(value, list):
            return tuple(ConsensusPoller._make_hashable(item) for item in value)
        if isinstance(value, tuple):
            return tuple(ConsensusPoller._make_hashable(item) for item in value)
        if isinstance(value, dict):
            return tuple(
                sorted((key, ConsensusPoller._make_hashable(item)) for key, item in value.items())
            )
        if isinstance(value, (str, int, float, bool, type(None))):
            return value

        # For other types, convert to string.
        return str(value)

    def batch_poll(
        self,
        model,
        tokenizer,
        prompts: List[str],
        tables_data: List[List[List[str]]],
        num_candidates: int = 10,
        temperature: float = 1.2,
        max_length: int = 256,
        **generation_kwargs
    ) -> List[ConsensusPollResult]:
        """
        Apply consensus polling to a batch of inputs.

        Args:
            model: Language model
            tokenizer: Tokenizer
            prompts: List of natural language queries
            tables_data: List of table data (one per query)
            num_candidates: Number of candidates per query
            temperature: Sampling temperature
            max_length: Maximum generation length
            **generation_kwargs: Additional generation parameters

        Returns:
            List of ConsensusPollResult, one per input
        """
        results = []
        for prompt, table_data in zip(prompts, tables_data):
            result = self.poll(
                model, tokenizer, prompt, table_data,
                num_candidates, temperature, max_length,
                **generation_kwargs
            )
            results.append(result)
        return results
