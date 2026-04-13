"""
FormulaSPIN: Self-Play Fine-Tuning for Natural Language to Spreadsheet Formula Generation

This package implements the FormulaSPIN framework for improving spreadsheet formula generation
through self-play fine-tuning with execution-based feedback.
"""

__version__ = "1.0.0"

from .execution_engine import FormulaExecutor, SampleGranularity
from .formula_spin_trainer import FormulaSPINTrainer
from .consensus_polling import ConsensusPoller

__all__ = [
    "FormulaExecutor",
    "FormulaSPINTrainer",
]
