"""
Formula Execution Engine

This module provides functionality to execute Excel formulas and compare their results.
It uses xlwings and xlcalc for formula execution and validation.
"""

import re
import traceback
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

try:
    from .formula_simulator import FormulaSimulationError, FormulaSimulator
except ImportError:
    from formula_simulator import FormulaSimulationError, FormulaSimulator


class ExecutionStatus(Enum):
    """Status of formula execution"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class SampleGranularity(Enum):
    """Granularity level for categorizing generated formulas"""
    TRIVIAL = "trivial"  # Exact match or execution error
    COARSE = "coarse"    # Wrong execution result (semantic error)
    FINE = "fine"        # Correct result, different formula (stylistic difference)


@dataclass
class ExecutionResult:
    """Result of formula execution"""
    status: ExecutionStatus
    value: Any = None
    error: Optional[str] = None


class FormulaExecutor:
    """
    Execute and validate spreadsheet formulas.

    This class provides methods to execute formulas on table data and
    compare results for validation.
    """

    def __init__(self, use_xlwings: bool = False):
        """
        Initialize the formula executor.

        Args:
            use_xlwings: Whether to use xlwings (requires Excel) or xlcalc (pure Python)
        """
        self.use_xlwings = use_xlwings
        self._current_table_data: List[List[str]] = []

        if use_xlwings:
            try:
                import xlwings as xw
                self.xw = xw
            except ImportError:
                print("Warning: xlwings not available, falling back to xlcalc")
                self.use_xlwings = False

        if not self.use_xlwings:
            try:
                from xlcalculator import ModelCompiler, Evaluator
                self.ModelCompiler = ModelCompiler
                self.Evaluator = Evaluator
            except ImportError:
                print("Warning: xlcalculator not available. Install with: pip install xlcalculator")

    def parse_table(self, table_data: List[List[str]]) -> Dict[str, Any]:
        """
        Parse table data into a cell reference dictionary.

        Args:
            table_data: 2D list representing table data

        Returns:
            Dictionary mapping cell references to values
        """
        cells = {}
        for row_idx, row in enumerate(table_data):
            for col_idx, value in enumerate(row):
                # Convert column index to letter (0->A, 1->B, etc.)
                col_letter = self._col_idx_to_letter(col_idx)
                cell_ref = f"{col_letter}{row_idx + 1}"
                cells[cell_ref] = value
        return cells

    @staticmethod
    def _col_idx_to_letter(idx: int) -> str:
        """Convert column index to Excel column letter"""
        result = ""
        while idx >= 0:
            result = chr(idx % 26 + ord('A')) + result
            idx = idx // 26 - 1
        return result

    def execute_formula(self, formula: str, table_data: List[List[str]],
                       timeout: int = 5) -> ExecutionResult:
        """
        Execute a formula on the given table data.

        Args:
            formula: Excel formula string (e.g., "=SUM(A1:A5)")
            table_data: 2D list of table data
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult containing status and value/error
        """
        try:
            # Ensure formula starts with =
            if not formula.startswith('='):
                formula = '=' + formula

            self._current_table_data = table_data

            # Parse table into cell references
            cells = self.parse_table(table_data)

            # Execute using available backend
            if self.use_xlwings:
                result = self._execute_with_xlwings(formula, cells, timeout)
            else:
                result = self._execute_with_xlcalc(formula, cells, timeout)

            return result

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=str(e)
            )

    def _execute_with_xlcalc(self, formula: str, cells: Dict[str, Any],
                            timeout: int) -> ExecutionResult:
        """Execute formula using xlcalculator (pure Python)"""
        try:
            simulator = FormulaSimulator(self._current_table_data)
            result_value = simulator.evaluate(formula)

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                value=result_value
            )
        except FormulaSimulationError as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"xlcalc error: {str(e)}"
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"xlcalc error: {str(e)}"
            )

    def _execute_with_xlwings(self, formula: str, cells: Dict[str, Any],
                             timeout: int) -> ExecutionResult:
        """Execute formula using xlwings (requires Excel)"""
        try:
            # Create a new workbook
            wb = self.xw.Book()
            sheet = wb.sheets[0]

            # Populate cells
            for cell_ref, value in cells.items():
                sheet.range(cell_ref).value = value

            # Execute formula
            result_cell = 'Z1'  # Use a cell far away from data
            sheet.range(result_cell).formula = formula
            result_value = sheet.range(result_cell).value

            # Clean up
            wb.close()

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                value=result_value
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"xlwings error: {str(e)}"
            )

    def _simple_evaluate(self, formula: str, cells: Dict[str, Any]) -> Any:
        """
        Simple formula evaluation for basic cases.
        This is a placeholder - real implementation should use xlcalculator or xlwings.
        """
        # Remove leading =
        formula = formula.lstrip('=')

        # Handle simple SUM
        if formula.upper().startswith('SUM('):
            match = re.match(r'SUM\(([A-Z]+)(\d+):([A-Z]+)(\d+)\)', formula.upper())
            if match:
                col1, row1, col2, row2 = match.groups()
                values = []
                for row in range(int(row1), int(row2) + 1):
                    cell_ref = f"{col1}{row}"
                    if cell_ref in cells:
                        try:
                            values.append(float(cells[cell_ref]))
                        except (ValueError, TypeError):
                            pass
                return sum(values)

        # Handle simple cell reference
        if formula in cells:
            return cells[formula]

        # If we can't parse it, raise an error
        raise ValueError(f"Cannot evaluate formula: {formula}")

    def compare_results(self, result1: ExecutionResult, result2: ExecutionResult,
                       tolerance: float = 1e-6) -> bool:
        """
        Compare two execution results for equality.

        Args:
            result1: First execution result
            result2: Second execution result
            tolerance: Numerical tolerance for float comparison

        Returns:
            True if results are equal, False otherwise
        """
        # If either failed, they're not equal
        if result1.status != ExecutionStatus.SUCCESS or result2.status != ExecutionStatus.SUCCESS:
            return False

        # Compare values
        v1, v2 = result1.value, result2.value

        def normalize(value):
            if isinstance(value, list) and len(value) == 1:
                return normalize(value[0])
            if isinstance(value, str):
                stripped = value.strip().replace(',', '')
                try:
                    return float(stripped)
                except ValueError:
                    return value.strip().lower()
            if isinstance(value, list):
                return [normalize(item) for item in value]
            return value

        v1 = normalize(v1)
        v2 = normalize(v2)

        # Handle None
        if v1 is None or v2 is None:
            return v1 == v2

        # Handle numerical comparison with tolerance
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            return abs(v1 - v2) < tolerance

        # Handle string comparison
        if isinstance(v1, str) and isinstance(v2, str):
            return v1 == v2

        # Default comparison
        return v1 == v2

    def categorize_sample(self, formula_gt: str, formula_gen: str,
                         table_data: List[List[str]]) -> SampleGranularity:
        """
        Categorize a generated formula sample based on execution results.

        This implements the Execution Filtering mechanism from FormulaSPIN.

        Args:
            formula_gt: Ground truth formula
            formula_gen: Generated formula
            table_data: Table data for execution

        Returns:
            SampleGranularity indicating the error type
        """
        # Exact match -> Trivial (zero gradient)
        if formula_gt.strip() == formula_gen.strip():
            return SampleGranularity.TRIVIAL

        # Execute both formulas
        result_gt = self.execute_formula(formula_gt, table_data)
        result_gen = self.execute_formula(formula_gen, table_data)

        # Execution error -> Trivial (inconsistent signal)
        if result_gen.status != ExecutionStatus.SUCCESS:
            return SampleGranularity.TRIVIAL

        # Compare execution results
        results_match = self.compare_results(result_gt, result_gen)

        if results_match:
            # Same result, different formula -> Fine (stylistic difference)
            return SampleGranularity.FINE
        else:
            # Different result -> Coarse (semantic error)
            return SampleGranularity.COARSE

    def batch_categorize(self, samples: List[Tuple[str, str, List[List[str]]]]) -> Dict[str, List[int]]:
        """
        Categorize a batch of samples.

        Args:
            samples: List of (formula_gt, formula_gen, table_data) tuples

        Returns:
            Dictionary mapping granularity to list of sample indices
        """
        categorized = {
            SampleGranularity.TRIVIAL: [],
            SampleGranularity.COARSE: [],
            SampleGranularity.FINE: []
        }

        for idx, (formula_gt, formula_gen, table_data) in enumerate(samples):
            category = self.categorize_sample(formula_gt, formula_gen, table_data)
            categorized[category].append(idx)

        return categorized
