import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None


class FormulaSimulationError(ValueError):
    pass


@dataclass
class AggregationSpec:
    op: str
    value: Any


class FormulaSimulator:
    def __init__(self, table_data: List[List[str]]):
        if len(table_data) < 2:
            raise FormulaSimulationError("Table data is too short")

        self.raw_table = table_data
        self.headers = table_data[1][1:] if len(table_data) > 1 else []
        self.data_rows = [row[1:] for row in table_data[2:]]
        self.num_rows = len(self.data_rows)
        self.num_cols = len(self.headers)
        self._tokens: List[Tuple[str, Any]] = []
        self._position = 0
        self._env: Dict[str, Any] = {}
        self._last_let_bindings: Dict[str, Any] = {}

    def evaluate(self, formula: str) -> Any:
        text = self._sanitize_formula_text(formula)

        self._tokens = self._tokenize(text)
        self._position = 0
        self._env = {}
        self._last_let_bindings = {}
        result = self._parse_expression()

        while self._peek()[0] == ",":
            trailing_bindings = dict(self._last_let_bindings)
            self._advance()
            if self._peek()[0] == "EOF":
                break
            self._env.update(trailing_bindings)
            result = self._parse_expression()

        while self._peek()[0] == ")":
            self._advance()

        if self._peek()[0] != "EOF":
            raise FormulaSimulationError(f"Unexpected trailing token: {self._peek()}")

        return result

    def _sanitize_formula_text(self, formula: str) -> str:
        text = formula.strip()
        if not text:
            return text

        first_line = text.splitlines()[0].strip()
        for marker in ("###", "```", "<!DOCTYPE", "<html>"):
            if marker in first_line:
                first_line = first_line.split(marker, 1)[0].rstrip()

        if first_line.startswith("="):
            first_line = first_line[1:].strip()

        return self._balance_delimiters(first_line)

    def _balance_delimiters(self, text: str) -> str:
        result: List[str] = []
        paren_depth = 0
        brace_depth = 0
        in_string = False

        for char in text:
            if char == '"':
                in_string = not in_string
                result.append(char)
                continue

            if not in_string:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    if paren_depth == 0:
                        continue
                    paren_depth -= 1
                elif char == '{':
                    brace_depth += 1
                elif char == '}':
                    if brace_depth == 0:
                        continue
                    brace_depth -= 1

            result.append(char)

        if paren_depth > 0:
            result.extend(')' * paren_depth)
        if brace_depth > 0:
            result.extend('}' * brace_depth)

        return ''.join(result)

    def _tokenize(self, text: str) -> List[Tuple[str, Any]]:
        tokens: List[Tuple[str, Any]] = []
        index = 0

        while index < len(text):
            char = text[index]

            if char.isspace():
                index += 1
                continue

            if char in "(),:+-*/{}":
                tokens.append((char, char))
                index += 1
                continue

            if char == "<" and index + 1 < len(text) and text[index + 1] == ">":
                tokens.append(("OP", "<>"))
                index += 2
                continue

            if char in "<>=" and index + 1 < len(text) and text[index + 1] == "=":
                tokens.append(("OP", text[index:index + 2]))
                index += 2
                continue

            if char in "<>=":
                tokens.append(("OP", char))
                index += 1
                continue

            if char == '"':
                end = index + 1
                value = []
                while end < len(text):
                    if text[end] == '"':
                        break
                    value.append(text[end])
                    end += 1
                if end >= len(text):
                    raise FormulaSimulationError("Unterminated string literal")
                tokens.append(("STRING", "".join(value)))
                index = end + 1
                continue

            if char.isdigit() or (char == '.' and index + 1 < len(text) and text[index + 1].isdigit()):
                end = index + 1
                while end < len(text) and (text[end].isdigit() or text[end] == "."):
                    end += 1
                raw_number = text[index:end]
                value = float(raw_number) if "." in raw_number else int(raw_number)
                tokens.append(("NUMBER", value))
                index = end
                continue

            if char.isalpha() or char == '_':
                end = index + 1
                while end < len(text) and (text[end].isalnum() or text[end] == '_'):
                    end += 1
                tokens.append(("IDENT", text[index:end]))
                index = end
                continue

            raise FormulaSimulationError(f"Unsupported token starting at: {text[index:index + 16]}")

        tokens.append(("EOF", None))
        return tokens

    def _peek(self, offset: int = 0) -> Tuple[str, Any]:
        position = min(self._position + offset, len(self._tokens) - 1)
        return self._tokens[position]

    def _advance(self) -> Tuple[str, Any]:
        token = self._tokens[self._position]
        self._position += 1
        return token

    def _expect(self, token_type: str, value: Optional[str] = None) -> Tuple[str, Any]:
        token = self._advance()
        if token[0] != token_type:
            raise FormulaSimulationError(f"Expected {token_type}, got {token}")
        if value is not None and token[1] != value:
            raise FormulaSimulationError(f"Expected {value}, got {token[1]}")
        return token

    def _parse_expression(self) -> Any:
        return self._parse_addition()

    def _parse_addition(self) -> Any:
        left = self._parse_multiplication()
        while self._peek()[0] in {"+", "-"}:
            operator = self._advance()[0]
            right = self._parse_multiplication()
            left = self._arithmetic_binary_op(left, right, operator)
        return left

    def _parse_multiplication(self) -> Any:
        left = self._parse_comparison()
        while self._peek()[0] in {"*", "/"}:
            operator = self._advance()[0]
            right = self._parse_comparison()
            left = self._arithmetic_binary_op(left, right, operator)
        return left

    def _parse_comparison(self) -> Any:
        left = self._parse_primary()
        if self._peek()[0] == "OP":
            operator = self._advance()[1]
            right = self._parse_primary()
            return self._compare(left, right, operator)
        return left

    def _parse_primary(self) -> Any:
        token_type, token_value = self._peek()

        if token_type == "-":
            self._advance()
            value = self._parse_primary()
            return self._arithmetic_binary_op(0, value, "-")

        if token_type == "NUMBER":
            self._advance()
            return token_value

        if token_type == "STRING":
            self._advance()
            return token_value

        if token_type == "(":
            self._advance()
            value = self._parse_expression()
            self._expect(")")
            return value

        if token_type == "{":
            self._advance()
            values: List[Any] = []
            if self._peek()[0] != "}":
                while True:
                    values.append(self._parse_expression())
                    if self._peek()[0] != ",":
                        break
                    self._advance()
            self._expect("}")
            return values

        if token_type == "IDENT":
            name = self._advance()[1]
            if self._peek()[0] == "(":
                return self._parse_function_call(name)
            if self._is_cell_ref(name) and self._peek()[0] == ":":
                self._advance()
                end_token = self._expect("IDENT")[1]
                return self._resolve_range(name, end_token)
            if name in self._env:
                return self._env[name]
            if self._is_cell_ref(name):
                return self._resolve_ref(name)
            raise FormulaSimulationError(f"Unknown identifier: {name}")

        raise FormulaSimulationError(f"Unexpected token: {(token_type, token_value)}")

    def _parse_function_call(self, name: str) -> Any:
        self._expect("(")
        upper_name = name.upper()

        if upper_name == "LET":
            result = self._eval_let()
            self._expect(")")
            return result

        arguments = self._parse_function_arguments()
        self._expect(")")
        return self._call_function(upper_name, arguments)

    def _parse_function_arguments(self) -> List[Any]:
        arguments: List[Any] = []
        if self._peek()[0] == ")":
            return arguments

        while True:
            if self._peek()[0] in {",", ")"}:
                arguments.append(None)
            else:
                arguments.append(self._parse_expression())

            if self._peek()[0] != ",":
                break

            self._advance()
            if self._peek()[0] == ")":
                arguments.append(None)
                break

        return arguments

    def _eval_let(self) -> Any:
        local_env = dict(self._env)
        new_bindings: Dict[str, Any] = {}
        try:
            result: Any = None
            while True:
                next_token = self._peek()
                next_next = self._peek(1)
                if next_token[0] == "IDENT" and next_next[0] == ",":
                    name = self._advance()[1]
                    self._expect(",")
                    value = self._parse_expression()
                    self._env[name] = value
                    new_bindings[name] = value
                    if self._peek()[0] == ",":
                        self._advance()
                        lookahead = self._peek()
                        if not (lookahead[0] == "IDENT" and self._peek(1)[0] == ","):
                            result = self._parse_expression()
                            while self._peek()[0] == ",":
                                self._advance()
                                if self._peek()[0] == ")":
                                    break
                                result = self._parse_expression()
                            return result
                        continue
                    result = value
                    break

                result = self._parse_expression()
                while self._peek()[0] == ",":
                    self._advance()
                    if self._peek()[0] == ")":
                        break
                    result = self._parse_expression()
                return result

            return result
        finally:
            self._last_let_bindings = new_bindings
            self._env = local_env

    def _call_function(self, name: str, args: Sequence[Any]) -> Any:
        if name == "FILTER":
            return self._fn_filter(*args)
        if name == "UNIQUE":
            return self._fn_unique(*args)
        if name == "CHOOSECOLS":
            return self._fn_choosecols(*args)
        if name == "ROWS":
            return self._fn_rows(*args)
        if name == "SUM":
            return self._fn_sum(*args)
        if name == "MAX":
            return self._fn_max(*args)
        if name == "MIN":
            return self._fn_min(*args)
        if name == "AVERAGE":
            return self._fn_average(*args)
        if name == "SUMIFS":
            return self._fn_sumifs(*args)
        if name == "MAXIFS":
            return self._fn_maxifs(*args)
        if name == "MINIFS":
            return self._fn_minifs(*args)
        if name == "AVERAGEIFS":
            return self._fn_averageifs(*args)
        if name == "SUMX":
            return AggregationSpec("sum", args[0])
        if name == "MAXX":
            return AggregationSpec("max", args[0])
        if name == "MINX":
            return AggregationSpec("min", args[0])
        if name == "AVERAGEX":
            return AggregationSpec("average", args[0])
        if name == "DCOUNTX":
            return AggregationSpec("count", args[0])
        if name == "SUMMARIZE":
            return self._fn_summarize(*args)
        if name == "HSTACK":
            return self._fn_hstack(*args)
        if name == "SORT":
            return self._fn_sort(*args)
        if name == "SORTBY":
            return self._fn_sortby(*args)
        if name == "TAKE":
            return self._fn_take(*args)
        if name == "XLOOKUP":
            return self._fn_xlookup(*args)
        if name == "INDEX":
            return self._fn_index(*args)
        if name == "ISNA":
            return self._fn_isna(*args)
        if name == "ISBLANK":
            return self._fn_isblank(*args)
        if name == "NOT":
            return self._fn_not(*args)
        if name == "CHOOSE":
            return self._fn_choose(*args)
        if name == "LOWER":
            return self._fn_lower(*args)
        if name == "RIGHT":
            return self._fn_right(*args)
        if name == "SEARCH":
            return self._fn_search(*args)
        if name == "IFERROR":
            return self._fn_iferror(*args)
        if name == "YEAR":
            return self._fn_year(*args)
        if name == "MONTH":
            return self._fn_month(*args)
        if name == "DAY":
            return self._fn_day(*args)
        if name == "SEQUENCE":
            return self._fn_sequence(*args)
        if name == "COUNTX":
            return AggregationSpec("count", args[0])
        raise FormulaSimulationError(f"Unsupported function: {name}")

    def _resolve_ref(self, ref: str) -> Any:
        col_name, row_number = self._split_ref(ref)
        column_index = self._col_to_index(col_name)

        if row_number == 1:
            return [self.data_rows[row_index][column_index] for row_index in range(self.num_rows)]

        data_index = row_number - 2
        if data_index < 0 or data_index >= self.num_rows:
            raise FormulaSimulationError(f"Row out of bounds: {ref}")
        return self.data_rows[data_index][column_index]

    def _resolve_range(self, start_ref: str, end_ref: str) -> Any:
        start_col, start_row = self._split_ref(start_ref)
        end_col, end_row = self._split_ref(end_ref)
        start_col_index = self._col_to_index(start_col)
        end_col_index = self._col_to_index(end_col)

        if start_row == 1 and end_row == 1:
            if start_col_index == end_col_index:
                return [row[start_col_index] for row in self.data_rows]
            return [row[start_col_index:end_col_index + 1] for row in self.data_rows]

        start_data_row = max(start_row, 2) - 2
        end_data_row = max(end_row, 2) - 2
        rows = self.data_rows[start_data_row:end_data_row + 1]
        if start_col_index == end_col_index:
            return [row[start_col_index] for row in rows]
        return [row[start_col_index:end_col_index + 1] for row in rows]

    def _binary_op(self, left: Any, right: Any, operation) -> Any:
        if self._is_vector(left) and self._is_vector(right):
            return [operation(a, b) for a, b in zip(left, right)]
        if self._is_vector(left):
            return [operation(item, right) for item in left]
        if self._is_vector(right):
            return [operation(left, item) for item in right]
        return operation(left, right)

    def _arithmetic_binary_op(self, left: Any, right: Any, operator: str) -> Any:
        def apply(a: Any, b: Any) -> float:
            left_number = self._coerce_number(a)
            right_number = self._coerce_number(b)
            if left_number is None or right_number is None:
                raise FormulaSimulationError(
                    f"Arithmetic operator {operator} requires numeric operands, got {a!r} and {b!r}"
                )
            if operator == "+":
                return left_number + right_number
            if operator == "-":
                return left_number - right_number
            if operator == "*":
                return left_number * right_number
            if operator == "/":
                if right_number == 0:
                    raise FormulaSimulationError("Division by zero")
                return left_number / right_number
            raise FormulaSimulationError(f"Unsupported arithmetic operator: {operator}")

        return self._binary_op(left, right, apply)

    def _compare(self, left: Any, right: Any, operator: str) -> Any:
        def compare_values(a: Any, b: Any) -> bool:
            a_norm = self._normalize_scalar(a)
            b_norm = self._normalize_scalar(b)
            if operator == "=":
                if isinstance(a_norm, str) and isinstance(b_norm, str):
                    return a_norm.lower() == b_norm.lower()
                return a_norm == b_norm
            if operator == "<>":
                if isinstance(a_norm, str) and isinstance(b_norm, str):
                    return a_norm.lower() != b_norm.lower()
                return a_norm != b_norm
            if operator == ">":
                if isinstance(a_norm, str) and isinstance(b_norm, (int, float)):
                    number = self._coerce_number(a_norm)
                    return False if number is None else number > b_norm
                if isinstance(b_norm, str) and isinstance(a_norm, (int, float)):
                    number = self._coerce_number(b_norm)
                    return False if number is None else a_norm > number
                return a_norm > b_norm
            if operator == "<":
                if isinstance(a_norm, str) and isinstance(b_norm, (int, float)):
                    number = self._coerce_number(a_norm)
                    return False if number is None else number < b_norm
                if isinstance(b_norm, str) and isinstance(a_norm, (int, float)):
                    number = self._coerce_number(b_norm)
                    return False if number is None else a_norm < number
                return a_norm < b_norm
            if operator == ">=":
                if isinstance(a_norm, str) and isinstance(b_norm, (int, float)):
                    number = self._coerce_number(a_norm)
                    return False if number is None else number >= b_norm
                if isinstance(b_norm, str) and isinstance(a_norm, (int, float)):
                    number = self._coerce_number(b_norm)
                    return False if number is None else a_norm >= number
                return a_norm >= b_norm
            if operator == "<=":
                if isinstance(a_norm, str) and isinstance(b_norm, (int, float)):
                    number = self._coerce_number(a_norm)
                    return False if number is None else number <= b_norm
                if isinstance(b_norm, str) and isinstance(a_norm, (int, float)):
                    number = self._coerce_number(b_norm)
                    return False if number is None else a_norm <= number
                return a_norm <= b_norm
            raise FormulaSimulationError(f"Unsupported operator: {operator}")

        return self._binary_op(left, right, compare_values)

    def _fn_filter(self, array: Any, include: Any, *extra_args: Any) -> Any:
        mask = self._coerce_mask(include, self._row_count(array))
        if_empty = None

        for extra in extra_args:
            extra_mask = self._coerce_mask(extra, self._row_count(array))
            if extra_mask is not None:
                mask = [bool(current) and bool(other) for current, other in zip(mask, extra_mask)]
            else:
                if_empty = extra

        if self._is_table(array):
            result = [row for row, keep in zip(array, mask) if keep]
            return result if result else ([] if if_empty is None else if_empty)
        if self._is_vector(array):
            result = [value for value, keep in zip(array, mask) if keep]
            return result if result else ([] if if_empty is None else if_empty)
        return [array] if mask and mask[0] else ([] if if_empty is None else if_empty)

    def _fn_unique(self, array: Any) -> Any:
        if self._is_table(array):
            seen = set()
            result = []
            for row in array:
                key = tuple(row)
                if key not in seen:
                    seen.add(key)
                    result.append(row)
            return result
        values = self._ensure_vector(array)
        seen = set()
        result = []
        for value in values:
            key = str(value).lower() if isinstance(value, str) else value
            if key not in seen:
                seen.add(key)
                result.append(value)
        return result

    def _fn_choosecols(self, array: Any, *indexes: Any) -> Any:
        if array == []:
            return []
        selected_indexes = [int(self._normalize_scalar(index)) - 1 for index in indexes]
        if self._is_table(array):
            rows = [[row[index] for index in selected_indexes] for row in array]
            if len(selected_indexes) == 1:
                return [row[0] for row in rows]
            return rows
        vector = self._ensure_vector(array)
        if len(selected_indexes) == 1:
            return vector
        return [[item for _ in selected_indexes] for item in vector]

    def _fn_rows(self, array: Any) -> int:
        return self._row_count(array)

    def _fn_sum(self, array: Any) -> float:
        return sum(self._numeric_values(array))

    def _fn_max(self, array: Any) -> Any:
        values = self._numeric_values(array)
        return max(values) if values else None

    def _fn_min(self, array: Any) -> Any:
        values = self._numeric_values(array)
        return min(values) if values else None

    def _fn_average(self, array: Any) -> Optional[float]:
        values = self._numeric_values(array)
        return (sum(values) / len(values)) if values else None

    def _fn_sumifs(self, sum_range: Any, *criteria_args: Any) -> float:
        matched = self._apply_criteria(self._ensure_vector(sum_range), criteria_args)
        return sum(self._coerce_number(value) for value in matched if self._coerce_number(value) is not None)

    def _fn_maxifs(self, value_range: Any, *criteria_args: Any) -> Any:
        matched = [self._coerce_number(value) for value in self._apply_criteria(self._ensure_vector(value_range), criteria_args)]
        matched = [value for value in matched if value is not None]
        return max(matched) if matched else None

    def _fn_minifs(self, value_range: Any, *criteria_args: Any) -> Any:
        matched = [self._coerce_number(value) for value in self._apply_criteria(self._ensure_vector(value_range), criteria_args)]
        matched = [value for value in matched if value is not None]
        return min(matched) if matched else None

    def _fn_averageifs(self, value_range: Any, *criteria_args: Any) -> Any:
        matched = [self._coerce_number(value) for value in self._apply_criteria(self._ensure_vector(value_range), criteria_args)]
        matched = [value for value in matched if value is not None]
        return (sum(matched) / len(matched)) if matched else None

    def _fn_summarize(self, group_range: Any, *aggregations: AggregationSpec) -> List[List[Any]]:
        if not aggregations or any(not isinstance(aggregation, AggregationSpec) for aggregation in aggregations):
            raise FormulaSimulationError("SUMMARIZE expects one or more aggregation specs")

        group_table = self._to_table(group_range)
        aggregation_vectors = [self._ensure_vector(aggregation.value) for aggregation in aggregations]
        grouped: Dict[Tuple[Any, ...], List[List[Any]]] = {}
        order: List[Tuple[Any, ...]] = []

        for row_index, group_row in enumerate(group_table):
            key = tuple(group_row)
            if key not in grouped:
                grouped[key] = [[] for _ in aggregations]
                order.append(key)
            for agg_index, vector in enumerate(aggregation_vectors):
                if row_index < len(vector):
                    grouped[key][agg_index].append(vector[row_index])

        rows: List[List[Any]] = []
        for key in order:
            result_row = list(key)
            for aggregation, values in zip(aggregations, grouped[key]):
                numeric_values = [self._coerce_number(value) for value in values]
                numeric_values = [value for value in numeric_values if value is not None]
                if aggregation.op == "sum":
                    aggregate_value = sum(numeric_values)
                elif aggregation.op == "average":
                    aggregate_value = sum(numeric_values) / len(numeric_values) if numeric_values else None
                elif aggregation.op == "count":
                    aggregate_value = len(values)
                elif aggregation.op == "min":
                    aggregate_value = min(numeric_values) if numeric_values else None
                elif aggregation.op == "max":
                    aggregate_value = max(numeric_values) if numeric_values else None
                else:
                    raise FormulaSimulationError(f"Unsupported aggregation: {aggregation.op}")
                result_row.append(aggregate_value)
            rows.append(result_row)
        return rows

    def _fn_hstack(self, *arrays: Any) -> List[List[Any]]:
        normalized = [self._to_table(array) for array in arrays]
        row_count = len(normalized[0]) if normalized else 0
        for table in normalized:
            if len(table) != row_count:
                raise FormulaSimulationError("HSTACK row count mismatch")
        result = []
        for row_index in range(row_count):
            row: List[Any] = []
            for table in normalized:
                row.extend(table[row_index])
            result.append(row)
        return result

    def _fn_sort(self, array: Any, sort_index: Any = 1, order: Any = 1) -> Any:
        sort_order = int(self._normalize_scalar(order))
        reverse = sort_order == -1
        if self._is_table(array):
            if not array:
                return []
            index = int(self._normalize_scalar(sort_index)) - 1
            if index < 0 or index >= len(array[0]):
                raise FormulaSimulationError("SORT column index out of range")
            return sorted(array, key=lambda row: self._normalize_scalar(row[index]), reverse=reverse)
        return sorted(self._ensure_vector(array), key=self._normalize_scalar, reverse=reverse)

    def _fn_sortby(self, array: Any, by_array: Any, order: Any = 1) -> Any:
        sort_order = int(self._normalize_scalar(order))
        reverse = sort_order == -1
        base = self._to_table(array)
        if not base:
            return []
        by_values = self._ensure_vector(by_array)
        pairs = list(zip(base, by_values))
        pairs.sort(key=lambda item: self._normalize_scalar(item[1]), reverse=reverse)
        rows = [row for row, _ in pairs]
        if self._is_table(array):
            return rows
        return [row[0] for row in rows]

    def _fn_take(self, array: Any, count: Any) -> Any:
        take_count = int(self._normalize_scalar(count))
        if self._is_table(array):
            return array[:take_count]
        return self._ensure_vector(array)[:take_count]

    def _fn_xlookup(self, lookup_value: Any, lookup_array: Any, return_array: Any) -> Any:
        lookup_values = self._ensure_vector(lookup_array)
        return_values = self._ensure_vector(return_array)
        target = self._normalize_scalar(lookup_value)
        for candidate, result in zip(lookup_values, return_values):
            normalized_candidate = self._normalize_scalar(candidate)
            if isinstance(normalized_candidate, str) and isinstance(target, str):
                if normalized_candidate.lower() == target.lower():
                    return result
            elif normalized_candidate == target:
                return result
        return None

    def _fn_index(self, array: Any, row_num: Any = 1, col_num: Any = 1) -> Any:
        table = self._to_table(array)
        if row_num is None and col_num is None:
            return table
        if row_num is None:
            col_index = int(self._normalize_scalar(col_num)) - 1
            return [row[col_index] for row in table]
        if col_num is None:
            row_index = int(self._normalize_scalar(row_num)) - 1
            return table[row_index]

        row_index = int(self._normalize_scalar(row_num)) - 1
        col_index = int(self._normalize_scalar(col_num)) - 1
        return table[row_index][col_index]

    def _fn_isna(self, value: Any) -> Any:
        if self._is_vector(value):
            return [item is None for item in value]
        return value is None

    def _fn_isblank(self, value: Any) -> Any:
        if self._is_vector(value):
            return [self._normalize_scalar(item) in (None, "") for item in value]
        return self._normalize_scalar(value) in (None, "")

    def _fn_not(self, value: Any) -> Any:
        if self._is_vector(value):
            return [not bool(item) for item in value]
        return not bool(value)

    def _fn_choose(self, selector: Any, *values: Any) -> Any:
        if self._is_vector(selector):
            indexes = [int(self._normalize_scalar(item)) for item in selector]
            columns = [self._ensure_vector(value) for value in values]
            return [[columns[index - 1][row_idx] for index in indexes] for row_idx in range(len(columns[0]))]
        index = int(self._normalize_scalar(selector))
        return values[index - 1]

    def _fn_lower(self, value: Any) -> Any:
        if self._is_vector(value):
            return [self._fn_lower(item) for item in value]
        if self._is_table(value):
            return [[self._fn_lower(item) for item in row] for row in value]
        return value.lower() if isinstance(value, str) else value

    def _fn_right(self, value: Any, count: Any = 1) -> Any:
        char_count = int(self._normalize_scalar(count))
        if self._is_vector(value):
            return [self._fn_right(item, char_count) for item in value]
        if isinstance(value, str):
            return value[-char_count:] if char_count > 0 else ""
        return value

    def _fn_search(self, needle: Any, haystack: Any) -> Any:
        if self._is_vector(haystack):
            return [self._fn_search(needle, item) for item in haystack]
        if haystack is None:
            return None
        needle_text = str(needle).lower()
        haystack_text = str(haystack).lower()
        index = haystack_text.find(needle_text)
        return None if index < 0 else index + 1

    def _fn_iferror(self, value: Any, fallback: Any) -> Any:
        if self._is_vector(value):
            return [fallback if item is None else item for item in value]
        return fallback if value is None else value

    def _fn_year(self, value: Any) -> Any:
        return self._date_part(value, "year")

    def _fn_month(self, value: Any) -> Any:
        return self._date_part(value, "month")

    def _fn_day(self, value: Any) -> Any:
        return self._date_part(value, "day")

    def _fn_sequence(self, count: Any) -> List[int]:
        total = int(self._normalize_scalar(count))
        return list(range(1, total + 1))

    def _apply_criteria(self, value_range: List[Any], criteria_args: Sequence[Any]) -> List[Any]:
        if len(criteria_args) % 2 != 0:
            raise FormulaSimulationError("IFS functions expect range/value pairs")

        criteria_ranges = []
        criteria_values = []
        for index in range(0, len(criteria_args), 2):
            criteria_ranges.append(self._ensure_vector(criteria_args[index]))
            criteria_values.append(criteria_args[index + 1])

        matched = []
        for row_index, current_value in enumerate(value_range):
            keep = True
            for criteria_range, expected in zip(criteria_ranges, criteria_values):
                comparator_result = self._compare(criteria_range[row_index], expected, "=")
                keep = keep and bool(comparator_result)
            if keep:
                matched.append(current_value)
        return matched

    def _numeric_values(self, value: Any) -> List[float]:
        flattened = self._flatten(value)
        numbers = []
        for item in flattened:
            numeric_value = self._coerce_number(item)
            if numeric_value is not None:
                numbers.append(numeric_value)
        return numbers

    def _flatten(self, value: Any) -> List[Any]:
        if self._is_table(value):
            return [item for row in value for item in row]
        if self._is_vector(value):
            return list(value)
        return [value]

    def _ensure_vector(self, value: Any) -> List[Any]:
        if self._is_table(value):
            if value and len(value[0]) == 1:
                return [row[0] for row in value]
            raise FormulaSimulationError("Expected a vector but received a table")
        if self._is_vector(value):
            return list(value)
        return [value]

    def _to_table(self, value: Any) -> List[List[Any]]:
        if self._is_table(value):
            return value
        if self._is_vector(value):
            return [[item] for item in value]
        return [[value]]

    def _row_count(self, value: Any) -> int:
        if self._is_table(value) or self._is_vector(value):
            return len(value)
        return 1

    def _normalize_scalar(self, value: Any) -> Any:
        number = self._coerce_number(value)
        if number is not None:
            return number
        if isinstance(value, str):
            return value.strip()
        return value

    def _date_part(self, value: Any, part: str) -> Any:
        if self._is_vector(value):
            return [self._date_part(item, part) for item in value]
        if self._is_table(value):
            result = []
            for row in value:
                parsed = None
                for item in row:
                    parsed = self._parse_date(item)
                    if parsed is not None:
                        break
                if parsed is None:
                    result.append(None)
                elif part == "year":
                    result.append(parsed.year)
                elif part == "month":
                    result.append(parsed.month)
                elif part == "day":
                    result.append(parsed.day)
            return result

        parsed = self._parse_date(value)
        if parsed is None:
            return None
        if part == "year":
            return parsed.year
        if part == "month":
            return parsed.month
        if part == "day":
            return parsed.day
        raise FormulaSimulationError(f"Unsupported date part: {part}")

    def _parse_date(self, value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str):
            return None

        text = value.strip()
        if not text:
            return None

        if date_parser is not None:
            try:
                return date_parser.parse(text, fuzzy=True)
            except (ValueError, OverflowError):
                pass

        for fmt in (
            "%B %d, %Y",
            "%b %d, %Y",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y",
        ):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    def _coerce_mask(self, value: Any, length: int) -> Optional[List[bool]]:
        if self._is_table(value):
            flattened = [bool(row[0]) if row else False for row in value]
            return flattened[:length]
        if self._is_vector(value):
            return [bool(item) for item in value[:length]]
        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return [bool(value)] * length
        return None

    def _coerce_number(self, value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None

            negative = False
            if stripped.startswith("(") and stripped.endswith(")"):
                negative = True
                stripped = stripped[1:-1].strip()

            percent = stripped.endswith("%")
            if percent:
                stripped = stripped[:-1].strip()

            stripped = stripped.replace(",", "")
            stripped = stripped.replace("$", "")

            if re.fullmatch(r'-?\d+(\.\d+)?', stripped):
                number = float(stripped)
                if negative:
                    number = -number
                if percent:
                    number /= 100.0
                return number
        return None

    def _is_cell_ref(self, token: str) -> bool:
        return re.fullmatch(r'[A-Za-z]+\d+', token) is not None

    def _split_ref(self, ref: str) -> Tuple[str, int]:
        match = re.fullmatch(r'([A-Za-z]+)(\d+)', ref)
        if not match:
            raise FormulaSimulationError(f"Invalid cell reference: {ref}")
        return match.group(1).upper(), int(match.group(2))

    def _col_to_index(self, col_name: str) -> int:
        result = 0
        for char in col_name.upper():
            result = result * 26 + (ord(char) - ord('A') + 1)
        index = result - 1
        if index < 0 or index >= self.num_cols:
            raise FormulaSimulationError(f"Column out of bounds: {col_name}")
        return index

    def _is_vector(self, value: Any) -> bool:
        return isinstance(value, list) and (not value or not isinstance(value[0], list))

    def _is_table(self, value: Any) -> bool:
        return isinstance(value, list) and bool(value) and isinstance(value[0], list)