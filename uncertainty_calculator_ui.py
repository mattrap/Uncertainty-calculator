#!/usr/bin/env python3
"""Minimal Tkinter UI for measurement and propagation uncertainty formulas."""
from __future__ import annotations

import json
import math
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError as exc:  # pragma: no cover - Tkinter should be available on stdlib builds
    raise SystemExit("Tkinter is required to run this interface") from exc


MEASUREMENT_FILE = Path(__file__).resolve().with_name("fluke_115_specs.json")
PROPAGATION_FILE = Path(__file__).resolve().with_name("uncertainty_propagation_rules.json")
PREFERENCES_FILE = Path(__file__).resolve().with_name("user_preferences.json")


@dataclass
class UserPreferences:
    significant_digits: int = 1
    decimal_delimiter: str = '.'

    def ensure_valid(self) -> None:
        if self.significant_digits <= 0:
            self.significant_digits = 1
        if self.decimal_delimiter not in {'.', ','}:
            self.decimal_delimiter = '.'

    def as_dict(self) -> Dict[str, object]:
        return {
            "significant_digits": int(self.significant_digits),
            "decimal_delimiter": self.decimal_delimiter,
        }


def load_user_preferences() -> UserPreferences:
    try:
        data = json.loads(PREFERENCES_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        prefs = UserPreferences()
        prefs.ensure_valid()
        return prefs
    except json.JSONDecodeError:
        prefs = UserPreferences()
        prefs.ensure_valid()
        return prefs
    significant = int(data.get("significant_digits", 1))
    delimiter = str(data.get("decimal_delimiter", '.'))
    prefs = UserPreferences(significant_digits=significant, decimal_delimiter=delimiter)
    prefs.ensure_valid()
    return prefs


def save_user_preferences(prefs: UserPreferences) -> None:
    prefs.ensure_valid()
    PREFERENCES_FILE.write_text(json.dumps(prefs.as_dict(), indent=2), encoding="utf-8")


class RuleLoadError(RuntimeError):
    """Raised when the JSON rule bundle cannot be loaded."""


class FormulaParseError(ValueError):
    """Raised when the propagation formula contains unsupported constructs."""


@dataclass
class MeasurementRange:
    range_max: float
    resolution: float
    p_reading: float
    counts: float
    unit: str
    display_range: str
    display_resolution: str
    accuracy_text: str

    @property
    def counts_term(self) -> float:
        return self.counts * self.resolution


@dataclass
class MeasurementResult:
    formula: str
    selected_range: MeasurementRange
    input_unit: str


@dataclass
class ExprInfo:
    value_expr: str
    abs_unc_expr: str
    rel_unc_expr: str


@dataclass
class OperatorRule:
    domain: str
    combine: str


@dataclass
class PropagationRuleSet:
    name: str
    operators: Dict[str, OperatorRule]


class MeasurementCalculator:
    """Handles measurement uncertainty formulas based on range data."""

    def __init__(
        self,
        ranges_by_mode: Dict[str, List[MeasurementRange]],
        unit_factors: Dict[str, OrderedDict[str, float]],
    ):
        self._ranges_by_mode = {
            mode: sorted(ranges, key=lambda r: r.range_max)
            for mode, ranges in ranges_by_mode.items()
        }
        self._unit_factors = unit_factors

    @property
    def modes(self) -> List[str]:
        return sorted(self._ranges_by_mode)

    def units(self, mode: str) -> List[str]:
        try:
            factors = self._unit_factors[mode]
        except KeyError as exc:
            raise ValueError(f"Unknown measurement mode '{mode}'") from exc
        return list(factors.keys())

    def default_unit(self, mode: str) -> str:
        factors = self._unit_factors.get(mode)
        if not factors:
            raise ValueError(f"Unknown measurement mode '{mode}'")
        return next(iter(factors))

    def unit_factor(self, mode: str, unit_label: str) -> float:
        try:
            return self._unit_factors[mode][unit_label]
        except KeyError as exc:
            raise ValueError(f"Unsupported unit '{unit_label}' for mode '{mode}'") from exc

    def _find_range(self, mode: str, magnitude_si: float) -> MeasurementRange:
        ranges = self._ranges_by_mode.get(mode)
        if not ranges:
            raise ValueError(f"Unknown measurement mode '{mode}'")
        for entry in ranges:
            if magnitude_si <= entry.range_max:
                return entry
        return ranges[-1]

    def formula_for(self, mode: str, cell_ref: str, measured_value: float, unit_label: str) -> MeasurementResult:
        try:
            unit_factor = self._unit_factors[mode][unit_label]
        except KeyError as exc:
            raise ValueError(f"Unsupported unit '{unit_label}' for mode '{mode}'") from exc
        magnitude_si = abs(measured_value * unit_factor)
        chosen = self._find_range(mode, magnitude_si)
        coeff = format_number(chosen.p_reading)
        counts_term_value = chosen.counts_term / unit_factor
        if math.isclose(counts_term_value, 0.0, abs_tol=1e-15):
            counts_term = '0'
        else:
            resolution_term = format_number(chosen.resolution / unit_factor)
            counts_literal = format_number(chosen.counts)
            counts_term = f"({resolution_term}*{counts_literal})"
        cell_expr = cell_ref.upper()
        formula = f"={coeff}*ABS({cell_expr})+{counts_term}"
        return MeasurementResult(formula=formula, selected_range=chosen, input_unit=unit_label)


ENGINEERING_PREFIXES: Dict[str, float] = {
    'n': 1e-9,
    'µ': 1e-6,
    'u': 1e-6,
    'm': 1e-3,
    'k': 1e3,
    'M': 1e6,
}


QUANTITY_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-zµΩ]+)\s*$")
ACCURACY_RE = re.compile(r"\(\s*(?P<percent>[0-9]+(?:\.[0-9]+)?)\s*%\s*\+\s*(?P<counts>[0-9]+(?:\.[0-9]+)?)\s*(?:counts|digits)?\s*\)")


def _split_unit_token(token: str) -> Tuple[float, str]:
    for prefix, factor in ENGINEERING_PREFIXES.items():
        if token.startswith(prefix) and len(token) > len(prefix):
            return factor, token[len(prefix):]
    return 1.0, token


def parse_quantity(text: str, context: str) -> Tuple[float, str, str, float]:
    match = QUANTITY_RE.match(text.strip())
    if not match:
        raise RuleLoadError(f"Cannot parse quantity '{text}' in {context}")
    magnitude = float(match.group(1))
    token = match.group(2)
    factor, base_unit = _split_unit_token(token)
    if not base_unit:
        raise RuleLoadError(f"Missing base unit in '{text}' ({context})")
    return magnitude * factor, base_unit, token, factor


def parse_accuracy_formula(spec_text: str, context: str) -> Tuple[float, float]:
    cleaned = spec_text.strip()
    if cleaned.startswith('±') or cleaned.startswith('\u00b1'):
        cleaned = cleaned[1:].lstrip()
    match = ACCURACY_RE.search(cleaned)
    if not match:
        raise RuleLoadError(f"Cannot parse accuracy expression '{spec_text}' in {context}")
    percent = float(match.group('percent')) / 100.0
    counts = float(match.group('counts'))
    return percent, counts


def expand_accuracy_options(accuracy_field: object, context: str) -> List[Tuple[Optional[str], float, float, str]]:
    if isinstance(accuracy_field, dict):
        options: List[Tuple[Optional[str], float, float, str]] = []
        for label, spec_text in accuracy_field.items():
            p_reading, counts = parse_accuracy_formula(str(spec_text), context)
            display_text = f"{label}: {str(spec_text).strip()}"
            options.append((label, p_reading, counts, display_text))
        return options
    if isinstance(accuracy_field, str):
        segments = [seg.strip() for seg in accuracy_field.split(';') if seg.strip()]
        if not segments:
            segments = [accuracy_field.strip()]
        options = []
        for segment in segments:
            label: Optional[str] = None
            spec_text = segment
            if ':' in segment:
                label_part, remainder = segment.split(':', 1)
                label = label_part.strip() or None
                spec_text = remainder.strip()
            p_reading, counts = parse_accuracy_formula(spec_text, context)
            display_text = segment.strip()
            options.append((label, p_reading, counts, display_text))
        return options
    raise RuleLoadError(f"Unsupported accuracy structure '{accuracy_field}' in {context}")


class PropagationCalculator:
    """Builds uncertainty formulas for calculated values."""

    CELL_REF_RE = re.compile(r"^([A-Za-z]+)([1-9][0-9]*)$")

    def __init__(self, rule_sets: Dict[str, PropagationRuleSet]):
        self._rule_sets = rule_sets

    @property
    def rules(self) -> List[str]:
        return sorted(self._rule_sets)

    def generate_formula(self, expression: str, rule_name: str) -> str:
        raw = expression.strip()
        if not raw:
            raise FormulaParseError("Enter an Excel formula (e.g., =C10*C11)")
        if raw.startswith("="):
            raw = raw[1:]
        rule_set = self._rule_sets.get(rule_name)
        if not rule_set:
            raise FormulaParseError(f"Unknown propagation rule set '{rule_name}'")
        tokens = tokenize(raw)
        parser = Parser(tokens)
        ast = parser.parse_expression()
        parser.ensure_consumed()
        context = PropagationContext(rule_set)
        info = ast.evaluate(context)
        return f"={info.abs_unc_expr}"

    @classmethod
    def map_value_cell_to_uncertainty(cls, value_cell: str) -> str:
        match = cls.CELL_REF_RE.match(value_cell)
        if not match:
            raise FormulaParseError(f"Unsupported cell reference '{value_cell}'")
        column_text, row_text = match.groups()
        column_index = column_to_index(column_text.upper())
        next_column = index_to_column(column_index + 1)
        return f"{next_column}{row_text}"


class PropagationContext:
    """Holds rule-set specific helpers during expression evaluation."""

    def __init__(self, rule_set: PropagationRuleSet):
        self.rule_set = rule_set

    def operator(self, symbol: str) -> OperatorRule:
        try:
            return self.rule_set.operators[symbol]
        except KeyError as exc:
            raise FormulaParseError(f"Operator '{symbol}' not supported in rule set '{self.rule_set.name}'") from exc

    def map_cell(self, cell_ref: str) -> Tuple[str, str]:
        cell = cell_ref.upper()
        uncert = PropagationCalculator.map_value_cell_to_uncertainty(cell)
        return cell, uncert


class ExprNode:
    def evaluate(self, context: PropagationContext) -> ExprInfo:
        raise NotImplementedError


class NumberNode(ExprNode):
    def __init__(self, literal: str):
        self.literal = literal

    def evaluate(self, context: PropagationContext) -> ExprInfo:  # noqa: D401 - short and clear
        return ExprInfo(value_expr=self.literal, abs_unc_expr="0", rel_unc_expr="0")


class CellNode(ExprNode):
    def __init__(self, identifier: str):
        self.identifier = identifier.upper()

    def evaluate(self, context: PropagationContext) -> ExprInfo:
        value_cell, uncert_cell = context.map_cell(self.identifier)
        abs_unc = uncert_cell
        rel_unc = safe_relative(abs_unc, value_cell)
        return ExprInfo(value_expr=value_cell, abs_unc_expr=abs_unc, rel_unc_expr=rel_unc)


class UnaryNode(ExprNode):
    def __init__(self, operator: str, operand: ExprNode):
        self.operator = operator
        self.operand = operand

    def evaluate(self, context: PropagationContext) -> ExprInfo:
        inner = self.operand.evaluate(context)
        if self.operator == '+':
            return inner
        if self.operator == '-':
            value = f"-({inner.value_expr})"
            return ExprInfo(value_expr=value, abs_unc_expr=inner.abs_unc_expr, rel_unc_expr=inner.rel_unc_expr)
        raise FormulaParseError(f"Unsupported unary operator '{self.operator}'")


class BinaryNode(ExprNode):
    def __init__(self, operator: str, left: ExprNode, right: ExprNode):
        self.operator = operator
        self.left = left
        self.right = right

    def evaluate(self, context: PropagationContext) -> ExprInfo:
        op = context.operator(self.operator)
        left_info = self.left.evaluate(context)
        right_info = self.right.evaluate(context)
        value_expr = self._build_value_expr(left_info.value_expr, right_info.value_expr)
        if op.domain == 'absolute':
            abs_unc = combine_absolute(left_info.abs_unc_expr, right_info.abs_unc_expr, op.combine)
            rel_unc = safe_relative(abs_unc, value_expr)
            return ExprInfo(value_expr=value_expr, abs_unc_expr=abs_unc, rel_unc_expr=rel_unc)
        if op.domain == 'relative':
            rel_unc = combine_relative(left_info.rel_unc_expr, right_info.rel_unc_expr, op.combine)
            abs_unc = f"ABS({value_expr})*({rel_unc})"
            return ExprInfo(value_expr=value_expr, abs_unc_expr=abs_unc, rel_unc_expr=rel_unc)
        raise FormulaParseError(f"Unsupported domain '{op.domain}' for operator '{self.operator}'")

    def _build_value_expr(self, left: str, right: str) -> str:
        if self.operator == '+':
            return f"({left}+{right})"
        if self.operator == '-':
            return f"({left}-{right})"
        if self.operator == '*':
            return f"({left}*{right})"
        if self.operator == '/':
            return f"({left}/{right})"
        if self.operator == '^':
            return f"({left}^{right})"
        raise FormulaParseError(f"Unsupported operator '{self.operator}'")


class PowerNode(ExprNode):
    def __init__(self, base: ExprNode, exponent_literal: str):
        self.base = base
        self.exponent_literal = exponent_literal

    def evaluate(self, context: PropagationContext) -> ExprInfo:
        base_info = self.base.evaluate(context)
        try:
            exponent_value = float(self.exponent_literal)
        except ValueError as exc:
            raise FormulaParseError("Exponent must be numeric for uncertainty propagation") from exc
        abs_exponent = format_number(abs(exponent_value))
        value_expr = f"({base_info.value_expr}^{self.exponent_literal})"
        rel_unc = f"{abs_exponent}*({base_info.rel_unc_expr})"
        abs_unc = f"ABS({value_expr})*({rel_unc})"
        return ExprInfo(value_expr=value_expr, abs_unc_expr=abs_unc, rel_unc_expr=rel_unc)


class Parser:
    def __init__(self, tokens: Sequence['Token']):
        self.tokens = list(tokens)
        self.position = 0

    def parse_expression(self) -> ExprNode:
        node = self._parse_term()
        while self._match('+', '-'):
            operator = self._previous().value
            right = self._parse_term()
            node = BinaryNode(operator, node, right)
        return node

    def ensure_consumed(self) -> None:
        if not self._is_at_end():
            token = self._peek()
            raise FormulaParseError(f"Unexpected token '{token.value}' in formula")

    def _parse_term(self) -> ExprNode:
        node = self._parse_power()
        while self._match('*', '/'):
            operator = self._previous().value
            right = self._parse_power()
            node = BinaryNode(operator, node, right)
        return node

    def _parse_power(self) -> ExprNode:
        node = self._parse_unary()
        if self._match('^'):
            sign = ''
            if self._match('+', '-'):
                sign = self._previous().value
            if not self._match_type('NUMBER'):
                token = self._peek() if not self._is_at_end() else None
                got = token.value if token else 'end of input'
                raise FormulaParseError(f"Exponent must be numeric, got '{got}'")
            literal = self._previous().value
            node = PowerNode(node, sign + literal)
        return node

    def _parse_unary(self) -> ExprNode:
        if self._match('+', '-'):
            operator = self._previous().value
            operand = self._parse_unary()
            return UnaryNode(operator, operand)
        return self._parse_primary()

    def _parse_primary(self) -> ExprNode:
        if self._match_type('NUMBER'):
            return NumberNode(self._previous().value)
        if self._match_type('CELL'):
            return CellNode(self._previous().value)
        if self._match('('):
            expr = self.parse_expression()
            self._consume(')')
            return expr
        token = self._peek()
        raise FormulaParseError(f"Unexpected token '{token.value}' in expression")

    def _match(self, *operators: str) -> bool:
        if self._is_at_end():
            return False
        if self._peek().type != 'OP':
            return False
        if self._peek().value not in operators:
            return False
        self.position += 1
        return True

    def _match_type(self, token_type: str) -> bool:
        if self._is_at_end():
            return False
        if self._peek().type != token_type:
            return False
        self.position += 1
        return True

    def _consume(self, expected: str) -> None:
        if self._match(expected):
            return
        token = self._peek() if not self._is_at_end() else None
        raise FormulaParseError(f"Expected '{expected}', got '{token.value if token else 'end of input'}'")

    def _previous(self) -> 'Token':
        return self.tokens[self.position - 1]

    def _peek(self) -> 'Token':
        return self.tokens[self.position]

    def _is_at_end(self) -> bool:
        return self.position >= len(self.tokens)


@dataclass
class Token:
    type: str
    value: str


def tokenize(expression: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    length = len(expression)
    while i < length:
        char = expression[i]
        if char.isspace():
            i += 1
            continue
        if char in '+-*/^()':
            tokens.append(Token('OP', char))
            i += 1
            continue
        if char.isdigit() or (char == '.' and i + 1 < length and expression[i + 1].isdigit()):
            match = NUMBER_RE.match(expression, i)
            if not match:
                raise FormulaParseError(f"Invalid number starting at '{expression[i:]} '")
            literal = match.group(0)
            tokens.append(Token('NUMBER', literal))
            i = match.end()
            continue
        if char.isalpha():
            match = IDENTIFIER_RE.match(expression, i)
            if not match:
                raise FormulaParseError(f"Invalid identifier starting at '{expression[i:]}'")
            identifier = match.group(0)
            j = match.end()
            digit_match = CELL_TAIL_RE.match(expression, j)
            if digit_match:
                tokens.append(Token('CELL', (identifier + digit_match.group(0))))
                i = digit_match.end()
                continue
            raise FormulaParseError("Excel functions are not supported in propagation formulas yet")
        raise FormulaParseError(f"Unexpected character '{char}' in expression")
    return tokens


NUMBER_RE = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
IDENTIFIER_RE = re.compile(r"[A-Za-z]+")
CELL_TAIL_RE = re.compile(r"[1-9][0-9]*")


def combine_absolute(left: str, right: str, style: str) -> str:
    if style == 'linear':
        return sum_terms([left, right])
    if style == 'rss':
        return rss_terms([left, right])
    raise FormulaParseError(f"Unsupported absolute combination style '{style}'")


def combine_relative(left: str, right: str, style: str) -> str:
    if style == 'linear':
        return sum_terms([left, right])
    if style == 'rss':
        return rss_terms([left, right])
    raise FormulaParseError(f"Unsupported relative combination style '{style}'")


def sum_terms(terms: Sequence[str]) -> str:
    parts = [term for term in terms if not is_zero(term)]
    if not parts:
        return '0'
    if len(parts) == 1:
        return parts[0]
    return '(' + '+'.join(parts) + ')'


def rss_terms(terms: Sequence[str]) -> str:
    parts = [term for term in terms if not is_zero(term)]
    if not parts:
        return '0'
    if len(parts) == 1:
        return f"ABS({parts[0]})"
    squares = '+'.join(f"({term})^2" for term in parts)
    return f"SQRT({squares})"


def safe_relative(absolute_expr: str, value_expr: str) -> str:
    if is_zero(absolute_expr):
        return '0'
    value_ref = f"({value_expr})"
    numerator = f"({absolute_expr})"
    return f"IF({value_ref}=0,0,{numerator}/ABS({value_expr}))"


def is_zero(expr: str) -> bool:
    stripped = expr.replace(' ', '')
    return stripped in {'0', '0.0', '0.00', '0E+0', 'ABS(0)', 'IF(1=1,0,0)'}


def format_number(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-15):
        return '0'
    return f"{value:.12g}"


def round_to_significant(value: float, figures: int = 1) -> float:
    if figures <= 0:
        raise ValueError("Significant figures must be positive")
    if math.isclose(value, 0.0, abs_tol=1e-30):
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    factor = 10 ** (exponent - figures + 1)
    return round(value / factor) * factor


def column_to_index(column: str) -> int:
    result = 0
    for char in column:
        if not char.isalpha() or not char.isupper():
            raise FormulaParseError(f"Unexpected column text '{column}'")
        result = result * 26 + (ord(char) - ord('A') + 1)
    return result


def index_to_column(index: int) -> str:
    if index <= 0:
        raise FormulaParseError("Column index must be positive")
    letters = []
    while index:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(ord('A') + remainder))
    return ''.join(reversed(letters))


class PlaceholderEntry(tk.Entry):
    """Entry widget that shows placeholder text when empty."""

    def __init__(self, master: tk.Widget, placeholder: str, *, placeholder_color: str = 'grey50', **kwargs):
        super().__init__(master, **kwargs)
        self._placeholder = placeholder
        self._placeholder_color = placeholder_color
        self._default_fg = self.cget('fg')
        self._has_placeholder = False
        self._set_placeholder()
        self.bind('<FocusIn>', self._on_focus_in, add=True)
        self.bind('<FocusOut>', self._on_focus_out, add=True)

    def _set_placeholder(self) -> None:
        if self._has_placeholder or self.get():
            return
        self._has_placeholder = True
        self.configure(fg=self._placeholder_color)
        self.insert(0, self._placeholder)

    def _clear_placeholder(self) -> None:
        if not self._has_placeholder:
            return
        self.delete(0, tk.END)
        self.configure(fg=self._default_fg)
        self._has_placeholder = False

    def _on_focus_in(self, _event: tk.Event) -> None:  # type: ignore[override]
        self._clear_placeholder()

    def _on_focus_out(self, _event: tk.Event) -> None:  # type: ignore[override]
        if not self.get():
            self._set_placeholder()

    def get_value(self) -> str:
        return '' if self._has_placeholder else self.get()

    def set_value(self, value: str) -> None:
        self._clear_placeholder()
        self.delete(0, tk.END)
        if value:
            self.insert(0, value)
        else:
            self._set_placeholder()

    def update_placeholder(self, placeholder: str) -> None:
        self._placeholder = placeholder
        if self._has_placeholder:
            self.delete(0, tk.END)
            self.insert(0, placeholder)
            self.configure(fg=self._placeholder_color)


class App(tk.Tk):
    def __init__(self, measurement: MeasurementCalculator, propagation: PropagationCalculator):
        super().__init__()
        self.measurement = measurement
        self.propagation = propagation
        self.preferences = load_user_preferences()
        self.sig_figs_var = tk.IntVar(value=self.preferences.significant_digits)
        self.decimal_delimiter_var = tk.StringVar(value=self.preferences.decimal_delimiter)
        self._delimiter_button: Optional[ttk.Button] = None
        self._settings_window: Optional[tk.Toplevel] = None
        self.title("Uncertainty Helper")
        self.geometry("540x420")
        self.resizable(False, False)
        self._setup_ui()

    def _setup_ui(self) -> None:
        top_frame = ttk.Frame(self)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        top_frame.columnconfigure(0, weight=1)
        top_frame.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(top_frame)
        toolbar.grid(row=0, column=0, sticky=tk.EW, pady=(0, 8))
        toolbar.columnconfigure(0, weight=1)

        title_label = ttk.Label(toolbar, text="Uncertainty Helper")
        title_label.grid(row=0, column=0, sticky=tk.W)
        settings_label = ttk.Label(toolbar, text="⚙ Settings", cursor="hand2")
        settings_label.grid(row=0, column=1, sticky=tk.E)
        settings_label.bind("<Button-1>", lambda _event: self._open_settings())

        notebook = ttk.Notebook(top_frame)
        notebook.grid(row=1, column=0, sticky=tk.NSEW)

        meas_frame = ttk.Frame(notebook)
        calc_frame = ttk.Frame(notebook)
        notebook.add(meas_frame, text="Measured value")
        notebook.add(calc_frame, text="Calculated value")

        self._build_measured_tab(meas_frame)
        self._build_calculated_tab(calc_frame)

    def _build_measured_tab(self, container: ttk.Frame) -> None:
        container.columnconfigure(1, weight=1)
        container.columnconfigure(2, weight=0)

        ttk.Label(container, text="Measurement type:").grid(row=0, column=0, sticky=tk.W, pady=(0, 6))
        first_mode = self.measurement.modes[0]
        self.measure_mode = tk.StringVar(value=first_mode)
        mode_menu = ttk.OptionMenu(
            container,
            self.measure_mode,
            first_mode,
            *self.measurement.modes,
            command=self._on_measure_mode_change,
        )
        mode_menu.grid(row=0, column=1, sticky=tk.EW, pady=(0, 6))

        self.measure_unit_var = tk.StringVar()
        self.measure_unit_menu = ttk.OptionMenu(container, self.measure_unit_var, "")
        self.measure_unit_menu.grid(row=0, column=2, sticky=tk.EW, padx=(6, 0), pady=(0, 6))

        ttk.Label(container, text="Excel cell (value):").grid(row=1, column=0, sticky=tk.W, pady=6)
        self.measure_cell = PlaceholderEntry(container, placeholder="A37")
        self.measure_cell.grid(row=1, column=1, sticky=tk.EW, pady=6)

        ttk.Label(container, text="Measured value:").grid(row=2, column=0, sticky=tk.W, pady=6)
        self.measure_value = PlaceholderEntry(container, placeholder=self._default_numeric_placeholder())
        self.measure_value.grid(row=2, column=1, sticky=tk.EW, pady=6)
        self._configure_numeric_entry(self.measure_value)

        ttk.Label(container, text="Uncertainty:").grid(row=3, column=0, sticky=tk.W, pady=6)
        self.measure_uncertainty_var = tk.StringVar()
        self.measure_uncertainty_entry = ttk.Entry(
            container,
            textvariable=self.measure_uncertainty_var,
            state='readonly',
        )
        self.measure_uncertainty_entry.grid(row=3, column=1, sticky=tk.EW, pady=6)

        ttk.Label(container, text="Formula:").grid(row=4, column=0, sticky=tk.W, pady=6)
        self.measure_formula_var = tk.StringVar()
        self.measure_formula_entry = ttk.Entry(container, textvariable=self.measure_formula_var, state='readonly')
        self.measure_formula_entry.grid(row=4, column=1, sticky=tk.EW, pady=6)
        ttk.Button(container, text="Copy", command=self._copy_measure_formula).grid(row=4, column=2, padx=(6, 0), pady=6)

        compute_btn = ttk.Button(container, text="Build formula", command=self._handle_measurement)
        compute_btn.grid(row=5, column=0, columnspan=3, pady=(12, 8))

        self.measure_output = tk.Text(container, height=6, width=40)
        self.measure_output.grid(row=6, column=0, columnspan=3, sticky=tk.EW, pady=(6, 0))
        self.measure_output.configure(state=tk.DISABLED)

        # populate unit selection based on initial mode
        self._on_measure_mode_change(first_mode)

    def _build_calculated_tab(self, container: ttk.Frame) -> None:
        container.columnconfigure(1, weight=1)
        container.columnconfigure(2, weight=0)

        ttk.Label(container, text="Formula for value:").grid(row=0, column=0, sticky=tk.W, pady=(0, 6))
        self.calc_formula_entry = tk.Entry(container)
        self.calc_formula_entry.insert(0, "=C10*C11")
        self.calc_formula_entry.grid(row=0, column=1, sticky=tk.EW, pady=(0, 6))

        ttk.Label(container, text="Propagation rule set:").grid(row=1, column=0, sticky=tk.W, pady=6)
        default_rule = self.propagation.rules[0]
        self.rules_var = tk.StringVar(value=default_rule)
        rules_menu = ttk.OptionMenu(container, self.rules_var, default_rule, *self.propagation.rules)
        rules_menu.grid(row=1, column=1, sticky=tk.EW, pady=6)

        ttk.Label(container, text="Uncertainty formula:").grid(row=2, column=0, sticky=tk.W, pady=6)
        self.calc_formula_var = tk.StringVar()
        self.calc_formula_display = ttk.Entry(container, textvariable=self.calc_formula_var, state='readonly')
        self.calc_formula_display.grid(row=2, column=1, sticky=tk.EW, pady=6)
        ttk.Button(container, text="Copy", command=self._copy_calc_formula).grid(row=2, column=2, padx=(6, 0), pady=6)

        compute_btn = ttk.Button(container, text="Build formula", command=self._handle_calculated)
        compute_btn.grid(row=3, column=0, columnspan=3, pady=(12, 8))

        self.calc_output = tk.Text(container, height=6, width=40)
        self.calc_output.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=(6, 0))
        self.calc_output.configure(state=tk.DISABLED)

    def _current_decimal_delimiter(self) -> str:
        value = self.decimal_delimiter_var.get()
        return value if value in {'.', ','} else '.'

    def _default_numeric_placeholder(self) -> str:
        return f"1{self._current_decimal_delimiter()}0"

    def _configure_numeric_entry(self, entry: PlaceholderEntry) -> None:
        command = (self.register(self._validate_numeric_input), "%P")
        entry.configure(validate='key', validatecommand=command)
        setattr(entry, "_validatecommand", command)

    def _validate_numeric_input(self, proposed: str) -> bool:
        if proposed == '':
            return True
        delimiter = self._current_decimal_delimiter()
        alternate = ',' if delimiter == '.' else '.'
        if alternate in proposed:
            return False
        allowed = set('0123456789+-')
        allowed.add(delimiter)
        if not all(char in allowed for char in proposed):
            return False
        if proposed.count(delimiter) > 1:
            return False
        if proposed in {delimiter, '+', '-'}:
            return True
        if proposed.startswith(('-', '+')):
            rest = proposed[1:]
            if rest.startswith(('-', '+')):
                return False
        if '+' in proposed[1:] or '-' in proposed[1:]:
            return False
        return True

    def _parse_user_float(self, text: str) -> float:
        delimiter = self._current_decimal_delimiter()
        alternate = ',' if delimiter == '.' else '.'
        if alternate in text:
            raise ValueError(f"Use '{delimiter}' as the decimal delimiter")
        if text.count(delimiter) > 1:
            raise ValueError("Enter a valid numeric value")
        normalized = text.replace(delimiter, '.') if delimiter != '.' else text
        try:
            return float(normalized)
        except ValueError as exc:
            raise ValueError(f"Enter a numeric value using '{delimiter}' as the decimal delimiter") from exc

    def _format_user_number(self, value: float) -> str:
        text = format_number(value)
        delimiter = self._current_decimal_delimiter()
        if delimiter == ',' and '.' in text:
            whole, _, fraction = text.partition('.')
            return f"{whole},{fraction}"
        return text

    def _apply_delimiter_to_entry(self, entry: PlaceholderEntry) -> None:
        entry.update_placeholder(self._default_numeric_placeholder())
        value = entry.get_value()
        if not value:
            entry.set_value('')
            return
        normalized = value.replace(',', '.')
        try:
            numeric = float(normalized)
        except ValueError:
            return
        entry.set_value(self._format_user_number(numeric))

    def _update_preferences_from_vars(self) -> None:
        try:
            figures = int(self.sig_figs_var.get())
        except Exception:
            figures = self.preferences.significant_digits
        if figures <= 0:
            figures = 1
        self.preferences.significant_digits = figures
        self.preferences.decimal_delimiter = self._current_decimal_delimiter()
        self.sig_figs_var.set(figures)
        self.decimal_delimiter_var.set(self.preferences.decimal_delimiter)

    def _save_preferences(self) -> None:
        save_user_preferences(self.preferences)

    def _delimiter_button_text(self) -> str:
        return f"Decimal delimiter: {self._current_decimal_delimiter()}"

    def _handle_delimiter_updated(self) -> None:
        if self._delimiter_button is not None and self._delimiter_button.winfo_exists():
            self._delimiter_button.configure(text=self._delimiter_button_text())
        self._apply_delimiter_to_entry(self.measure_value)
        self._update_preferences_from_vars()
        self._save_preferences()

    def _toggle_delimiter(self) -> None:
        current = self._current_decimal_delimiter()
        new_value = ',' if current == '.' else '.'
        self.decimal_delimiter_var.set(new_value)
        self._handle_delimiter_updated()

    def _on_measure_mode_change(self, *_: str) -> None:
        mode = self.measure_mode.get()
        try:
            units = self.measurement.units(mode)
            default_unit = self.measurement.default_unit(mode)
        except ValueError:
            units = []
            default_unit = ''
        menu = self.measure_unit_menu['menu']
        menu.delete(0, 'end')
        for unit in units:
            menu.add_command(label=unit, command=tk._setit(self.measure_unit_var, unit))
        if units:
            self.measure_unit_var.set(default_unit)
        else:
            self.measure_unit_var.set('')

    def _handle_measurement(self) -> None:
        self.measure_uncertainty_var.set('')
        try:
            mode = self.measure_mode.get()
            cell = self.measure_cell.get_value().strip().upper()
            if not cell:
                raise ValueError("Enter the Excel cell containing the measurement")
            unit_label = self.measure_unit_var.get().strip()
            if not unit_label:
                raise ValueError("Select the units for the measured value")
            value_text = self.measure_value.get_value().strip()
            if not value_text:
                raise ValueError("Enter the measured value")
            value = self._parse_user_float(value_text)
            target_cell = PropagationCalculator.map_value_cell_to_uncertainty(cell)
            result = self.measurement.formula_for(mode, cell, value, unit_label)
            unit_factor = self.measurement.unit_factor(mode, unit_label)
            figures = self.sig_figs_var.get()
            if figures <= 0:
                raise ValueError("Significant figures must be positive")
        except Exception as exc:  # noqa: BLE001 - show any validation issue to the user
            messagebox.showerror("Measurement error", str(exc))
            return

        self.measure_formula_var.set(result.formula)
        counts_term_value = result.selected_range.counts_term / unit_factor
        uncertainty_value = abs(value) * result.selected_range.p_reading + counts_term_value
        rounded_uncertainty = round_to_significant(uncertainty_value, figures)
        uncertainty_text = f"{self._format_user_number(rounded_uncertainty)} {result.input_unit}".strip()
        self.measure_uncertainty_var.set(uncertainty_text)

        counts_term = self._format_user_number(counts_term_value)
        text_lines = [
            f"Paste into: {target_cell}",
            "",
            "Details:",
            f"Measurement mode: {mode}",
            f"Input units: {result.input_unit}",
            f"Range max: {result.selected_range.display_range}",
            f"Resolution: {result.selected_range.display_resolution}",
            f"p(reading): {self._format_user_number(result.selected_range.p_reading)}",
            f"Counts: {self._format_user_number(result.selected_range.counts)}",
            f"Counts term ({result.input_unit}): {counts_term}",
            f"Uncertainty ({figures} sig fig): {uncertainty_text}",
            f"Accuracy: {result.selected_range.accuracy_text}",
        ]
        self._set_text(self.measure_output, '\n'.join(text_lines))

    def _handle_calculated(self) -> None:
        try:
            expression = self.calc_formula_entry.get()
            rule = self.rules_var.get()
            formula = self.propagation.generate_formula(expression, rule)
        except Exception as exc:  # noqa: BLE001 - show any validation issue to the user
            messagebox.showerror("Propagation error", str(exc))
            return
        self.calc_formula_var.set(formula)
        self._set_text(
            self.calc_output,
            "Input uncertainties are taken from the next column to each referenced value cell.",
        )

    def _set_text(self, widget: tk.Text, value: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete('1.0', tk.END)
        widget.insert(tk.END, value)
        widget.configure(state=tk.DISABLED)

    def _copy_measure_formula(self) -> None:
        self._copy_to_clipboard(self.measure_formula_var.get())

    def _copy_calc_formula(self) -> None:
        self._copy_to_clipboard(self.calc_formula_var.get())

    def _copy_to_clipboard(self, value: str) -> None:
        text = value.strip()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()

    def _open_settings(self) -> None:
        if self._settings_window and self._settings_window.winfo_exists():
            self._settings_window.lift()
            self._settings_window.focus_set()
            return

        window = tk.Toplevel(self)
        window.title("Settings")
        window.resizable(False, False)
        window.transient(self)
        window.grab_set()
        self._settings_window = window
        window.columnconfigure(0, weight=0)
        window.columnconfigure(1, weight=0)

        ttk.Label(window, text="Uncertainty significant figures:").grid(
            row=0, column=0, padx=12, pady=(12, 6), sticky=tk.W
        )
        sig_spin = ttk.Spinbox(window, from_=1, to=6, textvariable=self.sig_figs_var, width=5)
        sig_spin.grid(row=0, column=1, padx=12, pady=(12, 6), sticky=tk.W)
        sig_spin.focus_set()

        self._delimiter_button = ttk.Button(
            window,
            text=self._delimiter_button_text(),
            command=self._toggle_delimiter,
            width=18,
        )
        self._delimiter_button.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 6), sticky=tk.W)

        button_frame = ttk.Frame(window)
        button_frame.grid(row=2, column=0, columnspan=2, padx=12, pady=(6, 12), sticky=tk.E)

        def save() -> None:
            try:
                value = int(self.sig_figs_var.get())
            except Exception:  # noqa: BLE001 - minimal dialog validation
                messagebox.showerror("Settings", "Significant figures must be an integer")
                return
            if value <= 0:
                messagebox.showerror("Settings", "Significant figures must be positive")
                return
            self.sig_figs_var.set(value)
            self._update_preferences_from_vars()
            self._save_preferences()
            window.destroy()

        ttk.Button(button_frame, text="Save", command=save).grid(row=0, column=0, padx=(6, 0))

        def handle_close() -> None:
            self._settings_window = None
            self._delimiter_button = None
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", handle_close)

        def on_destroy(event: tk.Event) -> None:  # type: ignore[override]
            if event.widget is window:
                self._settings_window = None
                self._delimiter_button = None

        window.bind("<Destroy>", on_destroy, add=True)



def build_measurement_calculator_from_specs(payload: dict, source: str) -> MeasurementCalculator:
    specs = payload.get('specifications')
    if not isinstance(specs, dict):
        raise RuleLoadError(f"Missing 'specifications' section in {source}")

    ranges_by_mode: Dict[str, List[MeasurementRange]] = {}
    units_by_mode: Dict[str, OrderedDict[str, float]] = {}

    for base_mode, entries in specs.items():
        if not isinstance(entries, list):
            raise RuleLoadError(f"Ranges for mode '{base_mode}' must be a list in {source}")

        variant_groups: Dict[Optional[str], List[MeasurementRange]] = {}
        unit_map: Dict[str, float] = {}
        base_unit_name: Optional[str] = None

        for entry in entries:
            context = f"{base_mode} range ({entry.get('range_max', '?')})"
            range_value_si, range_unit, range_label, range_factor = parse_quantity(str(entry['range_max']), context)
            resolution_value_si, resolution_unit, resolution_label, resolution_factor = parse_quantity(
                str(entry['resolution']), context
            )

            if base_unit_name is None:
                base_unit_name = range_unit
            elif base_unit_name != range_unit:
                raise RuleLoadError(
                    f"Mixed base units for mode '{base_mode}' in {source}: '{base_unit_name}' vs '{range_unit}'"
                )
            if resolution_unit != range_unit:
                raise RuleLoadError(
                    f"Resolution unit '{resolution_unit}' does not match range unit '{range_unit}' for '{base_mode}'"
                )

            unit_map.setdefault(range_label, range_factor)
            unit_map.setdefault(resolution_label, resolution_factor)

            accuracy_field = entry.get('accuracy')
            if accuracy_field is None:
                raise RuleLoadError(f"Missing accuracy specification for '{base_mode}' in {source}")
            accuracy_options = expand_accuracy_options(accuracy_field, context)

            for variant_label, p_reading, counts, accuracy_text in accuracy_options:
                mode_key = variant_label.strip() if variant_label else None
                measurement_range = MeasurementRange(
                    range_max=range_value_si,
                    resolution=resolution_value_si,
                    p_reading=p_reading,
                    counts=counts,
                    unit=range_unit,
                    display_range=str(entry['range_max']),
                    display_resolution=str(entry['resolution']),
                    accuracy_text=accuracy_text,
                )
                variant_groups.setdefault(mode_key, []).append(measurement_range)

        if not variant_groups:
            raise RuleLoadError(f"No ranges available for mode '{base_mode}' in {source}")
        if base_unit_name is None:
            raise RuleLoadError(f"Could not determine base unit for mode '{base_mode}' in {source}")

        unit_map[base_unit_name] = 1.0
        ordered_units = OrderedDict(sorted(unit_map.items(), key=lambda item: item[1]))

        for variant_label, ranges in variant_groups.items():
            mode_name = base_mode if variant_label is None else f"{base_mode} [{variant_label}]"
            ranges_by_mode[mode_name] = sorted(ranges, key=lambda item: item.range_max)
            units_by_mode[mode_name] = OrderedDict(ordered_units)

    return MeasurementCalculator(ranges_by_mode, units_by_mode)


OPERATOR_KEY_TO_SYMBOLS: Dict[str, Tuple[str, ...]] = {
    'sum_or_difference': ('+', '-'),
    'product_or_quotient': ('*', '/'),
}

ALGORITHM_TO_RULE: Dict[str, Tuple[str, str]] = {
    'absolute_linear': ('absolute', 'linear'),
    'absolute_rss': ('absolute', 'rss'),
    'relative_linear': ('relative', 'linear'),
    'relative_rss': ('relative', 'rss'),
}


def build_propagation_calculator_from_rules(payload: dict, source: str) -> PropagationCalculator:
    families = payload.get('rule_families')
    if not isinstance(families, dict):
        raise RuleLoadError(f"Missing 'rule_families' section in {source}")

    rule_sets: Dict[str, PropagationRuleSet] = {}
    for name, details in families.items():
        operators: Dict[str, OperatorRule] = {}
        operator_entries = details.get('operators', {})
        if isinstance(operator_entries, dict):
            for op_key, info in operator_entries.items():
                algorithm = ((info or {}).get('algorithm') or {}).get('type')
                mapping = ALGORITHM_TO_RULE.get(str(algorithm))
                symbols = OPERATOR_KEY_TO_SYMBOLS.get(op_key, ())
                if not mapping or not symbols:
                    continue
                domain, combine = mapping
                for symbol in symbols:
                    operators[symbol] = OperatorRule(domain=domain, combine=combine)
        rule_sets[name] = PropagationRuleSet(name=name, operators=operators)

    if not rule_sets:
        raise RuleLoadError(f"No propagation rule families found in {source}")
    return PropagationCalculator(rule_sets)


def load_measurement_calculator(path: Path = MEASUREMENT_FILE) -> MeasurementCalculator:
    if not path.exists():
        raise RuleLoadError(f"Cannot find measurement specification file '{path.name}'")
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    return build_measurement_calculator_from_specs(payload, path.name)


def load_propagation_calculator(path: Path = PROPAGATION_FILE) -> PropagationCalculator:
    if not path.exists():
        raise RuleLoadError(f"Cannot find propagation rules file '{path.name}'")
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    return build_propagation_calculator_from_rules(payload, path.name)


def load_rules(
    measurement_path: Path = MEASUREMENT_FILE,
    propagation_path: Path = PROPAGATION_FILE,
) -> Tuple[MeasurementCalculator, PropagationCalculator]:
    return load_measurement_calculator(measurement_path), load_propagation_calculator(propagation_path)



def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        measurement_calc, propagation_calc = load_rules()
    except Exception as exc:  # noqa: BLE001 - present readable failure without assuming a Tk root exists
        print(f"Failed to load rules: {exc}", file=sys.stderr)
        return 1
    app = App(measurement_calc, propagation_calc)
    app.mainloop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
