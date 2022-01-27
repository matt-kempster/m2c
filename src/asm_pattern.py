import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from .parse_file import Label
from .parse_instruction import (
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmLiteral,
    BinOp,
    Instruction,
    InstructionMeta,
    JumpTarget,
    NaiveParsingArch,
    Register,
    parse_instruction,
)


BodyPart = Union[Instruction, Label]
PatternPart = Union[Instruction, Label, None]
Pattern = List[Tuple[PatternPart, bool]]


def make_pattern(*parts: str) -> Pattern:
    ret: Pattern = []
    for part in parts:
        optional = part.endswith("?")
        part = part.rstrip("?")
        if part == "*":
            ret.append((None, optional))
        elif part.endswith(":"):
            ret.append((Label(""), optional))
        else:
            ins = parse_instruction(part, InstructionMeta.missing(), NaiveParsingArch())
            ret.append((ins, optional))
    return ret


@dataclass
class AsmMatch:
    replacement: List[BodyPart]
    num_consumed: int


class AsmPattern(abc.ABC):
    @abc.abstractmethod
    def match(self, matcher: "AsmMatcher") -> Optional[AsmMatch]:
        ...


@dataclass
class TryMatchState:
    symbolic_registers: Dict[str, Register] = field(default_factory=dict)
    symbolic_labels: Dict[str, str] = field(default_factory=dict)
    symbolic_literals: Dict[str, int] = field(default_factory=dict)

    def match_reg(self, actual: Register, exp: Register) -> bool:
        if len(exp.register_name) <= 1:
            if exp.register_name not in self.symbolic_registers:
                self.symbolic_registers[exp.register_name] = actual
            elif self.symbolic_registers[exp.register_name] != actual:
                return False
        elif exp.register_name != actual.register_name:
            return False
        return True

    def eval_math(self, e: Argument) -> int:
        if isinstance(e, AsmLiteral):
            return e.value
        if isinstance(e, BinOp):
            if e.op == "+":
                return self.eval_math(e.lhs) + self.eval_math(e.rhs)
            if e.op == "-":
                return self.eval_math(e.lhs) - self.eval_math(e.rhs)
            if e.op == "<<":
                return self.eval_math(e.lhs) << self.eval_math(e.rhs)
            assert False, f"bad binop in math pattern: {e}"
        elif isinstance(e, AsmGlobalSymbol):
            assert e.symbol_name in self.symbolic_literals, \
                    f"undefined variable in math pattern: {e.symbol_name}"
            return self.symbolic_literals[e.symbol_name]
        else:
            assert False, f"bad pattern part in math pattern: {e}"

    def match_one(self, actual: BodyPart, exp: PatternPart) -> bool:
        if exp is None:
            return True
        if isinstance(exp, Label):
            name = self.symbolic_labels.get(exp.name)
            return isinstance(actual, Label) and (
                name is None or actual.name == name
            )
        if not isinstance(actual, Instruction):
            return False
        ins = actual
        if ins.mnemonic != exp.mnemonic:
            return False
        if exp.args:
            if len(exp.args) != len(ins.args):
                return False
            for (e, a) in zip(exp.args, ins.args):
                if isinstance(e, AsmLiteral):
                    if not isinstance(a, AsmLiteral) or a.value != e.value:
                        return False
                elif isinstance(e, Register):
                    if not isinstance(a, Register) or not self.match_reg(a, e):
                        return False
                elif isinstance(e, AsmGlobalSymbol):
                    if e.symbol_name.isupper():
                        if not isinstance(a, AsmLiteral):
                            return False
                        if e.symbol_name not in self.symbolic_literals:
                            self.symbolic_literals[e.symbol_name] = a.value
                        elif self.symbolic_literals[e.symbol_name] != a.value:
                            return False
                    else:
                        if (
                            not isinstance(a, AsmGlobalSymbol)
                            or a.symbol_name != e.symbol_name
                        ):
                            return False
                elif isinstance(e, AsmAddressMode):
                    if (
                        not isinstance(a, AsmAddressMode)
                        or a.lhs != e.lhs
                        or not self.match_reg(a.rhs, e.rhs)
                    ):
                        return False
                elif isinstance(e, JumpTarget):
                    if not isinstance(a, JumpTarget):
                        return False
                    if e.target not in self.symbolic_labels:
                        self.symbolic_labels[e.target] = a.target
                    elif self.symbolic_labels[e.target] != a.target:
                        return False
                elif isinstance(e, BinOp):
                    if not isinstance(a, AsmLiteral) or a.value != self.eval_math(e):
                        return False
                else:
                    assert False, f"bad pattern part: {exp} contains {type(e)}"
        return True


@dataclass
class AsmMatcher:
    input: List[BodyPart]
    output: List[BodyPart] = field(default_factory=list)
    index: int = 0

    def try_match(self, pattern: Pattern) -> Optional[List[BodyPart]]:
        state = TryMatchState()

        start_index = index = self.index
        for (pat, optional) in pattern:
            if index < len(self.input) and state.match_one(self.input[index], pat):
                index += 1
            elif not optional:
                return None
        return self.input[start_index:index]

    def apply(self, match: AsmMatch) -> None:
        self.output.extend(match.replacement)
        self.index += match.num_consumed


def simplify_patterns(body: List[BodyPart], patterns: List[AsmPattern]) -> List[BodyPart]:
    """Detect and simplify asm standard patterns emitted by known compilers. This is
    especially useful for patterns that involve branches, which are hard to deal with
    in the translate phase."""
    matcher = AsmMatcher(body)
    while matcher.index < len(matcher.input):
        for pattern in patterns:
            m = pattern.match(matcher)
            if m:
                matcher.apply(m)
                break
        else:
            matcher.apply(AsmMatch([matcher.input[matcher.index]], 1))

    return matcher.output
