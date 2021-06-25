import abc
import math
import struct
import sys
import traceback
from contextlib import contextmanager
from copy import copy
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import attr

from .c_types import TypeMap
from .error import DecompFailure
from .flow_graph import (
    FlowGraph,
    Function,
    Node,
    ReturnNode,
    SwitchNode,
    TerminalNode,
    build_flowgraph,
)
from .options import Formatter, Options
from .parse_file import AsmData, AsmDataEntry
from .parse_instruction import (
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmLiteral,
    BinOp,
    Instruction,
    Macro,
    Register,
)
from .types import (
    FunctionParam,
    FunctionSignature,
    Type,
    find_substruct_array,
    get_field,
    ptr_type_from_ctype,
    type_from_ctype,
)

ASSOCIATIVE_OPS: Set[str] = {"+", "&&", "||", "&", "|", "^", "*"}
COMPOUND_ASSIGNMENT_OPS: Set[str] = {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"}

ARGUMENT_REGS: List[Register] = list(
    map(Register, ["a0", "a1", "a2", "a3", "f12", "f14"])
)

SIMPLE_TEMP_REGS: List[Register] = list(
    map(
        Register,
        [
            "v0",
            "v1",
            "t0",
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
            "t7",
            "t8",
            "t9",
            "f0",
            "f1",
            "f2",
            "f3",
            "f4",
            "f5",
            "f6",
            "f7",
            "f8",
            "f9",
            "f10",
            "f11",
            "f13",
            "f15",
            "f16",
            "f17",
            "f18",
            "f19",
        ],
    )
)

TEMP_REGS: List[Register] = (
    ARGUMENT_REGS
    + SIMPLE_TEMP_REGS
    + list(
        map(
            Register,
            [
                "at",
                "hi",
                "lo",
                "condition_bit",
                "return",
            ],
        )
    )
)

SAVED_REGS: List[Register] = list(
    map(
        Register,
        [
            "s0",
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
            "s6",
            "s7",
            "f20",
            "f21",
            "f22",
            "f23",
            "f24",
            "f25",
            "f26",
            "f27",
            "f28",
            "f29",
            "f30",
            "f31",
            "ra",
            "31",
            "fp",
            "gp",
        ],
    )
)


@attr.s
class InstrProcessingFailure(Exception):
    instr: Instruction = attr.ib()

    def __str__(self) -> str:
        return f"Error while processing instruction:\n{self.instr}"


@contextmanager
def current_instr(instr: Instruction) -> Iterator[None]:
    """Mark an instruction as being the one currently processed, for the
    purposes of error messages. Use like |with current_instr(instr): ...|"""
    try:
        yield
    except Exception as e:
        raise InstrProcessingFailure(instr) from e


def as_type(expr: "Expression", type: Type, silent: bool) -> "Expression":
    if expr.type.unify(type):
        if silent or isinstance(expr, Literal):
            return expr
    return Cast(expr=expr, reinterpret=True, silent=False, type=type)


def as_f32(expr: "Expression") -> "Expression":
    return as_type(expr, Type.f32(), True)


def as_f64(expr: "Expression") -> "Expression":
    return as_type(expr, Type.f64(), True)


def as_s32(expr: "Expression", *, silent: bool = False) -> "Expression":
    return as_type(expr, Type.s32(), silent)


def as_u32(expr: "Expression") -> "Expression":
    return as_type(expr, Type.u32(), False)


def as_s64(expr: "Expression", *, silent: bool = False) -> "Expression":
    return as_type(expr, Type.s64(), silent)


def as_u64(expr: "Expression", *, silent: bool = False) -> "Expression":
    return as_type(expr, Type.u64(), silent)


def as_intish(expr: "Expression") -> "Expression":
    return as_type(expr, Type.intish(), True)


def as_int64(expr: "Expression") -> "Expression":
    return as_type(expr, Type.int64(), True)


def as_intptr(expr: "Expression") -> "Expression":
    return as_type(expr, Type.intptr(), True)


def as_ptr(expr: "Expression") -> "Expression":
    return as_type(expr, Type.ptr(), True)


def as_function_ptr(expr: "Expression") -> "Expression":
    return as_type(expr, Type.ptr(Type.function()), True)


@attr.s
class StackInfo:
    function: Function = attr.ib()
    global_info: "GlobalInfo" = attr.ib()
    allocated_stack_size: int = attr.ib(default=0)
    is_leaf: bool = attr.ib(default=True)
    is_variadic: bool = attr.ib(default=False)
    uses_framepointer: bool = attr.ib(default=False)
    subroutine_arg_top: int = attr.ib(default=0)
    return_addr_location: int = attr.ib(default=0)
    callee_save_reg_locations: Dict[Register, int] = attr.ib(factory=dict)
    callee_save_reg_region: Tuple[int, int] = attr.ib(default=(0, 0))
    unique_type_map: Dict[Any, "Type"] = attr.ib(factory=dict)
    local_vars: List["LocalVar"] = attr.ib(factory=list)
    temp_vars: List["EvalOnceStmt"] = attr.ib(factory=list)
    phi_vars: List["PhiExpr"] = attr.ib(factory=list)
    reg_vars: Dict[Register, "RegisterVar"] = attr.ib(factory=dict)
    used_reg_vars: Set[Register] = attr.ib(factory=set)
    arguments: List["PassedInArg"] = attr.ib(factory=list)
    temp_name_counter: Dict[str, int] = attr.ib(factory=dict)
    nonzero_accesses: Set["Expression"] = attr.ib(factory=set)
    param_names: Dict[int, str] = attr.ib(factory=dict)

    def temp_var(self, prefix: str) -> str:
        counter = self.temp_name_counter.get(prefix, 0) + 1
        self.temp_name_counter[prefix] = counter
        return prefix + (f"_{counter}" if counter > 1 else "")

    def in_subroutine_arg_region(self, location: int) -> bool:
        if self.is_leaf:
            return False
        assert self.subroutine_arg_top is not None
        return location < self.subroutine_arg_top

    def in_callee_save_reg_region(self, location: int) -> bool:
        lower_bound, upper_bound = self.callee_save_reg_region
        return lower_bound <= location < upper_bound

    def location_above_stack(self, location: int) -> bool:
        return location >= self.allocated_stack_size

    def add_known_param(self, offset: int, name: Optional[str], type: Type) -> None:
        if name:
            self.param_names[offset] = name
        _, arg = self.get_argument(offset)
        self.add_argument(arg)
        arg.type.unify(type)

    def get_param_name(self, offset: int) -> Optional[str]:
        return self.param_names.get(offset)

    def add_local_var(self, var: "LocalVar") -> None:
        if any(v.value == var.value for v in self.local_vars):
            return
        self.local_vars.append(var)
        # Make sure the local vars stay sorted in order on the stack.
        self.local_vars.sort(key=lambda v: v.value)

    def add_argument(self, arg: "PassedInArg") -> None:
        if any(a.value == arg.value for a in self.arguments):
            return
        self.arguments.append(arg)
        self.arguments.sort(key=lambda a: a.value)

    def get_argument(self, location: int) -> Tuple["Expression", "PassedInArg"]:
        real_location = location & -4
        arg = PassedInArg(
            real_location,
            copied=True,
            stack_info=self,
            type=self.unique_type_for("arg", real_location, Type.any_reg()),
        )
        if real_location == location - 3:
            return as_type(arg, Type.of_size(8), True), arg
        if real_location == location - 2:
            return as_type(arg, Type.of_size(16), True), arg
        return arg, arg

    def record_struct_access(self, ptr: "Expression", location: int) -> None:
        if location:
            self.nonzero_accesses.add(unwrap_deep(ptr))

    def has_nonzero_access(self, ptr: "Expression") -> bool:
        return unwrap_deep(ptr) in self.nonzero_accesses

    def unique_type_for(self, category: str, key: Any, default: Type) -> "Type":
        key = (category, key)
        if key not in self.unique_type_map:
            self.unique_type_map[key] = default
        return self.unique_type_map[key]

    def saved_reg_symbol(self, reg_name: str) -> "GlobalSymbol":
        sym_name = "saved_reg_" + reg_name
        type = self.unique_type_for("saved_reg", sym_name, Type.any_reg())
        return GlobalSymbol(symbol_name=sym_name, type=type)

    def should_save(self, expr: "Expression", offset: Optional[int]) -> bool:
        if isinstance(expr, GlobalSymbol) and expr.symbol_name.startswith("saved_reg_"):
            return True
        if (
            isinstance(expr, PassedInArg)
            and not expr.copied
            and (offset is None or offset == self.allocated_stack_size + expr.value)
        ):
            return True
        return False

    def get_stack_var(self, location: int, *, store: bool) -> "Expression":
        # See `get_stack_info` for explanation
        if self.location_above_stack(location):
            ret, arg = self.get_argument(location - self.allocated_stack_size)
            if not store:
                self.add_argument(arg)
            return ret
        elif self.in_subroutine_arg_region(location):
            return SubroutineArg(location, type=Type.any_reg())
        elif self.in_callee_save_reg_region(location):
            # Some annoying bookkeeping instruction. To avoid
            # further special-casing, just return whatever - it won't matter.
            return LocalVar(location, type=Type.any_reg())
        else:
            # Local variable
            return LocalVar(
                location, type=self.unique_type_for("stack", location, Type.any_reg())
            )

    def maybe_get_register_var(self, reg: Register) -> Optional["RegisterVar"]:
        return self.reg_vars.get(reg)

    def add_register_var(self, reg: Register) -> None:
        type = Type.floatish() if reg.is_float() else Type.intptr()
        self.reg_vars[reg] = RegisterVar(reg, type)

    def use_register_var(self, var: "RegisterVar") -> None:
        self.used_reg_vars.add(var.register)

    def is_stack_reg(self, reg: Register) -> bool:
        if reg.register_name == "sp":
            return True
        if reg.register_name == "fp":
            return self.uses_framepointer
        return False

    def get_struct_type_map(self) -> Dict["Expression", Dict[int, Type]]:
        """Reorganize struct information in unique_type_map by var & offset"""
        struct_type_map: Dict[Expression, Dict[int, Type]] = {}
        for (category, key), type in self.unique_type_map.items():
            if category != "struct":
                continue
            var, offset = key
            if var not in struct_type_map:
                struct_type_map[var] = {}
            struct_type_map[var][offset] = type
        return struct_type_map

    def __str__(self) -> str:
        return "\n".join(
            [
                f"Stack info for function {self.function.name}:",
                f"Allocated stack size: {self.allocated_stack_size}",
                f"Leaf? {self.is_leaf}",
                f"Bounds of callee-saved vars region: {self.callee_save_reg_locations}",
                f"Location of return addr: {self.return_addr_location}",
                f"Locations of callee save registers: {self.callee_save_reg_locations}",
            ]
        )


def get_stack_info(
    function: Function,
    global_info: "GlobalInfo",
    flow_graph: FlowGraph,
) -> StackInfo:
    info = StackInfo(function, global_info)

    # The goal here is to pick out special instructions that provide information
    # about this function's stack setup.
    #
    # IDO puts local variables *above* the saved registers on the stack, but
    # GCC puts local variables *below* the saved registers.
    # To support both, we explicitly determine both the upper & lower bounds of the
    # saved registers. Then, we estimate the boundary of the subroutine arguments
    # by finding the lowest stack offset that is loaded from or computed. (This
    # assumes that the compiler will never reuse a section of stack for *both*
    # a local variable *and* a subroutine argument.) Anything within the stack frame,
    # but outside of these two regions, is considered a local variable.
    callee_saved_offset_and_size: List[Tuple[int, int]] = []
    for inst in flow_graph.entry_node().block.instructions:
        if not inst.args or not isinstance(inst.args[0], Register):
            continue

        destination = inst.args[0]

        if inst.mnemonic == "jal":
            break
        elif inst.mnemonic == "addiu" and destination.register_name == "sp":
            # Moving the stack pointer.
            assert isinstance(inst.args[2], AsmLiteral)
            info.allocated_stack_size = abs(inst.args[2].signed_value())
        elif (
            inst.mnemonic == "move"
            and destination.register_name == "fp"
            and isinstance(inst.args[1], Register)
            and inst.args[1].register_name == "sp"
        ):
            # "move fp, sp" very likely means the code is compiled with frame
            # pointers enabled; thus fp should be treated the same as sp.
            info.uses_framepointer = True
        elif (
            inst.mnemonic == "sw"
            and destination.register_name == "ra"
            and isinstance(inst.args[1], AsmAddressMode)
            and inst.args[1].rhs.register_name == "sp"
            and info.is_leaf
        ):
            # Saving the return address on the stack.
            info.is_leaf = False
            stack_offset = inst.args[1].lhs_as_literal()
            info.return_addr_location = stack_offset
            callee_saved_offset_and_size.append((stack_offset, 4))
        elif (
            inst.mnemonic in ["sw", "swc1", "sdc1"]
            and destination in SAVED_REGS
            and isinstance(inst.args[1], AsmAddressMode)
            and inst.args[1].rhs.register_name == "sp"
            and destination not in info.callee_save_reg_locations
        ):
            # Initial saving of callee-save register onto the stack.
            stack_offset = inst.args[1].lhs_as_literal()
            info.callee_save_reg_locations[destination] = stack_offset
            callee_saved_offset_and_size.append(
                (stack_offset, 8 if inst.mnemonic == "sdc1" else 4)
            )

    if not info.is_leaf:
        # Iterate over the whole function, not just the first basic block,
        # to estimate the boundary for the subroutine argument region
        info.subroutine_arg_top = info.allocated_stack_size
        for node in flow_graph.nodes:
            for inst in node.block.instructions:
                if not inst.args or not isinstance(inst.args[0], Register):
                    continue
                destination = inst.args[0]

                if (
                    inst.mnemonic in ["lw", "lwc1", "ldc1"]
                    and isinstance(inst.args[1], AsmAddressMode)
                    and inst.args[1].rhs.register_name == "sp"
                    and inst.args[1].lhs_as_literal() >= 16
                ):
                    info.subroutine_arg_top = min(
                        info.subroutine_arg_top, inst.args[1].lhs_as_literal()
                    )
                elif (
                    inst.mnemonic == "addiu"
                    and destination.register_name != "sp"
                    and inst.args[1] == Register("sp")
                    and isinstance(inst.args[2], AsmLiteral)
                    and inst.args[2].value < info.allocated_stack_size
                ):
                    info.subroutine_arg_top = min(
                        info.subroutine_arg_top, inst.args[2].value
                    )

        # Compute the bounds of the callee-saved register region, including padding
        callee_saved_offset_and_size.sort()
        bottom, last_size = callee_saved_offset_and_size[0]

        # Both IDO & GCC save registers in two subregions:
        # (a) One for double-sized registers
        # (b) One for word-sized registers, padded to a multiple of 8 bytes
        # IDO has (a) lower than (b); GCC has (b) lower than (a)
        # Check that there are no gaps in this region, other than a single
        # 4-byte word between subregions.
        top = bottom
        internal_padding_added = False
        for offset, size in callee_saved_offset_and_size:
            if offset != top:
                if (
                    not internal_padding_added
                    and size != last_size
                    and offset == top + 4
                ):
                    internal_padding_added = True
                else:
                    raise DecompFailure(
                        f"Gap in callee-saved word stack region. "
                        f"Saved: {callee_saved_offset_and_size}, "
                        f"gap at: {offset} != {top}."
                    )
            top = offset + size
            last_size = size
        # Expand boundaries to multiples of 8 bytes
        info.callee_save_reg_region = (bottom, top)

        # Subroutine arguments must be at the very bottom of the stack, so they
        # must come after the callee-saved region
        info.subroutine_arg_top = min(info.subroutine_arg_top, bottom)

    return info


def format_hex(val: int) -> str:
    return format(val, "x").upper()


def escape_byte(b: int) -> bytes:
    table = {
        b"\0": b"\\0",
        b"\b": b"\\b",
        b"\f": b"\\f",
        b"\n": b"\\n",
        b"\r": b"\\r",
        b"\t": b"\\t",
        b"\v": b"\\v",
        b"\\": b"\\\\",
        b'"': b'\\"',
    }
    bs = bytes([b])
    if bs in table:
        return table[bs]
    if b < 0x20 or b in (0xFF, 0x7F):
        return f"\\x{b:02x}".encode("ascii")
    return bs


@attr.s(eq=False)
class Var:
    stack_info: StackInfo = attr.ib(repr=False)
    prefix: str = attr.ib()
    num_usages: int = attr.ib(default=0)
    name: Optional[str] = attr.ib(default=None)

    def format(self, fmt: Formatter) -> str:
        if self.name is None:
            self.name = self.stack_info.temp_var(self.prefix)
        return self.name

    def __str__(self) -> str:
        return "<temp>"


class Expression(abc.ABC):
    type: Type

    @abc.abstractmethod
    def dependencies(self) -> List["Expression"]:
        ...

    def use(self) -> None:
        """Mark an expression as "will occur in the output". Various subclasses
        override this to provide special behavior; for instance, EvalOnceExpr
        checks if it occurs more than once in the output and if so emits a temp.
        It is important to get the number of use() calls correct:
        * if use() is called but the expression is not emitted, it may cause
          function calls to be silently dropped.
        * if use() is not called but the expression is emitted, it may cause phi
          variables to be printed as unnamed-phi($reg), without any assignment
          to that phi.
        * if use() is called once but the expression is emitted twice, it may
          cause function calls to be duplicated."""
        for expr in self.dependencies():
            expr.use()

    @abc.abstractmethod
    def format(self, fmt: Formatter) -> str:
        ...

    def __str__(self) -> str:
        """Stringify an expression for debug purposes. The output can change
        depending on when this is called, e.g. because of EvalOnceExpr state.
        To avoid using it by accident, output is quoted."""
        fmt = Formatter(debug=True)
        return '"' + self.format(fmt) + '"'


class Condition(Expression):
    @abc.abstractmethod
    def negated(self) -> "Condition":
        ...


class Statement(abc.ABC):
    @abc.abstractmethod
    def should_write(self) -> bool:
        ...

    @abc.abstractmethod
    def format(self, fmt: Formatter) -> str:
        ...

    def __str__(self) -> str:
        """Stringify a statement for debug purposes. The output can change
        depending on when this is called, e.g. because of EvalOnceExpr state.
        To avoid using it by accident, output is quoted."""
        fmt = Formatter(debug=True)
        return '"' + self.format(fmt) + '"'


@attr.s(frozen=True, eq=False)
class ErrorExpr(Condition):
    desc: Optional[str] = attr.ib(default=None)
    type: Type = attr.ib(factory=Type.any_reg)

    def dependencies(self) -> List[Expression]:
        return []

    def negated(self) -> "Condition":
        return self

    def format(self, fmt: Formatter) -> str:
        if self.desc is not None:
            return f"MIPS2C_ERROR({self.desc})"
        return "MIPS2C_ERROR()"


@attr.s(frozen=True, eq=False)
class SecondF64Half(Expression):
    type: Type = attr.ib(factory=Type.any_reg)

    def dependencies(self) -> List[Expression]:
        return []

    def format(self, fmt: Formatter) -> str:
        return "(second half of f64)"


@attr.s(frozen=True, eq=False)
class BinaryOp(Condition):
    left: Expression = attr.ib()
    op: str = attr.ib()
    right: Expression = attr.ib()
    type: Type = attr.ib()
    floating: bool = attr.ib(default=False)

    @staticmethod
    def int(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(
            left=as_intish(left), op=op, right=as_intish(right), type=Type.intish()
        )

    @staticmethod
    def int64(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(
            left=as_int64(left), op=op, right=as_int64(right), type=Type.int64()
        )

    @staticmethod
    def intptr(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(
            left=as_intptr(left), op=op, right=as_intptr(right), type=Type.intptr()
        )

    @staticmethod
    def icmp(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(
            left=as_intptr(left), op=op, right=as_intptr(right), type=Type.bool()
        )

    @staticmethod
    def scmp(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(
            left=as_s32(left, silent=True),
            op=op,
            right=as_s32(right, silent=True),
            type=Type.bool(),
        )

    @staticmethod
    def ucmp(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(left=as_u32(left), op=op, right=as_u32(right), type=Type.bool())

    @staticmethod
    def fcmp(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(
            left=as_f32(left),
            op=op,
            right=as_f32(right),
            type=Type.bool(),
            floating=True,
        )

    @staticmethod
    def dcmp(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(
            left=as_f64(left),
            op=op,
            right=as_f64(right),
            type=Type.bool(),
            floating=True,
        )

    @staticmethod
    def s32(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(left=as_s32(left), op=op, right=as_s32(right), type=Type.s32())

    @staticmethod
    def u32(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(left=as_u32(left), op=op, right=as_u32(right), type=Type.u32())

    @staticmethod
    def s64(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(left=as_s64(left), op=op, right=as_s64(right), type=Type.s64())

    @staticmethod
    def u64(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(left=as_u64(left), op=op, right=as_u64(right), type=Type.u64())

    @staticmethod
    def f32(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(
            left=as_f32(left),
            op=op,
            right=as_f32(right),
            type=Type.f32(),
            floating=True,
        )

    @staticmethod
    def f64(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(
            left=as_f64(left),
            op=op,
            right=as_f64(right),
            type=Type.f64(),
            floating=True,
        )

    def is_boolean(self) -> bool:
        return self.op in ["==", "!=", ">", "<", ">=", "<="]

    def negated(self) -> "Condition":
        if not self.is_boolean() or (
            self.floating and self.op in ["<", ">", "<=", ">="]
        ):
            # Floating-point comparisons cannot be negated in any nice way,
            # due to nans.
            return UnaryOp("!", self, type=Type.bool())
        return BinaryOp(
            left=self.left,
            op={"==": "!=", "!=": "==", ">": "<=", "<": ">=", ">=": "<", "<=": ">"}[
                self.op
            ],
            right=self.right,
            type=Type.bool(),
        )

    def dependencies(self) -> List[Expression]:
        return [self.left, self.right]

    def format(self, fmt: Formatter) -> str:
        if (
            self.is_boolean()
            and isinstance(self.left, Literal)
            and not isinstance(self.right, Literal)
        ):
            return BinaryOp(
                left=self.right,
                op=self.op.translate(str.maketrans("<>", "><")),
                right=self.left,
                type=self.type,
            ).format(fmt)

        if (
            not self.floating
            and isinstance(self.right, Literal)
            and self.right.value < 0
        ):
            if self.op == "+":
                neg = Literal(value=-self.right.value, type=self.right.type)
                sub = BinaryOp(op="-", left=self.left, right=neg, type=self.type)
                return sub.format(fmt)
            if self.op in ("&", "|"):
                neg = Literal(value=~self.right.value, type=self.right.type)
                right = UnaryOp("~", neg, type=Type.any_reg())
                expr = BinaryOp(op=self.op, left=self.left, right=right, type=self.type)
                return expr.format(fmt)

        # For commutative, left-associative operations, strip unnecessary parentheses.
        left_expr = late_unwrap(self.left)
        lhs = left_expr.format(fmt)
        if (
            isinstance(left_expr, BinaryOp)
            and left_expr.op == self.op
            and self.op in ASSOCIATIVE_OPS
        ):
            lhs = lhs[1:-1]

        return f"({lhs} {self.op} {self.right.format(fmt)})"


@attr.s(frozen=True, eq=False)
class UnaryOp(Condition):
    op: str = attr.ib()
    expr: Expression = attr.ib()
    type: Type = attr.ib()

    def dependencies(self) -> List[Expression]:
        return [self.expr]

    def negated(self) -> "Condition":
        if self.op == "!" and isinstance(self.expr, (UnaryOp, BinaryOp)):
            return self.expr
        return UnaryOp("!", self, type=Type.bool())

    def format(self, fmt: Formatter) -> str:
        return f"{self.op}{self.expr.format(fmt)}"


@attr.s(frozen=True, eq=False)
class ExprCondition(Condition):
    expr: Expression = attr.ib()
    type: Type = attr.ib()
    is_negated: bool = attr.ib(default=False)

    def dependencies(self) -> List[Expression]:
        return [self.expr]

    def negated(self) -> "Condition":
        return ExprCondition(self.expr, self.type, not self.is_negated)

    def format(self, fmt: Formatter) -> str:
        neg = "!" if self.is_negated else ""
        return f"{neg}{self.expr.format(fmt)}"


@attr.s(frozen=True, eq=False)
class CommaConditionExpr(Condition):
    statements: List["Statement"] = attr.ib()
    condition: "Condition" = attr.ib()
    type: Type = Type.bool()

    def dependencies(self) -> List[Expression]:
        assert False, "CommaConditionExpr should not be used within translate.py"
        return []

    def negated(self) -> "Condition":
        return CommaConditionExpr(self.statements, self.condition.negated())

    def format(self, fmt: Formatter) -> str:
        comma_joined = ", ".join(
            stmt.format(fmt).rstrip(";") for stmt in self.statements
        )
        return f"({comma_joined}, {self.condition.format(fmt)})"


@attr.s(frozen=True, eq=False)
class Cast(Expression):
    expr: Expression = attr.ib()
    type: Type = attr.ib()
    reinterpret: bool = attr.ib(default=False)
    silent: bool = attr.ib(default=True)

    def dependencies(self) -> List[Expression]:
        return [self.expr]

    def use(self) -> None:
        # Try to unify, to make stringification output better.
        self.expr.type.unify(self.type)
        super().use()

    def needed_for_store(self) -> bool:
        if not self.reinterpret:
            # int <-> float casts should be emitted even for stores.
            return True
        if not self.expr.type.unify(self.type):
            # Emit casts when types fail to unify.
            return True
        return False

    def is_trivial(self) -> bool:
        return (
            self.reinterpret
            and self.expr.type.is_float() == self.type.is_float()
            and is_trivial_expression(self.expr)
        )

    def format(self, fmt: Formatter) -> str:
        if self.reinterpret and self.expr.type.is_float() != self.type.is_float():
            # This shouldn't happen, but mark it in the output if it does.
            if fmt.valid_syntax:
                return (
                    f"MIPS2C_BITWISE({self.type.format(fmt)}, {self.expr.format(fmt)})"
                )
            return f"(bitwise {self.type.format(fmt)}) {self.expr.format(fmt)}"
        if self.reinterpret and (
            self.silent
            or (is_type_obvious(self.expr) and self.expr.type.unify(self.type))
        ):
            return self.expr.format(fmt)
        if fmt.skip_casts:
            return self.expr.format(fmt)

        # Function casts require special logic because function calls have
        # higher precedence than casts
        fn_sig = self.type.get_function_pointer_signature()
        if fn_sig:
            prototype_sig = self.expr.type.get_function_pointer_signature()
            if not prototype_sig or not prototype_sig.unify_with_args(fn_sig):
                # A function pointer cast is required if the inner expr is not
                # a function pointer, or has incompatible argument types
                return f"(({self.type.format(fmt)}) {self.expr.format(fmt)})"
            if not prototype_sig.return_type.unify(fn_sig.return_type):
                # Only cast the return value of the call
                return f"({fn_sig.return_type.format(fmt)}) {self.expr.format(fmt)}"
            # No cast needed
            return self.expr.format(fmt)

        return f"({self.type.format(fmt)}) {self.expr.format(fmt)}"


@attr.s(frozen=True, eq=False)
class FuncCall(Expression):
    function: Expression = attr.ib()
    args: List[Expression] = attr.ib()
    type: Type = attr.ib()

    def dependencies(self) -> List[Expression]:
        return self.args + [self.function]

    def format(self, fmt: Formatter) -> str:
        # TODO: The function type may have a different number of params than it had
        # when the FuncCall was created. Should we warn that there may be the wrong
        # number of arguments at this callsite?
        args = ", ".join(format_expr(arg, fmt) for arg in self.args)
        return f"{self.function.format(fmt)}({args})"


@attr.s(frozen=True, eq=True)
class LocalVar(Expression):
    value: int = attr.ib()
    type: Type = attr.ib(eq=False)

    def dependencies(self) -> List[Expression]:
        return []

    def format(self, fmt: Formatter) -> str:
        return f"sp{format_hex(self.value)}"


@attr.s(frozen=True, eq=False)
class RegisterVar(Expression):
    register: Register = attr.ib()
    type: Type = attr.ib()

    def dependencies(self) -> List[Expression]:
        return []

    def format(self, fmt: Formatter) -> str:
        return self.register.register_name


@attr.s(frozen=True, eq=True)
class PassedInArg(Expression):
    value: int = attr.ib()
    copied: bool = attr.ib(eq=False)
    stack_info: StackInfo = attr.ib(eq=False, repr=False)
    type: Type = attr.ib(eq=False)

    def dependencies(self) -> List[Expression]:
        return []

    def format(self, fmt: Formatter) -> str:
        assert self.value % 4 == 0
        name = self.stack_info.get_param_name(self.value)
        return name or f"arg{format_hex(self.value // 4)}"


@attr.s(frozen=True, eq=True)
class SubroutineArg(Expression):
    value: int = attr.ib()
    type: Type = attr.ib(eq=False)

    def dependencies(self) -> List[Expression]:
        return []

    def format(self, fmt: Formatter) -> str:
        return f"subroutine_arg{format_hex(self.value // 4)}"


@attr.s(eq=True, hash=True)
class StructAccess(Expression):
    # Represents struct_var->offset.
    # This has eq=True since it represents a live expression and not an access
    # at a certain point in time -- this sometimes helps get rid of phi nodes.
    # prevent_later_uses makes sure it's not used after writes/function calls
    # that may invalidate it.
    struct_var: Expression = attr.ib()
    offset: int = attr.ib()
    target_size: Optional[int] = attr.ib()
    field_name: Optional[str] = attr.ib(eq=False)
    stack_info: StackInfo = attr.ib(eq=False, repr=False)
    type: Type = attr.ib(eq=False)
    has_late_field_name: bool = attr.ib(default=False, eq=False)

    def dependencies(self) -> List[Expression]:
        return [self.struct_var]

    def late_field_name(self) -> Optional[str]:
        # If we didn't have a type at the time when the struct access was
        # constructed, but now we do, compute field name.
        if (
            self.field_name is None
            and self.stack_info.global_info.typemap
            and not self.has_late_field_name
        ):
            var = late_unwrap(self.struct_var)
            self.field_name = get_field(
                var.type,
                self.offset,
                self.stack_info.global_info.typemap,
                target_size=self.target_size,
            )[0]
            self.has_late_field_name = True
        return self.field_name

    def late_has_known_type(self) -> bool:
        if self.late_field_name() is not None:
            return True
        if self.offset == 0 and self.stack_info.global_info.typemap:
            var = late_unwrap(self.struct_var)
            if (
                not self.stack_info.has_nonzero_access(var)
                and isinstance(var, AddressOf)
                and isinstance(var.expr, GlobalSymbol)
                and var.expr.symbol_name
                in self.stack_info.global_info.typemap.var_types
            ):
                return True
        return False

    def format(self, fmt: Formatter) -> str:
        var = late_unwrap(self.struct_var)
        has_nonzero_access = self.stack_info.has_nonzero_access(var)

        field_name = self.late_field_name()

        if field_name:
            has_nonzero_access = True
        elif fmt.valid_syntax and (self.offset != 0 or has_nonzero_access):
            offset_str = (
                f"0x{format_hex(self.offset)}" if self.offset > 0 else f"{self.offset}"
            )
            return f"MIPS2C_FIELD({var.format(fmt)}, {Type.ptr(self.type).format(fmt)}, {offset_str})"
        else:
            field_name = "unk" + format_hex(self.offset)

        if isinstance(var, AddressOf):
            if self.offset == 0 and not has_nonzero_access:
                return f"{var.expr.format(fmt)}"
            else:
                return f"{parenthesize_for_struct_access(var.expr, fmt)}.{field_name}"
        else:
            if self.offset == 0 and not has_nonzero_access:
                return f"*{var.format(fmt)}"
            else:
                return f"{parenthesize_for_struct_access(var, fmt)}->{field_name}"


@attr.s(frozen=True, eq=True)
class ArrayAccess(Expression):
    # Represents ptr[index]. eq=True for symmetry with StructAccess.
    ptr: Expression = attr.ib()
    index: Expression = attr.ib()
    type: Type = attr.ib(eq=False)

    def dependencies(self) -> List[Expression]:
        return [self.ptr, self.index]

    def format(self, fmt: Formatter) -> str:
        base = parenthesize_for_struct_access(self.ptr, fmt)
        index = format_expr(self.index, fmt)
        return f"{base}[{index}]"


@attr.s(frozen=True, eq=True)
class GlobalSymbol(Expression):
    symbol_name: str = attr.ib()
    type: Type = attr.ib(eq=False)

    def dependencies(self) -> List[Expression]:
        return []

    def format(self, fmt: Formatter) -> str:
        return self.symbol_name


@attr.s(frozen=True, eq=True)
class Literal(Expression):
    value: int = attr.ib()
    type: Type = attr.ib(eq=False, factory=Type.any)

    def dependencies(self) -> List[Expression]:
        return []

    def format(self, fmt: Formatter) -> str:
        if self.type.is_float():
            if self.type.get_size_bits() == 32:
                return format_f32_imm(self.value) + "f"
            else:
                return format_f64_imm(self.value)
        if self.type.is_pointer() and self.value == 0:
            return "NULL"

        prefix = ""
        suffix = ""
        if not fmt.skip_casts:
            if self.type.is_pointer():
                prefix = "(void *)"
            elif self.type.get_size_bits() == 8:
                prefix = "(u8)"
            elif self.type.get_size_bits() == 16:
                prefix = "(u16)"
            if self.type.is_unsigned():
                suffix = "U"
        mid = (
            str(self.value)
            if abs(self.value) < 10
            else hex(self.value).upper().replace("X", "x")
        )
        return prefix + mid + suffix


@attr.s(frozen=True)
class StringLiteral(Expression):
    data: bytes = attr.ib()
    type: Type = attr.ib()

    def dependencies(self) -> List[Expression]:
        return []

    def format(self, fmt: Formatter) -> str:
        has_trailing_null = False
        data = self.data
        while data and data[-1] == 0:
            data = data[:-1]
            has_trailing_null = True
        data = b"".join(map(escape_byte, data))

        strdata = data.decode("utf-8", "backslashreplace")
        ret = f'"{strdata}"'
        if not has_trailing_null:
            ret += " /* not null-terminated */"
        return ret


@attr.s(frozen=True, eq=True)
class AddressOf(Expression):
    expr: Expression = attr.ib()
    type: Type = attr.ib(eq=False, factory=Type.ptr)

    def dependencies(self) -> List[Expression]:
        return [self.expr]

    def format(self, fmt: Formatter) -> str:
        if self.expr.type.is_function():
            # Functions are automatically converted to function pointers
            # without an explicit `&` by the compiler
            return f"{self.expr.format(fmt)}"
        return f"&{self.expr.format(fmt)}"


@attr.s(frozen=True)
class Lwl(Expression):
    load_expr: Expression = attr.ib()
    key: Tuple[int, object] = attr.ib()
    type: Type = attr.ib(eq=False, factory=Type.any_reg)

    def dependencies(self) -> List[Expression]:
        return [self.load_expr]

    def format(self, fmt: Formatter) -> str:
        return f"MIPS2C_LWL({self.load_expr.format(fmt)})"


@attr.s(frozen=True)
class Load3Bytes(Expression):
    load_expr: Expression = attr.ib()
    type: Type = attr.ib(eq=False, factory=Type.any_reg)

    def dependencies(self) -> List[Expression]:
        return [self.load_expr]

    def format(self, fmt: Formatter) -> str:
        if fmt.valid_syntax:
            return f"MIPS2C_FIRST3BYTES({self.load_expr.format(fmt)})"
        return f"(first 3 bytes) {self.load_expr.format(fmt)}"


@attr.s(frozen=True)
class UnalignedLoad(Expression):
    load_expr: Expression = attr.ib()
    type: Type = attr.ib(eq=False, factory=Type.any_reg)

    def dependencies(self) -> List[Expression]:
        return [self.load_expr]

    def format(self, fmt: Formatter) -> str:
        if fmt.valid_syntax:
            return f"MIPS2C_UNALIGNED32({self.load_expr.format(fmt)})"
        return f"(unaligned s32) {self.load_expr.format(fmt)}"


@attr.s(frozen=False, eq=False)
class EvalOnceExpr(Expression):
    wrapped_expr: Expression = attr.ib()
    var: Var = attr.ib()
    type: Type = attr.ib()

    # True for function calls/errors
    emit_exactly_once: bool = attr.ib()

    # Mutable state:

    # True if this EvalOnceExpr should be totally transparent and not emit a variable,
    # It may dynamically change from true to false due to forced emissions.
    # Initially, it is based on is_trivial_expression.
    trivial: bool = attr.ib()

    # True if this EvalOnceExpr is wrapped by a ForceVarExpr which has been triggered.
    # This state really live in ForceVarExpr, but there's a hack in RegInfo.__getitem__
    # where we strip off ForceVarExpr's... This is a mess, sorry. :(
    forced_emit: bool = attr.ib(default=False)

    # The number of expressions that depend on this EvalOnceExpr; we emit a variable
    # if this is > 1.
    num_usages: int = attr.ib(default=0)

    def dependencies(self) -> List[Expression]:
        # (this is a bit iffy since state can change over time, but improves uses_expr)
        if self.need_decl():
            return []
        return [self.wrapped_expr]

    def use(self) -> None:
        self.num_usages += 1
        if self.trivial or (self.num_usages == 1 and not self.emit_exactly_once):
            self.wrapped_expr.use()

    def need_decl(self) -> bool:
        return self.num_usages > 1 and not self.trivial

    def format(self, fmt: Formatter) -> str:
        if not self.need_decl():
            return self.wrapped_expr.format(fmt)
        else:
            return self.var.format(fmt)


@attr.s(eq=False)
class ForceVarExpr(Expression):
    wrapped_expr: EvalOnceExpr = attr.ib()
    type: Type = attr.ib()

    def dependencies(self) -> List[Expression]:
        return [self.wrapped_expr]

    def use(self) -> None:
        # Transition the EvalOnceExpr to non-trivial, and mark it as used
        # multiple times to force a var.
        # TODO: If it was originally trivial, we may previously have marked its
        # wrappee used multiple times, even though we now know that it should
        # have been marked just once... We could fix that by moving marking of
        # trivial EvalOnceExpr's to the very end. At least the consequences of
        # getting this wrong are pretty mild -- it just causes extraneous var
        # emission in rare cases.
        self.wrapped_expr.trivial = False
        self.wrapped_expr.forced_emit = True
        self.wrapped_expr.use()
        self.wrapped_expr.use()

    def format(self, fmt: Formatter) -> str:
        return self.wrapped_expr.format(fmt)


@attr.s(frozen=False, eq=False)
class PhiExpr(Expression):
    reg: Register = attr.ib()
    node: Node = attr.ib()
    type: Type = attr.ib()
    used_phis: List["PhiExpr"] = attr.ib()
    name: Optional[str] = attr.ib(default=None)
    num_usages: int = attr.ib(default=0)
    replacement_expr: Optional[Expression] = attr.ib(default=None)
    used_by: Optional["PhiExpr"] = attr.ib(default=None)

    def dependencies(self) -> List[Expression]:
        return []

    def get_var_name(self) -> str:
        return self.name or f"unnamed-phi({self.reg.register_name})"

    def use(self, from_phi: Optional["PhiExpr"] = None) -> None:
        if self.num_usages == 0:
            self.used_phis.append(self)
            self.used_by = from_phi
        self.num_usages += 1
        if self.used_by != from_phi:
            self.used_by = None
        if self.replacement_expr is not None:
            self.replacement_expr.use()

    def propagates_to(self) -> "PhiExpr":
        """Compute the phi that stores to this phi should propagate to. This is
        usually the phi itself, but if the phi is only once for the purpose of
        computing another phi, we forward the store there directly. This is
        admittedly a bit sketchy, in case the phi is in scope here and used
        later on... but we have that problem with regular phi assignments as
        well."""
        if self.used_by is None:
            return self
        return self.used_by.propagates_to()

    def format(self, fmt: Formatter) -> str:
        if self.replacement_expr:
            return self.replacement_expr.format(fmt)
        return self.get_var_name()


@attr.s
class EvalOnceStmt(Statement):
    expr: EvalOnceExpr = attr.ib()

    def need_decl(self) -> bool:
        return self.expr.need_decl()

    def should_write(self) -> bool:
        if self.expr.emit_exactly_once:
            return self.expr.num_usages != 1
        else:
            return self.need_decl()

    def format(self, fmt: Formatter) -> str:
        val_str = format_expr(elide_casts_for_store(self.expr.wrapped_expr), fmt)
        if self.expr.emit_exactly_once and self.expr.num_usages == 0:
            return f"{val_str};"
        return f"{self.expr.var.format(fmt)} = {val_str};"


@attr.s
class SetPhiStmt(Statement):
    phi: PhiExpr = attr.ib()
    expr: Expression = attr.ib()

    def should_write(self) -> bool:
        expr = self.expr
        if isinstance(expr, PhiExpr) and expr.propagates_to() != expr:
            # When we have phi1 = phi2, and phi2 is only used in this place,
            # the SetPhiStmt for phi2 will store directly to phi1 and we can
            # skip this store.
            assert expr.propagates_to() == self.phi.propagates_to()
            return False
        if unwrap_deep(expr) == self.phi.propagates_to():
            # Elide "phi = phi".
            return False
        return True

    def format(self, fmt: Formatter) -> str:
        return format_assignment(self.phi.propagates_to(), self.expr, fmt)


@attr.s
class ExprStmt(Statement):
    expr: Expression = attr.ib()

    def should_write(self) -> bool:
        return True

    def format(self, fmt: Formatter) -> str:
        return f"{format_expr(self.expr, fmt)};"


@attr.s
class StoreStmt(Statement):
    source: Expression = attr.ib()
    dest: Expression = attr.ib()

    def should_write(self) -> bool:
        return True

    def format(self, fmt: Formatter) -> str:
        dest = self.dest
        source = self.source
        if (
            isinstance(dest, StructAccess) and dest.late_has_known_type()
        ) or isinstance(dest, (ArrayAccess, LocalVar, RegisterVar, SubroutineArg)):
            # Known destination; fine to elide some casts.
            source = elide_casts_for_store(source)
        return format_assignment(dest, source, fmt)


@attr.s
class CommentStmt(Statement):
    contents: str = attr.ib()

    def should_write(self) -> bool:
        return True

    def format(self, fmt: Formatter) -> str:
        return f"// {self.contents}"


@attr.s(frozen=True)
class AddressMode:
    offset: int = attr.ib()
    rhs: Register = attr.ib()

    def __str__(self) -> str:
        if self.offset:
            return f"{self.offset}({self.rhs})"
        else:
            return f"({self.rhs})"


@attr.s(frozen=True)
class RawSymbolRef:
    offset: int = attr.ib()
    sym: AsmGlobalSymbol = attr.ib()

    def __str__(self) -> str:
        if self.offset:
            return f"{self.sym.symbol_name} + {self.offset}"
        else:
            return self.sym.symbol_name


@attr.s
class RegInfo:
    contents: Dict[Register, Expression] = attr.ib()
    stack_info: StackInfo = attr.ib(repr=False)

    def __getitem__(self, key: Register) -> Expression:
        if key == Register("zero"):
            return Literal(0)
        ret = self.get_raw(key)
        if ret is None:
            return ErrorExpr(f"Read from unset register {key}")
        if isinstance(ret, PassedInArg) and not ret.copied:
            # Create a new argument object to better distinguish arguments we
            # are called with from arguments passed to subroutines. Also, unify
            # the argument's type with what we can guess from the register used.
            val, arg = self.stack_info.get_argument(ret.value)
            self.stack_info.add_argument(arg)
            val.type.unify(ret.type)
            return val
        if isinstance(ret, ForceVarExpr):
            # Some of the logic in this file is unprepared to deal with
            # ForceVarExpr transparent wrappers... so for simplicity, we mark
            # it used and return the wrappee. Not optimal (what if the value
            # isn't used after all?), but it works decently well.
            ret.use()
            ret = ret.wrapped_expr
        return ret

    def __contains__(self, key: Register) -> bool:
        return key in self.contents

    def __setitem__(self, key: Register, value: Expression) -> None:
        assert key != Register("zero")
        self.contents[key] = value

    def __delitem__(self, key: Register) -> None:
        assert key != Register("zero")
        del self.contents[key]

    def get_raw(self, key: Register) -> Optional[Expression]:
        return self.contents.get(key, None)

    def __str__(self) -> str:
        return ", ".join(
            f"{k}: {v}"
            for k, v in sorted(self.contents.items())
            if not self.stack_info.should_save(v, None)
        )


@attr.s
class BlockInfo:
    """
    Contains translated assembly code (to_write), the block's branch condition,
    and block's final register states.
    """

    to_write: List[Statement] = attr.ib()
    return_value: Optional[Expression] = attr.ib()
    switch_value: Optional[Expression] = attr.ib()
    branch_condition: Optional[Condition] = attr.ib()
    final_register_states: RegInfo = attr.ib()
    has_custom_return: bool = attr.ib()
    has_function_call: bool = attr.ib()

    def __str__(self) -> str:
        newline = "\n\t"
        return "\n".join(
            [
                f"Statements: {newline.join(str(w) for w in self.to_write if w.should_write())}",
                f"Branch condition: {self.branch_condition}",
                f"Final register states: {self.final_register_states}",
            ]
        )


@attr.s
class InstrArgs:
    raw_args: List[Argument] = attr.ib()
    regs: RegInfo = attr.ib(repr=False)
    stack_info: StackInfo = attr.ib(repr=False)

    def raw_arg(self, index: int) -> Argument:
        assert index >= 0
        if index >= len(self.raw_args):
            raise DecompFailure(
                f"Too few arguments for instruction, expected at least {index + 1}"
            )
        return self.raw_args[index]

    def reg_ref(self, index: int) -> Register:
        ret = self.raw_arg(index)
        if not isinstance(ret, Register):
            raise DecompFailure(
                f"Expected instruction argument to be a register, but found {ret}"
            )
        return ret

    def reg(self, index: int) -> Expression:
        return self.regs[self.reg_ref(index)]

    def dreg(self, index: int) -> Expression:
        """Extract a double from a register. This may involve reading both the
        mentioned register and the next."""
        reg = self.reg_ref(index)
        if not reg.is_float():
            raise DecompFailure(
                f"Expected instruction argument {reg} to be a float register"
            )
        ret = self.regs[reg]
        if not isinstance(ret, Literal) or ret.type.get_size_bits() == 64:
            return ret
        reg_num = int(reg.register_name[1:])
        if reg_num % 2 != 0:
            raise DecompFailure(
                "Tried to use a double-precision instruction with odd-numbered float "
                f"register {reg}"
            )
        other = self.regs[Register(f"f{reg_num+1}")]
        if not isinstance(other, Literal) or other.type.get_size_bits() == 64:
            raise DecompFailure(
                f"Unable to determine a value for double-precision register {reg} "
                "whose second half is non-static. This is a mips_to_c restriction "
                "which may be lifted in the future."
            )
        value = ret.value | (other.value << 32)
        return Literal(value, type=Type.f64())

    def full_imm(self, index: int) -> Expression:
        arg = strip_macros(self.raw_arg(index))
        ret = literal_expr(arg, self.stack_info)
        if isinstance(ret, GlobalSymbol):
            return self.stack_info.global_info.address_of_gsym(ret.symbol_name)
        return ret

    def imm(self, index: int) -> Expression:
        ret = self.full_imm(index)
        if isinstance(ret, Literal):
            return Literal(((ret.value + 0x8000) & 0xFFFF) - 0x8000)
        return ret

    def unsigned_imm(self, index: int) -> Expression:
        ret = self.full_imm(index)
        if isinstance(ret, Literal):
            return Literal(ret.value & 0xFFFF)
        return ret

    def hi_imm(self, index: int) -> Argument:
        arg = self.raw_arg(index)
        if not isinstance(arg, Macro) or arg.macro_name != "hi":
            raise DecompFailure("Got lui instruction with macro other than %hi")
        return arg.argument

    def memory_ref(self, index: int) -> Union[AddressMode, RawSymbolRef]:
        ret = strip_macros(self.raw_arg(index))

        # Allow e.g. "lw $v0, symbol + 4", which isn't valid MIPS assembly, but is
        # outputted by some disassemblers (like IDA).
        if isinstance(ret, AsmGlobalSymbol):
            return RawSymbolRef(offset=0, sym=ret)
        if (
            isinstance(ret, BinOp)
            and ret.op in "+-"
            and isinstance(ret.lhs, AsmGlobalSymbol)
            and isinstance(ret.rhs, AsmLiteral)
        ):
            sign = 1 if ret.op == "+" else -1
            return RawSymbolRef(offset=(ret.rhs.value * sign), sym=ret.lhs)

        if not isinstance(ret, AsmAddressMode):
            raise DecompFailure(
                "Expected instruction argument to be of the form offset($register), "
                f"but found {ret}"
            )
        if ret.lhs is None:
            return AddressMode(offset=0, rhs=ret.rhs)
        if not isinstance(ret.lhs, AsmLiteral):
            raise DecompFailure(
                f"Unable to parse offset for instruction argument {ret}. "
                "Expected a constant or a %lo macro."
            )
        return AddressMode(offset=ret.lhs.signed_value(), rhs=ret.rhs)

    def count(self) -> int:
        return len(self.raw_args)


def deref(
    arg: Union[AddressMode, RawSymbolRef],
    regs: RegInfo,
    stack_info: StackInfo,
    *,
    size: int,
    store: bool = False,
) -> Expression:
    offset = arg.offset
    if isinstance(arg, AddressMode):
        if stack_info.is_stack_reg(arg.rhs):
            return stack_info.get_stack_var(offset, store=store)
        var = regs[arg.rhs]
    else:
        var = stack_info.global_info.address_of_gsym(arg.sym.symbol_name)

    # Struct member is being dereferenced.

    # Cope slightly better with raw pointers.
    if isinstance(var, Literal) and var.value % (2 ** 16) == 0:
        var = Literal(var.value + offset, type=var.type)
        offset = 0

    # Handle large struct offsets.
    uw_var = early_unwrap(var)
    if isinstance(uw_var, BinaryOp) and uw_var.op == "+":
        for base, addend in [(uw_var.left, uw_var.right), (uw_var.right, uw_var.left)]:
            if (
                isinstance(addend, Literal)
                and addend.value % 2 ** 15 in [0, 2 ** 15 - 1]
                and addend.value < 0x1000000
            ):
                offset += addend.value
                var = base
                break

    var.type.unify(Type.ptr())
    stack_info.record_struct_access(var, offset)
    field_name: Optional[str] = None
    type: Type = stack_info.unique_type_for("struct", (uw_var, offset), Type.any())

    # Struct access with type information.
    typemap = stack_info.global_info.typemap
    if typemap:
        array_expr = array_access_from_add(
            var, offset, stack_info, typemap, target_size=size, ptr=False
        )
        if array_expr is not None:
            return array_expr
        field_name, new_type, _, _ = get_field(
            var.type, offset, typemap, target_size=size
        )
        if field_name is not None:
            new_type.unify(type)
            type = new_type

    # Dereferencing pointers of known types
    target = var.type.get_pointer_target()
    if field_name is None and target is not None:
        sub_size, sub_align = target.get_size_align_bytes()
        if sub_size == size and offset % size == 0 and sub_align != 0:
            # TODO: This only turns the deref into an ArrayAccess if the type
            # is *known* to be an array (CType). This could be expanded to support
            # arrays of other types.
            if offset != 0 and target.is_ctype():
                index = Literal(value=offset // size, type=Type.s32())
                return ArrayAccess(var, index, type=target)
            type = target

    return StructAccess(
        struct_var=var,
        offset=offset,
        target_size=size,
        field_name=field_name,
        stack_info=stack_info,
        type=type,
    )


def is_trivial_expression(expr: Expression) -> bool:
    # Determine whether an expression should be evaluated only once or not.
    if isinstance(
        expr,
        (
            EvalOnceExpr,
            ForceVarExpr,
            Literal,
            GlobalSymbol,
            LocalVar,
            PassedInArg,
            PhiExpr,
            RegisterVar,
            SubroutineArg,
        ),
    ):
        return True
    if isinstance(expr, AddressOf):
        return is_trivial_expression(expr.expr)
    if isinstance(expr, Cast):
        return expr.is_trivial()
    return False


def is_type_obvious(expr: Expression) -> bool:
    """
    Determine whether an expression's type is "obvious", e.g. because the
    expression refers to a variable which has a declaration. With perfect type
    information this this function would not be needed.

    This function may produce wrong results while code is being generated,
    since at that point we don't know the final status of EvalOnceExpr's.
    """
    if isinstance(
        expr,
        (
            Cast,
            Literal,
            StringLiteral,
            AddressOf,
            LocalVar,
            PhiExpr,
            PassedInArg,
            RegisterVar,
            FuncCall,
        ),
    ):
        return True
    if isinstance(expr, ForceVarExpr):
        return is_type_obvious(expr.wrapped_expr)
    if isinstance(expr, EvalOnceExpr):
        if expr.need_decl():
            return True
        return is_type_obvious(expr.wrapped_expr)
    return False


def simplify_condition(expr: Expression) -> Expression:
    """
    Simplify a boolean expression.

    This function may produce wrong results while code is being generated,
    since at that point we don't know the final status of EvalOnceExpr's.
    """
    if isinstance(expr, EvalOnceExpr) and not expr.need_decl():
        return simplify_condition(expr.wrapped_expr)
    if isinstance(expr, BinaryOp):
        left = simplify_condition(expr.left)
        right = simplify_condition(expr.right)
        if isinstance(left, BinaryOp) and left.is_boolean() and right == Literal(0):
            if expr.op == "==":
                return simplify_condition(left.negated())
            if expr.op == "!=":
                return left
        return BinaryOp(left=left, op=expr.op, right=right, type=expr.type)
    return expr


def balanced_parentheses(string: str) -> bool:
    """
    Check if parentheses in a string are balanced, ignoring any non-parenthesis
    characters. E.g. true for "(x())yz", false for ")(" or "(".
    """
    bal = 0
    for c in string:
        if c == "(":
            bal += 1
        elif c == ")":
            if bal == 0:
                return False
            bal -= 1
    return bal == 0


def format_expr(expr: Expression, fmt: Formatter) -> str:
    """Stringify an expression, stripping unnecessary parentheses around it."""
    ret = expr.format(fmt)
    if ret.startswith("(") and balanced_parentheses(ret[1:-1]):
        return ret[1:-1]
    return ret


def format_assignment(dest: Expression, source: Expression, fmt: Formatter) -> str:
    """Stringify `dest = source;`."""
    dest = late_unwrap(dest)
    source = late_unwrap(source)
    if isinstance(source, BinaryOp) and source.op in COMPOUND_ASSIGNMENT_OPS:
        rhs = None
        if late_unwrap(source.left) == dest:
            rhs = source.right
        elif late_unwrap(source.right) == dest and source.op in ASSOCIATIVE_OPS:
            rhs = source.left
        if rhs is not None:
            return f"{dest.format(fmt)} {source.op}= {format_expr(rhs, fmt)};"
    return f"{dest.format(fmt)} = {format_expr(source, fmt)};"


def parenthesize_for_struct_access(expr: Expression, fmt: Formatter) -> str:
    # Nested dereferences may need to be parenthesized. All other
    # expressions will already have adequate parentheses added to them.
    # (Except Cast's, TODO...)
    s = expr.format(fmt)
    if s.startswith("*") or s.startswith("&"):
        return f"({s})"
    return s


def elide_casts_for_store(expr: Expression) -> Expression:
    uw_expr = late_unwrap(expr)
    if isinstance(uw_expr, Cast) and not uw_expr.needed_for_store():
        return elide_casts_for_store(uw_expr.expr)
    if isinstance(uw_expr, Literal) and uw_expr.type.is_int():
        return Literal(uw_expr.value, type=Type.intish())
    return uw_expr


def uses_expr(expr: Expression, expr_filter: Callable[[Expression], bool]) -> bool:
    if expr_filter(expr):
        return True
    for e in expr.dependencies():
        if uses_expr(e, expr_filter):
            return True
    return False


def late_unwrap(expr: Expression) -> Expression:
    """
    Unwrap EvalOnceExpr's and ForceVarExpr's, stopping at variable boundaries.

    This function may produce wrong results while code is being generated,
    since at that point we don't know the final status of EvalOnceExpr's.
    """
    if isinstance(expr, ForceVarExpr):
        return late_unwrap(expr.wrapped_expr)
    if isinstance(expr, EvalOnceExpr) and not expr.need_decl():
        return late_unwrap(expr.wrapped_expr)
    if isinstance(expr, PhiExpr) and expr.replacement_expr is not None:
        return late_unwrap(expr.replacement_expr)
    return expr


def early_unwrap(expr: Expression) -> Expression:
    """
    Unwrap EvalOnceExpr's, even past variable boundaries.

    This is fine to use even while code is being generated, but disrespects decisions
    to use a temp for a value, so use with care.

    TODO: unwrap ForceVarExpr as well when safe, pushing the forces down into the
    expression tree.
    """
    if (
        isinstance(expr, EvalOnceExpr)
        and not expr.forced_emit
        and not expr.emit_exactly_once
    ):
        return early_unwrap(expr.wrapped_expr)
    return expr


def unwrap_deep(expr: Expression) -> Expression:
    """
    Unwrap EvalOnceExpr's and ForceVarExpr's, even past variable boundaries.

    This is generally a sketchy thing to do, try to avoid it. In particular:
    - the returned expression is not usable for emission, because it may contain
      accesses at an earlier point in time or an expression that should not be repeated.
    - just because unwrap_deep(a) == unwrap_deep(b) doesn't mean a and b are
      interchangable, because they may be computed in different places.
    """
    if isinstance(expr, (EvalOnceExpr, ForceVarExpr)):
        return unwrap_deep(expr.wrapped_expr)
    return expr


def literal_expr(arg: Argument, stack_info: StackInfo) -> Expression:
    if isinstance(arg, AsmGlobalSymbol):
        return stack_info.global_info.global_symbol(arg.symbol_name)
    if isinstance(arg, AsmLiteral):
        return Literal(arg.value)
    if isinstance(arg, BinOp):
        lhs = literal_expr(arg.lhs, stack_info)
        rhs = literal_expr(arg.rhs, stack_info)
        return BinaryOp.int(left=lhs, op=arg.op, right=rhs)
    raise DecompFailure(f"Instruction argument {arg} must be a literal")


def imm_add_32(expr: Expression) -> Expression:
    if isinstance(expr, Literal):
        return as_intish(Literal(expr.value + 32))
    else:
        return BinaryOp.int(expr, "+", Literal(32))


def fn_op(fn_name: str, args: List[Expression], type: Type) -> FuncCall:
    fn_sig = FunctionSignature(
        return_type=type,
        params=[FunctionParam(type=arg.type) for arg in args],
        params_known=True,
        is_variadic=False,
    )
    return FuncCall(
        function=GlobalSymbol(symbol_name=fn_name, type=Type.function(fn_sig)),
        args=args,
        type=type,
    )


def void_fn_op(fn_name: str, args: List[Expression]) -> FuncCall:
    return fn_op(fn_name, args, Type.any_reg())


def load_upper(args: InstrArgs) -> Expression:
    arg = args.raw_arg(1)
    if not isinstance(arg, Macro):
        assert not isinstance(
            arg, Literal
        ), "normalize_instruction should convert lui <literal> to li"
        raise DecompFailure(f"lui argument must be a literal or %hi macro, found {arg}")

    hi_arg = args.hi_imm(1)
    if (
        isinstance(hi_arg, BinOp)
        and hi_arg.op in "+-"
        and isinstance(hi_arg.lhs, AsmGlobalSymbol)
        and isinstance(hi_arg.rhs, AsmLiteral)
    ):
        sym = hi_arg.lhs
        offset = hi_arg.rhs.value * (-1 if hi_arg.op == "-" else 1)
    elif isinstance(hi_arg, AsmGlobalSymbol):
        sym = hi_arg
        offset = 0
    else:
        raise DecompFailure(f"Invalid %hi argument {hi_arg}")

    stack_info = args.stack_info
    source = stack_info.global_info.address_of_gsym(sym.symbol_name)
    imm = Literal(offset)
    return handle_addi_real(args.reg_ref(0), None, source, imm, stack_info)


def handle_convert(expr: Expression, dest_type: Type, source_type: Type) -> Cast:
    # int <-> float casts should be explicit
    silent = dest_type.data().kind != source_type.data().kind
    expr.type.unify(source_type)
    return Cast(expr=expr, type=dest_type, silent=silent, reinterpret=False)


def handle_la(args: InstrArgs) -> Expression:
    target = args.memory_ref(1)
    stack_info = args.stack_info
    if isinstance(target, AddressMode):
        return handle_addi(
            InstrArgs(
                raw_args=[args.reg_ref(0), target.rhs, AsmLiteral(target.offset)],
                regs=args.regs,
                stack_info=args.stack_info,
            )
        )
    var = stack_info.global_info.address_of_gsym(target.sym.symbol_name)
    return add_imm(var, Literal(target.offset), stack_info)


def handle_ori(args: InstrArgs) -> Expression:
    imm = args.unsigned_imm(2)
    r = args.reg(1)
    if isinstance(r, Literal) and isinstance(imm, Literal):
        return Literal(value=(r.value | imm.value))
    # Regular bitwise OR.
    return BinaryOp.int(left=r, op="|", right=imm)


def handle_sltu(args: InstrArgs) -> Expression:
    right = args.reg(2)
    if args.reg_ref(1) == Register("zero"):
        # (0U < x) is equivalent to (x != 0)
        uw_right = early_unwrap(right)
        if isinstance(uw_right, BinaryOp) and uw_right.op == "^":
            # ((a ^ b) != 0) is equivalent to (a != b)
            return BinaryOp.icmp(uw_right.left, "!=", uw_right.right)
        return BinaryOp.icmp(right, "!=", Literal(0))
    else:
        left = args.reg(1)
        return BinaryOp.ucmp(left, "<", right)


def handle_sltiu(args: InstrArgs) -> Expression:
    left = args.reg(1)
    right = args.imm(2)
    if isinstance(right, Literal):
        value = right.value & 0xFFFFFFFF
        if value == 1:
            # (x < 1U) is equivalent to (x == 0)
            uw_left = early_unwrap(left)
            if isinstance(uw_left, BinaryOp) and uw_left.op == "^":
                # ((a ^ b) == 0) is equivalent to (a == b)
                return BinaryOp.icmp(uw_left.left, "==", uw_left.right)
            return BinaryOp.icmp(left, "==", Literal(0))
        else:
            right = Literal(value)
    return BinaryOp.ucmp(left, "<", right)


def handle_addi(args: InstrArgs) -> Expression:
    stack_info = args.stack_info
    source_reg = args.reg_ref(1)
    source = args.reg(1)
    imm = args.imm(2)
    return handle_addi_real(args.reg_ref(0), source_reg, source, imm, stack_info)


def handle_addi_real(
    output_reg: Register,
    source_reg: Optional[Register],
    source: Expression,
    imm: Expression,
    stack_info: StackInfo,
) -> Expression:
    if source_reg is not None and stack_info.is_stack_reg(source_reg):
        # Adding to sp, i.e. passing an address.
        assert isinstance(imm, Literal)
        if stack_info.is_stack_reg(output_reg):
            # Changing sp. Just ignore that.
            return source
        # Keep track of all local variables that we take addresses of.
        var = stack_info.get_stack_var(imm.value, store=False)
        if isinstance(var, LocalVar):
            stack_info.add_local_var(var)
        return AddressOf(var, type=Type.ptr(var.type))
    else:
        return add_imm(source, imm, stack_info)


def add_imm(source: Expression, imm: Expression, stack_info: StackInfo) -> Expression:
    if imm == Literal(0):
        # addiu $reg1, $reg2, 0 is a move
        # (this happens when replacing %lo(...) by 0)
        return source
    elif source.type.is_pointer():
        # Pointer addition (this may miss some pointers that get detected later;
        # unfortunately that's hard to do anything about with mips_to_c's single-pass
        # architecture.
        typemap = stack_info.global_info.typemap
        if typemap and isinstance(imm, Literal):
            array_access = array_access_from_add(
                source, imm.value, stack_info, typemap, target_size=None, ptr=True
            )
            if array_access is not None:
                return array_access

            field_name, subtype, ptr_type, is_array = get_field(
                source.type, imm.value, typemap, target_size=None
            )
            if field_name is not None:
                if is_array:
                    return StructAccess(
                        struct_var=source,
                        offset=imm.value,
                        target_size=None,
                        field_name=field_name,
                        stack_info=stack_info,
                        type=ptr_type,
                    )
                else:
                    return AddressOf(
                        StructAccess(
                            struct_var=source,
                            offset=imm.value,
                            target_size=None,
                            field_name=field_name,
                            stack_info=stack_info,
                            type=subtype,
                        ),
                        type=ptr_type,
                    )
        if isinstance(imm, Literal):
            target = source.type.get_pointer_target()
            if target:
                target_size, _ = target.get_size_align_bytes()
                if target_size and imm.value % target_size == 0:
                    # Pointer addition.
                    return BinaryOp(
                        left=source, op="+", right=as_intish(imm), type=source.type
                    )
        return BinaryOp(left=source, op="+", right=as_intish(imm), type=Type.ptr())
    elif isinstance(source, Literal) and isinstance(imm, Literal):
        return Literal(source.value + imm.value)
    else:
        # Regular binary addition.
        return BinaryOp.intptr(left=source, op="+", right=imm)


def handle_load(args: InstrArgs, type: Type) -> Expression:
    # For now, make the cast silent so that output doesn't become cluttered.
    # Though really, it would be great to expose the load types somehow...
    # If the type size is unknown, default to 32 bits
    size = (type.get_size_bits() or 32) // 8
    expr = deref(args.memory_ref(1), args.regs, args.stack_info, size=size)

    # Detect rodata constants
    if isinstance(expr, StructAccess) and expr.offset == 0:
        target = early_unwrap(expr.struct_var)
        if (
            isinstance(target, AddressOf)
            and isinstance(target.expr, GlobalSymbol)
            and type.is_float()
        ):
            sym_name = target.expr.symbol_name
            ent = args.stack_info.global_info.asm_data_value(sym_name)
            if (
                ent
                and ent.data
                and isinstance(ent.data[0], bytes)
                and len(ent.data[0]) >= size
                and ent.is_readonly
                and type.unify(target.expr.type)
            ):
                data = ent.data[0][:size]
                val: int
                if size == 4:
                    (val,) = struct.unpack(">I", data)
                else:
                    (val,) = struct.unpack(">Q", data)
                return Literal(value=val, type=type)

    return as_type(expr, type, silent=True)


def deref_unaligned(
    arg: Union[AddressMode, RawSymbolRef],
    regs: RegInfo,
    stack_info: StackInfo,
    *,
    store: bool = False,
) -> Expression:
    # We don't know the correct size pass to deref. Passing None would signal that we
    # are taking an address, cause us to prefer entire substructs as referenced fields,
    # which would be confusing. Instead, we lie and pass 1. Hopefully nothing bad will
    # happen...
    return deref(arg, regs, stack_info, size=1, store=store)


def handle_lwl(args: InstrArgs) -> Expression:
    ref = args.memory_ref(1)
    expr = deref_unaligned(ref, args.regs, args.stack_info)
    key: Tuple[int, object]
    if isinstance(ref, AddressMode):
        key = (ref.offset, args.regs[ref.rhs])
    else:
        key = (ref.offset, ref.sym)
    return Lwl(expr, key)


def handle_lwr(args: InstrArgs, old_value: Expression) -> Expression:
    # This lwr may merge with an existing lwl, if it loads from the same target
    # but with an offset that's +3.
    uw_old_value = early_unwrap(old_value)
    ref = args.memory_ref(1)
    lwl_key: Tuple[int, object]
    if isinstance(ref, AddressMode):
        lwl_key = (ref.offset - 3, args.regs[ref.rhs])
    else:
        lwl_key = (ref.offset - 3, ref.sym)
    if isinstance(uw_old_value, Lwl) and uw_old_value.key[0] == lwl_key[0]:
        return UnalignedLoad(uw_old_value.load_expr)
    if ref.offset % 4 == 2:
        left_mem_ref = attr.evolve(ref, offset=ref.offset - 2)
        load_expr = deref_unaligned(left_mem_ref, args.regs, args.stack_info)
        return Load3Bytes(load_expr)
    return ErrorExpr("Unable to handle lwr; missing a corresponding lwl")


def make_store(args: InstrArgs, type: Type) -> Optional[StoreStmt]:
    # If the type size is unknown, default to 32 bits
    size = (type.get_size_bits() or 32) // 8
    stack_info = args.stack_info
    source_reg = args.reg_ref(0)
    source_raw = args.regs.get_raw(source_reg)
    if type.is_float() and type.get_size_bits() == 64:
        source_val = args.dreg(0)
    else:
        source_val = args.reg(0)
    target = args.memory_ref(1)
    is_stack = isinstance(target, AddressMode) and stack_info.is_stack_reg(target.rhs)
    if (
        is_stack
        and source_raw is not None
        and stack_info.should_save(source_raw, target.offset)
    ):
        # Elide register preserval.
        return None
    dest = deref(target, args.regs, stack_info, size=size, store=True)
    dest.type.unify(type)
    return StoreStmt(source=as_type(source_val, type, silent=is_stack), dest=dest)


def handle_swl(args: InstrArgs) -> Optional[StoreStmt]:
    # swl in practice only occurs together with swr, so we can treat it as a regular
    # store, with the expression wrapped in UnalignedLoad if needed.
    source = args.reg(0)
    target = args.memory_ref(1)
    if not isinstance(early_unwrap(source), UnalignedLoad):
        source = UnalignedLoad(source)
    dest = deref_unaligned(target, args.regs, args.stack_info, store=True)
    return StoreStmt(source=source, dest=dest)


def handle_swr(args: InstrArgs) -> Optional[StoreStmt]:
    expr = early_unwrap(args.reg(0))
    target = args.memory_ref(1)
    if not isinstance(expr, Load3Bytes):
        # Elide swr's that don't come from 3-byte-loading lwr's; they probably
        # come with a corresponding swl which has already been emitted.
        return None
    real_target = attr.evolve(target, offset=target.offset - 2)
    dest = deref_unaligned(real_target, args.regs, args.stack_info, store=True)
    return StoreStmt(source=expr, dest=dest)


def handle_sra(args: InstrArgs) -> Expression:
    lhs = args.reg(1)
    shift = args.imm(2)
    if isinstance(shift, Literal) and shift.value in [16, 24]:
        expr = early_unwrap(lhs)
        pow2 = 1 << shift.value
        if isinstance(expr, BinaryOp) and isinstance(expr.right, Literal):
            tp = Type.s16() if shift.value == 16 else Type.s8()
            rhs = expr.right.value
            if expr.op == "<<" and rhs == shift.value:
                return as_type(expr.left, tp, silent=False)
            elif expr.op == "*" and rhs % pow2 == 0 and rhs != pow2:
                mul = BinaryOp.int(expr.left, "*", Literal(value=rhs // pow2))
                return as_type(mul, tp, silent=False)
    return BinaryOp(as_s32(lhs), ">>", as_intish(shift), type=Type.s32())


def format_f32_imm(num: int) -> str:
    packed = struct.pack(">I", num & (2 ** 32 - 1))
    value = struct.unpack(">f", packed)[0]

    if not value or value == 4294967296.0:
        # Zero, negative zero, nan, or INT_MAX.
        return str(value)

    # Write values smaller than 1e-7 / greater than 1e7 using scientific notation,
    # and values in between using fixed point.
    if abs(math.log10(abs(value))) > 6.9:
        fmt_char = "e"
    elif abs(value) < 1:
        fmt_char = "f"
    else:
        fmt_char = "g"

    def fmt(prec: int) -> str:
        """Format 'value' with 'prec' significant digits/decimals, in either scientific
        or regular notation depending on 'fmt_char'."""
        ret = ("{:." + str(prec) + fmt_char + "}").format(value)
        if fmt_char == "e":
            return ret.replace("e+", "e").replace("e0", "e").replace("e-0", "e-")
        if "e" in ret:
            # The "g" format character can sometimes introduce scientific notation if
            # formatting with too few decimals. If this happens, return an incorrect
            # value to prevent the result from being used.
            #
            # Since the value we are formatting is within (1e-7, 1e7) in absolute
            # value, it will at least be possible to format with 7 decimals, which is
            # less than float precision. Thus, this annoying Python limitation won't
            # lead to us outputting numbers with more precision than we really have.
            return "0"
        return ret

    # 20 decimals is more than enough for a float. Start there, then try to shrink it.
    prec = 20
    while prec > 0:
        prec -= 1
        value2 = float(fmt(prec))
        if struct.pack(">f", value2) != packed:
            prec += 1
            break

    if prec == 20:
        # Uh oh, even the original value didn't format correctly. Fall back to str(),
        # which ought to work.
        return str(value)

    ret = fmt(prec)
    if "." not in ret:
        ret += ".0"
    return ret


def format_f64_imm(num: int) -> str:
    (value,) = struct.unpack(">d", struct.pack(">Q", num & (2 ** 64 - 1)))
    return str(value)


def fold_mul_chains(expr: Expression) -> Expression:
    """Simplify an expression involving +, -, * and << to a single multiplication,
    e.g. 4*x - x -> 3*x, or x<<2 -> x*4. This includes some logic for preventing
    folds of consecutive sll, and keeping multiplications by large powers of two
    as bitshifts at the top layer."""

    def fold(
        expr: Expression, toplevel: bool, allow_sll: bool
    ) -> Tuple[Expression, int]:
        if isinstance(expr, BinaryOp):
            lbase, lnum = fold(expr.left, False, (expr.op != "<<"))
            rbase, rnum = fold(expr.right, False, (expr.op != "<<"))
            if expr.op == "<<" and isinstance(expr.right, Literal) and allow_sll:
                # Left-shifts by small numbers are easier to understand if
                # written as multiplications (they compile to the same thing).
                if toplevel and lnum == 1 and not (1 <= expr.right.value <= 4):
                    return (expr, 1)
                return (lbase, lnum << expr.right.value)
            if (
                expr.op == "*"
                and isinstance(expr.right, Literal)
                and (allow_sll or expr.right.value % 2 != 0)
            ):
                return (lbase, lnum * expr.right.value)
            if early_unwrap(lbase) == early_unwrap(rbase):
                if expr.op == "+":
                    return (lbase, lnum + rnum)
                if expr.op == "-":
                    return (lbase, lnum - rnum)
        if isinstance(expr, UnaryOp) and not toplevel:
            base, num = fold(expr.expr, False, True)
            return (base, -num)
        if (
            isinstance(expr, EvalOnceExpr)
            and not expr.emit_exactly_once
            and not expr.forced_emit
        ):
            base, num = fold(early_unwrap(expr), False, allow_sll)
            if num != 1 and is_trivial_expression(base):
                return (base, num)
        return (expr, 1)

    base, num = fold(expr, True, True)
    if num == 1:
        return expr
    return BinaryOp.int(left=base, op="*", right=Literal(num))


def array_access_from_add(
    expr: Expression,
    offset: int,
    stack_info: StackInfo,
    typemap: TypeMap,
    *,
    target_size: Optional[int],
    ptr: bool,
) -> Optional[Expression]:
    expr = early_unwrap(expr)
    if not isinstance(expr, BinaryOp) or expr.op != "+":
        return None
    base = expr.left
    addend = expr.right
    if addend.type.is_pointer() and not base.type.is_pointer():
        base, addend = addend, base

    index: Expression
    scale: int
    uw_addend = early_unwrap(addend)
    if (
        isinstance(uw_addend, BinaryOp)
        and uw_addend.op == "*"
        and isinstance(uw_addend.right, Literal)
    ):
        index = uw_addend.left
        scale = uw_addend.right.value
    elif (
        isinstance(uw_addend, BinaryOp)
        and uw_addend.op == "<<"
        and isinstance(uw_addend.right, Literal)
    ):
        index = uw_addend.left
        scale = 1 << uw_addend.right.value
    else:
        index = addend
        scale = 1

    target_type = base.type.get_pointer_target()
    if target_type is None:
        return None

    if target_type.get_size_align_bytes()[0] == scale:
        # base[index]
        pass
    else:
        # base->subarray[index]
        substr_array = find_substruct_array(base.type, offset, scale, typemap)
        if substr_array is None:
            return None
        sub_field_name, sub_offset, elem_type = substr_array
        base = StructAccess(
            struct_var=base,
            offset=sub_offset,
            target_size=None,
            field_name=sub_field_name,
            stack_info=stack_info,
            type=Type.ptr(elem_type),
        )
        offset -= sub_offset
        target_type = elem_type

    # Add .field if necessary
    ret: Expression = ArrayAccess(base, index, type=target_type)
    field_name, new_type, ptr_type, is_array = get_field(
        base.type, offset, typemap, target_size=target_size
    )
    if offset != 0 or (target_size is not None and target_size != scale):
        ret = StructAccess(
            struct_var=AddressOf(ret, type=Type.ptr()),
            offset=offset,
            target_size=target_size,
            field_name=field_name,
            stack_info=stack_info,
            type=ptr_type if is_array else new_type,
        )

    if ptr and not is_array:
        ret = AddressOf(ret, type=ptr_type)
    return ret


def handle_add(args: InstrArgs) -> Expression:
    lhs = args.reg(1)
    rhs = args.reg(2)
    stack_info = args.stack_info
    type = Type.intptr()
    if lhs.type.is_pointer():
        type = Type.ptr()
    elif rhs.type.is_pointer():
        type = Type.ptr()

    # addiu instructions can sometimes be emitted as addu instead, when the
    # offset is too large.
    if isinstance(rhs, Literal):
        return handle_addi_real(args.reg_ref(0), args.reg_ref(1), lhs, rhs, stack_info)
    if isinstance(lhs, Literal):
        return handle_addi_real(args.reg_ref(0), args.reg_ref(2), rhs, lhs, stack_info)

    expr = BinaryOp(left=as_intptr(lhs), op="+", right=as_intptr(rhs), type=type)
    folded_expr = fold_mul_chains(expr)
    if folded_expr is not expr:
        return folded_expr
    typemap = stack_info.global_info.typemap
    if typemap is not None:
        array_expr = array_access_from_add(
            expr, 0, stack_info, typemap, target_size=None, ptr=True
        )
        if array_expr is not None:
            return array_expr
    return expr


def handle_add_float(args: InstrArgs) -> Expression:
    if args.reg_ref(1) == args.reg_ref(2):
        two = Literal(1 << 30, type=Type.f32())
        return BinaryOp.f32(two, "*", args.reg(1))
    return BinaryOp.f32(args.reg(1), "+", args.reg(2))


def handle_add_double(args: InstrArgs) -> Expression:
    if args.reg_ref(1) == args.reg_ref(2):
        two = Literal(1 << 62, type=Type.f64())
        return BinaryOp.f64(two, "*", args.dreg(1))
    return BinaryOp.f64(args.dreg(1), "+", args.dreg(2))


def handle_bgez(args: InstrArgs) -> Condition:
    expr = args.reg(0)
    uw_expr = early_unwrap(expr)
    if (
        isinstance(uw_expr, BinaryOp)
        and uw_expr.op == "<<"
        and isinstance(uw_expr.right, Literal)
    ):
        shift = uw_expr.right.value
        bitand = BinaryOp.int(uw_expr.left, "&", Literal(1 << (31 - shift)))
        return UnaryOp("!", bitand, type=Type.bool())
    return BinaryOp.scmp(expr, ">=", Literal(0))


def strip_macros(arg: Argument) -> Argument:
    """Replace %lo(...) by 0, and assert that there are no %hi(...). We assume that
    %hi's only ever occur in lui, where we expand them to an entire value, and not
    just the upper part. This preserves semantics in most cases (though not when %hi's
    are reused for different %lo's...)"""
    if isinstance(arg, Macro):
        if arg.macro_name == "hi":
            raise DecompFailure("%hi macro outside of lui")
        if arg.macro_name != "lo":
            raise DecompFailure(f"Unrecognized linker macro %{arg.macro_name}")
        return AsmLiteral(0)
    elif isinstance(arg, AsmAddressMode) and isinstance(arg.lhs, Macro):
        if arg.lhs.macro_name != "lo":
            raise DecompFailure(
                f"Bad linker macro in instruction argument {arg}, expected %lo"
            )
        return AsmAddressMode(lhs=None, rhs=arg.rhs)
    else:
        return arg


@attr.s
class AbiStackSlot:
    offset: int = attr.ib()
    reg: Optional[Register] = attr.ib()
    name: Optional[str] = attr.ib()
    type: Type = attr.ib()


def function_abi(
    fn_sig: FunctionSignature, *, for_call: bool
) -> Tuple[List[AbiStackSlot], List[Register]]:
    """Compute stack positions/registers used by a function according to the o32 ABI,
    based on C type information. Additionally computes a list of registers that might
    contain arguments, if the function is a varargs function. (Additional varargs
    arguments may be passed on the stack; we could compute the offset at which that
    would start but right now don't care -- we just slurp up everything.)"""
    if not fn_sig.params_known:
        return [], [Register(r) for r in ["f12", "f13", "f14", "a0", "a1", "a2", "a3"]]

    offset = 0
    only_floats = True
    slots: List[AbiStackSlot] = []
    possible: List[Register] = []
    if fn_sig.return_type.is_struct_type():
        # The ABI for struct returns is to pass a pointer to where it should be written
        # as the first argument.
        slots.append(
            AbiStackSlot(
                offset=0,
                reg=Register("a0"),
                name="__return__",
                type=Type.ptr(fn_sig.return_type),
            )
        )
        offset = 4
        only_floats = False

    for ind, param in enumerate(fn_sig.params):
        size, align = param.type.get_size_align_bytes()
        size = (size + 3) & ~3
        only_floats = only_floats and param.type.is_float()
        offset = (offset + align - 1) & -align
        name = param.name
        reg2: Optional[Register]
        if ind < 2 and only_floats:
            reg = Register("f12" if ind == 0 else "f14")
            is_double = param.type.is_float() and param.type.get_size_bits() == 64
            slots.append(
                AbiStackSlot(offset=offset, reg=reg, name=name, type=param.type)
            )
            if is_double and not for_call:
                name2 = f"{name}_lo" if name else None
                reg2 = Register("f13" if ind == 0 else "f15")
                slots.append(
                    AbiStackSlot(
                        offset=offset + 4, reg=reg2, name=name2, type=Type.any_reg()
                    )
                )
        else:
            for i in range(offset // 4, (offset + size) // 4):
                unk_offset = 4 * i - offset
                name2 = f"{name}_unk{unk_offset:X}" if name and unk_offset else name
                reg2 = Register(f"a{i}") if i < 4 else None
                slots.append(
                    AbiStackSlot(offset=4 * i, reg=reg2, name=name2, type=param.type)
                )
        offset += size

    if fn_sig.is_variadic:
        for i in range(offset // 4, 4):
            possible.append(Register(f"a{i}"))

    return slots, possible


InstrSet = Set[str]
InstrMap = Dict[str, Callable[[InstrArgs], Expression]]
LwrInstrMap = Dict[str, Callable[[InstrArgs, Expression], Expression]]
CmpInstrMap = Dict[str, Callable[[InstrArgs], Condition]]
StoreInstrMap = Dict[str, Callable[[InstrArgs], Optional[StoreStmt]]]
MaybeInstrMap = Dict[str, Callable[[InstrArgs], Optional[Expression]]]
PairInstrMap = Dict[str, Callable[[InstrArgs], Tuple[Expression, Expression]]]

CASES_IGNORE: InstrSet = {
    # Ignore FCSR sets; they are leftovers from float->unsigned conversions.
    # FCSR gets are as well, but it's fine to read MIPS2C_ERROR for those.
    "ctc1",
    "nop",
    "b",
    "j",
}
CASES_STORE: StoreInstrMap = {
    # Storage instructions
    "sb": lambda a: make_store(a, type=Type.of_size(8)),
    "sh": lambda a: make_store(a, type=Type.of_size(16)),
    "sw": lambda a: make_store(a, type=Type.of_size(32)),
    "sd": lambda a: make_store(a, type=Type.of_size(64)),
    # Unaligned stores
    "swl": lambda a: handle_swl(a),
    "swr": lambda a: handle_swr(a),
    # Floating point storage/conversion
    "swc1": lambda a: make_store(a, type=Type.f32()),
    "sdc1": lambda a: make_store(a, type=Type.f64()),
}
CASES_BRANCHES: CmpInstrMap = {
    # Branch instructions/pseudoinstructions
    "beq": lambda a: BinaryOp.icmp(a.reg(0), "==", a.reg(1)),
    "bne": lambda a: BinaryOp.icmp(a.reg(0), "!=", a.reg(1)),
    "beqz": lambda a: BinaryOp.icmp(a.reg(0), "==", Literal(0)),
    "bnez": lambda a: BinaryOp.icmp(a.reg(0), "!=", Literal(0)),
    "blez": lambda a: BinaryOp.scmp(a.reg(0), "<=", Literal(0)),
    "bgtz": lambda a: BinaryOp.scmp(a.reg(0), ">", Literal(0)),
    "bltz": lambda a: BinaryOp.scmp(a.reg(0), "<", Literal(0)),
    "bgez": lambda a: handle_bgez(a),
}
CASES_FLOAT_BRANCHES: InstrSet = {
    # Floating-point branch instructions
    "bc1t",
    "bc1f",
}
CASES_JUMPS: InstrSet = {
    # Unconditional jump
    "jr"
}
CASES_FN_CALL: InstrSet = {
    # Function call
    "jal",
    "jalr",
}
CASES_NO_DEST: InstrMap = {
    # Conditional traps (happen with Pascal code sometimes, might as well give a nicer
    # output than MIPS2C_ERROR(...))
    "teq": lambda a: void_fn_op(
        "MIPS2C_TRAP_IF", [BinaryOp.icmp(a.reg(0), "==", a.reg(1))]
    ),
    "tne": lambda a: void_fn_op(
        "MIPS2C_TRAP_IF", [BinaryOp.icmp(a.reg(0), "!=", a.reg(1))]
    ),
    "tlt": lambda a: void_fn_op(
        "MIPS2C_TRAP_IF", [BinaryOp.scmp(a.reg(0), "<", a.reg(1))]
    ),
    "tltu": lambda a: void_fn_op(
        "MIPS2C_TRAP_IF", [BinaryOp.ucmp(a.reg(0), "<", a.reg(1))]
    ),
    "tge": lambda a: void_fn_op(
        "MIPS2C_TRAP_IF", [BinaryOp.scmp(a.reg(0), ">=", a.reg(1))]
    ),
    "tgeu": lambda a: void_fn_op(
        "MIPS2C_TRAP_IF", [BinaryOp.ucmp(a.reg(0), ">=", a.reg(1))]
    ),
    "teqi": lambda a: void_fn_op(
        "MIPS2C_TRAP_IF", [BinaryOp.icmp(a.reg(0), "==", a.imm(1))]
    ),
    "tnei": lambda a: void_fn_op(
        "MIPS2C_TRAP_IF", [BinaryOp.icmp(a.reg(0), "!=", a.imm(1))]
    ),
    "tlti": lambda a: void_fn_op(
        "MIPS2C_TRAP_IF", [BinaryOp.scmp(a.reg(0), "<", a.imm(1))]
    ),
    "tltiu": lambda a: void_fn_op(
        "MIPS2C_TRAP_IF", [BinaryOp.ucmp(a.reg(0), "<", a.imm(1))]
    ),
    "tgei": lambda a: void_fn_op(
        "MIPS2C_TRAP_IF", [BinaryOp.scmp(a.reg(0), ">=", a.imm(1))]
    ),
    "tgeiu": lambda a: void_fn_op(
        "MIPS2C_TRAP_IF", [BinaryOp.ucmp(a.reg(0), ">=", a.imm(1))]
    ),
    "break": lambda a: void_fn_op("MIPS2C_BREAK", [a.imm(0)] if a.count() >= 1 else []),
    "sync": lambda a: void_fn_op("MIPS2C_SYNC", []),
}
CASES_FLOAT_COMP: CmpInstrMap = {
    # Float comparisons that don't raise exception on nan
    "c.eq.s": lambda a: BinaryOp.fcmp(a.reg(0), "==", a.reg(1)),
    "c.olt.s": lambda a: BinaryOp.fcmp(a.reg(0), "<", a.reg(1)),
    "c.oge.s": lambda a: BinaryOp.fcmp(a.reg(0), ">=", a.reg(1)),
    "c.ole.s": lambda a: BinaryOp.fcmp(a.reg(0), "<=", a.reg(1)),
    "c.ogt.s": lambda a: BinaryOp.fcmp(a.reg(0), ">", a.reg(1)),
    "c.neq.s": lambda a: BinaryOp.fcmp(a.reg(0), "==", a.reg(1)).negated(),
    "c.uge.s": lambda a: BinaryOp.fcmp(a.reg(0), "<", a.reg(1)).negated(),
    "c.ult.s": lambda a: BinaryOp.fcmp(a.reg(0), ">=", a.reg(1)).negated(),
    "c.ugt.s": lambda a: BinaryOp.fcmp(a.reg(0), "<=", a.reg(1)).negated(),
    "c.ule.s": lambda a: BinaryOp.fcmp(a.reg(0), ">", a.reg(1)).negated(),
    # Float comparisons that may raise exception on nan
    "c.seq.s": lambda a: BinaryOp.fcmp(a.reg(0), "==", a.reg(1)),
    "c.lt.s": lambda a: BinaryOp.fcmp(a.reg(0), "<", a.reg(1)),
    "c.ge.s": lambda a: BinaryOp.fcmp(a.reg(0), ">=", a.reg(1)),
    "c.le.s": lambda a: BinaryOp.fcmp(a.reg(0), "<=", a.reg(1)),
    "c.gt.s": lambda a: BinaryOp.fcmp(a.reg(0), ">", a.reg(1)),
    "c.sne.s": lambda a: BinaryOp.fcmp(a.reg(0), "==", a.reg(1)).negated(),
    "c.nle.s": lambda a: BinaryOp.fcmp(a.reg(0), "<=", a.reg(1)).negated(),
    "c.nlt.s": lambda a: BinaryOp.fcmp(a.reg(0), "<", a.reg(1)).negated(),
    "c.nge.s": lambda a: BinaryOp.fcmp(a.reg(0), ">=", a.reg(1)).negated(),
    "c.ngt.s": lambda a: BinaryOp.fcmp(a.reg(0), ">", a.reg(1)).negated(),
    # Double comparisons that don't raise exception on nan
    "c.eq.d": lambda a: BinaryOp.dcmp(a.dreg(0), "==", a.dreg(1)),
    "c.olt.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<", a.dreg(1)),
    "c.oge.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">=", a.dreg(1)),
    "c.ole.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<=", a.dreg(1)),
    "c.ogt.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">", a.dreg(1)),
    "c.neq.d": lambda a: BinaryOp.dcmp(a.dreg(0), "==", a.dreg(1)).negated(),
    "c.uge.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<", a.dreg(1)).negated(),
    "c.ult.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">=", a.dreg(1)).negated(),
    "c.ugt.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<=", a.dreg(1)).negated(),
    "c.ule.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">", a.dreg(1)).negated(),
    # Double comparisons that may raise exception on nan
    "c.seq.d": lambda a: BinaryOp.dcmp(a.dreg(0), "==", a.dreg(1)),
    "c.lt.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<", a.dreg(1)),
    "c.ge.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">=", a.dreg(1)),
    "c.le.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<=", a.dreg(1)),
    "c.gt.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">", a.dreg(1)),
    "c.sne.d": lambda a: BinaryOp.dcmp(a.dreg(0), "==", a.dreg(1)).negated(),
    "c.nle.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<=", a.dreg(1)).negated(),
    "c.nlt.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<", a.dreg(1)).negated(),
    "c.nge.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">=", a.dreg(1)).negated(),
    "c.ngt.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">", a.dreg(1)).negated(),
}
CASES_HI_LO: PairInstrMap = {
    # Div and mul output two results, to LO/HI registers. (Format: (hi, lo))
    "div": lambda a: (
        BinaryOp.s32(a.reg(0), "%", a.reg(1)),
        BinaryOp.s32(a.reg(0), "/", a.reg(1)),
    ),
    "divu": lambda a: (
        BinaryOp.u32(a.reg(0), "%", a.reg(1)),
        BinaryOp.u32(a.reg(0), "/", a.reg(1)),
    ),
    "ddiv": lambda a: (
        BinaryOp.s64(a.reg(0), "%", a.reg(1)),
        BinaryOp.s64(a.reg(0), "/", a.reg(1)),
    ),
    "ddivu": lambda a: (
        BinaryOp.u64(a.reg(0), "%", a.reg(1)),
        BinaryOp.u64(a.reg(0), "/", a.reg(1)),
    ),
    # GCC uses the high part of multiplication to optimize division/modulo
    # by constant. Output some nonsense to avoid an error.
    "mult": lambda a: (
        fn_op("MULT_HI", [a.reg(0), a.reg(1)], Type.s32()),
        BinaryOp.int(a.reg(0), "*", a.reg(1)),
    ),
    "multu": lambda a: (
        fn_op("MULTU_HI", [a.reg(0), a.reg(1)], Type.u32()),
        BinaryOp.int(a.reg(0), "*", a.reg(1)),
    ),
    "dmult": lambda a: (
        fn_op("DMULT_HI", [a.reg(0), a.reg(1)], Type.s64()),
        BinaryOp.int64(a.reg(0), "*", a.reg(1)),
    ),
    "dmultu": lambda a: (
        fn_op("DMULTU_HI", [a.reg(0), a.reg(1)], Type.u64()),
        BinaryOp.int64(a.reg(0), "*", a.reg(1)),
    ),
}
CASES_SOURCE_FIRST: InstrMap = {
    # Floating point moving instruction
    "mtc1": lambda a: a.reg(0)
}
CASES_DESTINATION_FIRST: InstrMap = {
    # Flag-setting instructions
    "slt": lambda a: BinaryOp.scmp(a.reg(1), "<", a.reg(2)),
    "slti": lambda a: BinaryOp.scmp(a.reg(1), "<", a.imm(2)),
    "sltu": lambda a: handle_sltu(a),
    "sltiu": lambda a: handle_sltiu(a),
    # Integer arithmetic
    "addi": lambda a: handle_addi(a),
    "addiu": lambda a: handle_addi(a),
    "addu": lambda a: handle_add(a),
    "subu": lambda a: fold_mul_chains(BinaryOp.intptr(a.reg(1), "-", a.reg(2))),
    "negu": lambda a: fold_mul_chains(
        UnaryOp(op="-", expr=as_s32(a.reg(1)), type=Type.s32())
    ),
    "neg": lambda a: fold_mul_chains(
        UnaryOp(op="-", expr=as_s32(a.reg(1)), type=Type.s32())
    ),
    "div.fictive": lambda a: BinaryOp.s32(a.reg(1), "/", a.full_imm(2)),
    "mod.fictive": lambda a: BinaryOp.s32(a.reg(1), "%", a.full_imm(2)),
    # 64-bit integer arithmetic, treated mostly the same as 32-bit for now
    "daddi": lambda a: handle_addi(a),
    "daddiu": lambda a: handle_addi(a),
    "daddu": lambda a: handle_add(a),
    "dsubu": lambda a: fold_mul_chains(BinaryOp.intptr(a.reg(1), "-", a.reg(2))),
    "dnegu": lambda a: fold_mul_chains(
        UnaryOp(op="-", expr=as_s64(a.reg(1)), type=Type.s64())
    ),
    "dneg": lambda a: fold_mul_chains(
        UnaryOp(op="-", expr=as_s64(a.reg(1)), type=Type.s64())
    ),
    # Hi/lo register uses (used after division/multiplication)
    "mfhi": lambda a: a.regs[Register("hi")],
    "mflo": lambda a: a.regs[Register("lo")],
    # Floating point arithmetic
    "add.s": lambda a: handle_add_float(a),
    "sub.s": lambda a: BinaryOp.f32(a.reg(1), "-", a.reg(2)),
    "neg.s": lambda a: UnaryOp("-", as_f32(a.reg(1)), type=Type.f32()),
    "abs.s": lambda a: fn_op("fabsf", [as_f32(a.reg(1))], Type.f32()),
    "sqrt.s": lambda a: fn_op("sqrtf", [as_f32(a.reg(1))], Type.f32()),
    "div.s": lambda a: BinaryOp.f32(a.reg(1), "/", a.reg(2)),
    "mul.s": lambda a: BinaryOp.f32(a.reg(1), "*", a.reg(2)),
    # Double-precision arithmetic
    "add.d": lambda a: handle_add_double(a),
    "sub.d": lambda a: BinaryOp.f64(a.dreg(1), "-", a.dreg(2)),
    "neg.d": lambda a: UnaryOp("-", as_f64(a.dreg(1)), type=Type.f64()),
    "abs.d": lambda a: fn_op("fabs", [as_f64(a.dreg(1))], Type.f64()),
    "sqrt.d": lambda a: fn_op("sqrt", [as_f64(a.dreg(1))], Type.f64()),
    "div.d": lambda a: BinaryOp.f64(a.dreg(1), "/", a.dreg(2)),
    "mul.d": lambda a: BinaryOp.f64(a.dreg(1), "*", a.dreg(2)),
    # Floating point conversions
    "cvt.d.s": lambda a: handle_convert(a.reg(1), Type.f64(), Type.f32()),
    "cvt.d.w": lambda a: handle_convert(a.reg(1), Type.f64(), Type.intish()),
    "cvt.s.d": lambda a: handle_convert(a.dreg(1), Type.f32(), Type.f64()),
    "cvt.s.w": lambda a: handle_convert(a.reg(1), Type.f32(), Type.intish()),
    "cvt.w.d": lambda a: handle_convert(a.dreg(1), Type.s32(), Type.f64()),
    "cvt.w.s": lambda a: handle_convert(a.reg(1), Type.s32(), Type.f32()),
    "cvt.s.u.fictive": lambda a: handle_convert(a.reg(1), Type.f32(), Type.u32()),
    "cvt.u.d.fictive": lambda a: handle_convert(a.dreg(1), Type.u32(), Type.f64()),
    "cvt.u.s.fictive": lambda a: handle_convert(a.reg(1), Type.u32(), Type.f32()),
    "trunc.w.s": lambda a: handle_convert(a.reg(1), Type.s32(), Type.f32()),
    "trunc.w.d": lambda a: handle_convert(a.dreg(1), Type.s32(), Type.f64()),
    # Bit arithmetic
    "ori": lambda a: handle_ori(a),
    "and": lambda a: BinaryOp.int(left=a.reg(1), op="&", right=a.reg(2)),
    "or": lambda a: BinaryOp.int(left=a.reg(1), op="|", right=a.reg(2)),
    "not": lambda a: UnaryOp("~", a.reg(1), type=Type.intish()),
    "nor": lambda a: UnaryOp(
        "~", BinaryOp.int(left=a.reg(1), op="|", right=a.reg(2)), type=Type.intish()
    ),
    "xor": lambda a: BinaryOp.int(left=a.reg(1), op="^", right=a.reg(2)),
    "andi": lambda a: BinaryOp.int(left=a.reg(1), op="&", right=a.unsigned_imm(2)),
    "xori": lambda a: BinaryOp.int(left=a.reg(1), op="^", right=a.unsigned_imm(2)),
    # Shifts
    "sll": lambda a: fold_mul_chains(
        BinaryOp.int(left=a.reg(1), op="<<", right=as_intish(a.imm(2)))
    ),
    "sllv": lambda a: BinaryOp.int(left=a.reg(1), op="<<", right=as_intish(a.reg(2))),
    "srl": lambda a: BinaryOp(
        left=as_u32(a.reg(1)), op=">>", right=as_intish(a.imm(2)), type=Type.u32()
    ),
    "srlv": lambda a: BinaryOp(
        left=as_u32(a.reg(1)), op=">>", right=as_intish(a.reg(2)), type=Type.u32()
    ),
    "sra": lambda a: handle_sra(a),
    "srav": lambda a: BinaryOp(
        left=as_s32(a.reg(1)), op=">>", right=as_intish(a.reg(2)), type=Type.s32()
    ),
    # 64-bit shifts
    "dsll": lambda a: fold_mul_chains(
        BinaryOp.int64(left=a.reg(1), op="<<", right=as_intish(a.imm(2)))
    ),
    "dsll32": lambda a: fold_mul_chains(
        BinaryOp.int64(left=a.reg(1), op="<<", right=imm_add_32(a.imm(2)))
    ),
    "dsllv": lambda a: BinaryOp.int64(
        left=a.reg(1), op="<<", right=as_intish(a.reg(2))
    ),
    "dsrl": lambda a: BinaryOp(
        left=as_u64(a.reg(1)), op=">>", right=as_intish(a.imm(2)), type=Type.u64()
    ),
    "dsrl32": lambda a: BinaryOp(
        left=as_u64(a.reg(1)), op=">>", right=imm_add_32(a.imm(2)), type=Type.u64()
    ),
    "dsrlv": lambda a: BinaryOp(
        left=as_u64(a.reg(1)), op=">>", right=as_intish(a.reg(2)), type=Type.u64()
    ),
    "dsra": lambda a: BinaryOp(
        left=as_s64(a.reg(1)), op=">>", right=as_intish(a.imm(2)), type=Type.s64()
    ),
    "dsra32": lambda a: BinaryOp(
        left=as_s64(a.reg(1)), op=">>", right=imm_add_32(a.imm(2)), type=Type.s64()
    ),
    "dsrav": lambda a: BinaryOp(
        left=as_s64(a.reg(1)), op=">>", right=as_intish(a.reg(2)), type=Type.s64()
    ),
    # Move pseudoinstruction
    "move": lambda a: a.reg(1),
    # Floating point moving instructions
    "mfc1": lambda a: a.reg(1),
    "mov.s": lambda a: a.reg(1),
    "mov.d": lambda a: as_f64(a.dreg(1)),
    # FCSR get
    "cfc1": lambda a: ErrorExpr("cfc1"),
    # Immediates
    "li": lambda a: a.full_imm(1),
    "lui": lambda a: load_upper(a),
    "la": lambda a: handle_la(a),
    # Loading instructions
    "lb": lambda a: handle_load(a, type=Type.s8()),
    "lbu": lambda a: handle_load(a, type=Type.u8()),
    "lh": lambda a: handle_load(a, type=Type.s16()),
    "lhu": lambda a: handle_load(a, type=Type.u16()),
    "lw": lambda a: handle_load(a, type=Type.of_size(32)),
    "lwu": lambda a: handle_load(a, type=Type.u32()),
    "lwc1": lambda a: handle_load(a, type=Type.f32()),
    "ldc1": lambda a: handle_load(a, type=Type.f64()),
    # Unaligned load for the left part of a register (lwl can technically merge
    # with a pre-existing lwr, but doesn't in practice, so we treat this as a
    # standard destination-first operation)
    "lwl": lambda a: handle_lwl(a),
}
CASES_LWR: LwrInstrMap = {
    # Unaligned load for the right part of a register. Only writes a partial
    # register.
    "lwr": lambda a, old_value: handle_lwr(a, old_value),
}


def output_regs_for_instr(
    instr: Instruction, typemap: Optional[TypeMap]
) -> List[Register]:
    def reg_at(index: int) -> List[Register]:
        reg = instr.args[index]
        if not isinstance(reg, Register):
            # We'll deal with this error later
            return []
        ret = [reg]
        if reg.register_name in ["f0", "v0"]:
            ret.append(Register("return"))
        return ret

    mnemonic = instr.mnemonic
    if (
        mnemonic in CASES_JUMPS
        or mnemonic in CASES_STORE
        or mnemonic in CASES_BRANCHES
        or mnemonic in CASES_FLOAT_BRANCHES
        or mnemonic in CASES_IGNORE
        or mnemonic in CASES_NO_DEST
    ):
        return []
    if mnemonic == "jal" and typemap:
        fn_target = instr.args[0]
        if isinstance(fn_target, AsmGlobalSymbol):
            c_fn = typemap.functions.get(fn_target.symbol_name)
            if c_fn and c_fn.ret_type is None:
                return []
    if mnemonic in CASES_FN_CALL:
        return list(map(Register, ["return", "f0", "v0", "v1"]))
    if mnemonic in CASES_SOURCE_FIRST:
        return reg_at(1)
    if mnemonic in CASES_DESTINATION_FIRST:
        return reg_at(0)
    if mnemonic in CASES_LWR:
        return reg_at(0)
    if mnemonic in CASES_FLOAT_COMP:
        return [Register("condition_bit")]
    if mnemonic in CASES_HI_LO:
        return [Register("hi"), Register("lo")]
    if instr.args and isinstance(instr.args[0], Register):
        return reg_at(0)
    return []


def regs_clobbered_until_dominator(
    node: Node, typemap: Optional[TypeMap]
) -> Set[Register]:
    if node.immediate_dominator is None:
        return set()
    seen = {node.immediate_dominator}
    stack = copy(node.parents)
    clobbered = set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        for instr in n.block.instructions:
            with current_instr(instr):
                clobbered.update(output_regs_for_instr(instr, typemap))
                if instr.mnemonic in CASES_FN_CALL:
                    clobbered.update(TEMP_REGS)
        stack.extend(n.parents)
    return clobbered


def reg_always_set(
    node: Node, reg: Register, typemap: Optional[TypeMap], *, dom_set: bool
) -> bool:
    if node.immediate_dominator is None:
        return False
    seen = {node.immediate_dominator}
    stack = copy(node.parents)
    while stack:
        n = stack.pop()
        if n == node.immediate_dominator and not dom_set:
            return False
        if n in seen:
            continue
        seen.add(n)
        clobbered: Optional[bool] = None
        for instr in n.block.instructions:
            with current_instr(instr):
                if instr.mnemonic in CASES_FN_CALL and reg in TEMP_REGS:
                    clobbered = True
                if reg in output_regs_for_instr(instr, typemap):
                    clobbered = False
        if clobbered == True:
            return False
        if clobbered is None:
            stack.extend(n.parents)
    return True


def assign_phis(used_phis: List[PhiExpr], stack_info: StackInfo) -> None:
    i = 0
    # Iterate over used phis until there are no more remaining. New ones may
    # appear during iteration, hence the while loop.
    while i < len(used_phis):
        phi = used_phis[i]
        assert phi.num_usages > 0
        # assert len(phi.node.parents) >= 2
        if len(phi.node.parents) < 2:
            i += 1
            continue
        exprs = []
        for node in phi.node.parents:
            block_info = node.block.block_info
            # assert isinstance(block_info, BlockInfo)
            if not block_info:
                i += 1
                continue
            exprs.append(block_info.final_register_states[phi.reg])

        first_uw = early_unwrap(exprs[0])
        if all(early_unwrap(e) == first_uw for e in exprs[1:]):
            # All the phis have the same value (e.g. because we recomputed an
            # expression after a store, or restored a register after a function
            # call). Just use that value instead of introducing a phi node.
            # TODO: the unwrapping here is necessary, but also kinda sketchy:
            # we may set as replacement_expr an expression that really shouldn't
            # be repeated, e.g. a StructAccess. It would make sense to use less
            # eager unwrapping, and/or to emit an EvalOnceExpr at this point
            # (though it's too late for it to be able to participate in the
            # prevent_later_uses machinery).
            phi.replacement_expr = as_type(first_uw, phi.type, silent=True)
            for e in exprs:
                e.type.unify(phi.type)
            for _ in range(phi.num_usages):
                first_uw.use()
        else:
            for node in phi.node.parents:
                block_info = node.block.block_info
                assert isinstance(block_info, BlockInfo)
                expr = block_info.final_register_states[phi.reg]
                if isinstance(expr, PhiExpr):
                    # Explicitly mark how the expression is used if it's a phi,
                    # so we can propagate phi sets (to get rid of temporaries).
                    expr.use(from_phi=phi)
                else:
                    expr.use()
                typed_expr = as_type(expr, phi.type, silent=True)
                block_info.to_write.append(SetPhiStmt(phi, typed_expr))
        i += 1

    name_counter: Dict[Register, int] = {}
    for phi in used_phis:
        if not phi.replacement_expr and phi.propagates_to() == phi:
            counter = name_counter.get(phi.reg, 0) + 1
            name_counter[phi.reg] = counter
            prefix = f"phi_{phi.reg.register_name}"
            phi.name = f"{prefix}_{counter}" if counter > 1 else prefix
            stack_info.phi_vars.append(phi)


def compute_has_custom_return(nodes: List[Node]) -> None:
    """Propagate the "has_custom_return" property using fixed-point iteration."""
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if isinstance(n, TerminalNode):
                continue
            block_info = n.block.block_info
            assert isinstance(block_info, BlockInfo)
            if block_info.has_custom_return or block_info.has_function_call:
                continue
            for p in n.parents:
                block_info2 = p.block.block_info
                # assert isinstance(block_info2, BlockInfo)
                if not block_info2:
                    continue
                if block_info2.has_custom_return:
                    block_info.has_custom_return = True
                    changed = True


def translate_node_body(node: Node, regs: RegInfo, stack_info: StackInfo) -> BlockInfo:
    """
    Given a node and current register contents, return a BlockInfo containing
    the translated AST for that node.
    """

    to_write: List[Union[Statement]] = []
    local_var_writes: Dict[LocalVar, Tuple[Register, Expression]] = {}
    subroutine_args: List[Tuple[Expression, SubroutineArg]] = []
    branch_condition: Optional[Condition] = None
    switch_value: Optional[Expression] = None
    has_custom_return: bool = False
    has_function_call: bool = False

    def eval_once(
        expr: Expression,
        *,
        emit_exactly_once: bool,
        trivial: bool,
        prefix: str = "",
        reuse_var: Optional[Var] = None,
    ) -> EvalOnceExpr:
        if emit_exactly_once:
            # (otherwise this will be marked used once num_usages reaches 1)
            expr.use()
        assert reuse_var or prefix
        if prefix == "condition_bit":
            prefix = "cond"
        var = reuse_var or Var(stack_info, "temp_" + prefix)
        expr = EvalOnceExpr(
            wrapped_expr=expr,
            var=var,
            type=expr.type,
            emit_exactly_once=emit_exactly_once,
            trivial=trivial,
        )
        var.num_usages += 1
        stmt = EvalOnceStmt(expr)
        to_write.append(stmt)
        stack_info.temp_vars.append(stmt)
        return expr

    def prevent_later_uses(expr_filter: Callable[[Expression], bool]) -> None:
        """Prevent later uses of registers whose contents match a callback filter."""
        for r in regs.contents.keys():
            e = regs.get_raw(r)
            assert e is not None
            if not isinstance(e, ForceVarExpr) and expr_filter(e):
                # Mark the register as "if used, emit the expression's once
                # var". I think we should always have a once var at this point,
                # but if we don't, create one.
                if not isinstance(e, EvalOnceExpr):
                    e = eval_once(
                        e,
                        emit_exactly_once=False,
                        trivial=False,
                        prefix=r.register_name,
                    )
                regs[r] = ForceVarExpr(e, type=e.type)

    def prevent_later_value_uses(sub_expr: Expression) -> None:
        """Prevent later uses of registers that recursively contain a given
        subexpression."""
        # Unused PassedInArg are fine; they can pass the uses_expr test simply based
        # on having the same variable name. If we didn't filter them out here it could
        # cause them to be incorrectly passed as function arguments -- the function
        # call logic sees an opaque wrapper and doesn't realize that they are unused
        # arguments that should not be passed on.
        prevent_later_uses(
            lambda e: uses_expr(e, lambda e2: e2 == sub_expr)
            and not (isinstance(e, PassedInArg) and not e.copied)
        )

    def prevent_later_function_calls() -> None:
        """Prevent later uses of registers that recursively contain a function call."""
        prevent_later_uses(lambda e: uses_expr(e, lambda e2: isinstance(e2, FuncCall)))

    def prevent_later_reads() -> None:
        """Prevent later uses of registers that recursively contain a read."""
        contains_read = lambda e: isinstance(e, (StructAccess, ArrayAccess))
        prevent_later_uses(lambda e: uses_expr(e, contains_read))

    def set_reg_maybe_return(reg: Register, expr: Expression) -> None:
        nonlocal has_custom_return
        regs[reg] = expr
        if reg.register_name in ["f0", "v0"]:
            regs[Register("return")] = expr
            has_custom_return = True

    def set_reg(reg: Register, expr: Optional[Expression]) -> None:
        if expr is None:
            if reg in regs:
                del regs[reg]
            return

        if isinstance(expr, LocalVar):
            if (
                isinstance(node, ReturnNode)
                and stack_info.maybe_get_register_var(reg)
                and (
                    stack_info.callee_save_reg_locations.get(reg) == expr.value
                    or (
                        reg == Register("ra")
                        and stack_info.return_addr_location == expr.value
                    )
                )
            ):
                # Elide saved register restores with --reg-vars (it doesn't
                # matter in other cases).
                return
            if expr in local_var_writes:
                # Elide register restores (only for the same register for now,
                # to be conversative).
                orig_reg, orig_expr = local_var_writes[expr]
                if orig_reg == reg:
                    expr = orig_expr

        uw_expr = expr
        if not isinstance(expr, Literal):
            expr = eval_once(
                expr,
                emit_exactly_once=False,
                trivial=is_trivial_expression(expr),
                prefix=reg.register_name,
            )

        if reg == Register("zero"):
            # Emit the expression as is. It's probably a volatile load.
            expr.use()
            to_write.append(ExprStmt(expr))
        else:
            dest = stack_info.maybe_get_register_var(reg)
            if dest is not None:
                stack_info.use_register_var(dest)
                # Avoid emitting x = x, but still refresh EvalOnceExpr's etc.
                if not (isinstance(uw_expr, RegisterVar) and uw_expr.register == reg):
                    source = as_type(expr, dest.type, True)
                    source.use()
                    to_write.append(StoreStmt(source=source, dest=dest))
                expr = dest
            set_reg_maybe_return(reg, expr)

    def clear_caller_save_regs() -> None:
        for reg in TEMP_REGS:
            if reg in regs:
                del regs[reg]

    def overwrite_reg(reg: Register, expr: Expression) -> None:
        prev = regs.get_raw(reg)
        at = regs.get_raw(Register("at"))
        if isinstance(prev, ForceVarExpr):
            prev = prev.wrapped_expr
        if (
            not isinstance(prev, EvalOnceExpr)
            or isinstance(expr, Literal)
            or reg == Register("sp")
            or reg == Register("at")
            or not prev.type.unify(expr.type)
            or (at is not None and uses_expr(at, lambda e2: e2 == prev))
        ):
            set_reg(reg, expr)
        else:
            # TODO: This is a bit heavy-handed: we're preventing later uses
            # even though we are not sure whether we will actually emit the
            # overwrite. Doing this properly is hard, however -- it would
            # involve tracking "time" for uses, and sometimes moving timestamps
            # backwards when EvalOnceExpr's get emitted as vars.
            if reg in regs:
                # For ease of debugging, don't let prevent_later_value_uses see
                # the register we're writing to.
                del regs[reg]

            prevent_later_value_uses(prev)
            set_reg_maybe_return(
                reg,
                eval_once(
                    expr,
                    emit_exactly_once=False,
                    trivial=is_trivial_expression(expr),
                    reuse_var=prev.var,
                ),
            )

    def process_instr(instr: Instruction) -> None:
        nonlocal branch_condition, switch_value, has_custom_return, has_function_call

        mnemonic = instr.mnemonic
        args = InstrArgs(instr.args, regs, stack_info)

        # Figure out what code to generate!
        if mnemonic in CASES_IGNORE:
            pass

        elif mnemonic in CASES_STORE:
            # Store a value in a permanent place.
            to_store = CASES_STORE[mnemonic](args)
            if to_store is None:
                # Elided register preserval.
                pass
            elif isinstance(to_store.dest, SubroutineArg):
                # About to call a subroutine with this argument. Skip arguments for the
                # first four stack slots; they are also passed in registers.
                if to_store.dest.value >= 0x10:
                    subroutine_args.append((to_store.source, to_store.dest))
            else:
                if isinstance(to_store.dest, LocalVar):
                    stack_info.add_local_var(to_store.dest)
                    raw_value = to_store.source
                    if isinstance(raw_value, Cast) and raw_value.reinterpret:
                        # When preserving values on the stack across function calls,
                        # ignore the type of the stack variable. The same stack slot
                        # might be used to preserve values of different types.
                        raw_value = raw_value.expr
                    local_var_writes[to_store.dest] = (args.reg_ref(0), raw_value)
                # Emit a write. This includes four steps:
                # - mark the expression as used (since writes are always emitted)
                # - mark the dest used (if it's a struct access it needs to be
                # evaluated, though ideally we would not mark the top-level expression
                # used; it may cause early emissions that shouldn't happen)
                # - mark other usages of the dest as "emit before this point if used".
                # - emit the actual write.
                #
                # Note that the prevent_later_value_uses step happens after use(), since
                # the stored expression is allowed to reference its destination var,
                # but before the write is written, since prevent_later_value_uses might
                # emit writes of its own that should go before this write. In practice
                # that probably never occurs -- all relevant register contents should be
                # EvalOnceExpr's that can be emitted at their point of creation, but
                # I'm not 100% certain that that's always the case and will remain so.
                to_store.source.use()
                to_store.dest.use()
                prevent_later_value_uses(to_store.dest)
                prevent_later_function_calls()
                to_write.append(to_store)

        elif mnemonic in CASES_SOURCE_FIRST:
            # Just 'mtc1'. It's reversed, so we have to specially handle it.
            set_reg(args.reg_ref(1), CASES_SOURCE_FIRST[mnemonic](args))

        elif mnemonic in CASES_BRANCHES:
            assert branch_condition is None
            branch_condition = CASES_BRANCHES[mnemonic](args)

        elif mnemonic in CASES_FLOAT_BRANCHES:
            assert branch_condition is None
            cond_bit = regs[Register("condition_bit")]
            if not isinstance(cond_bit, BinaryOp):
                cond_bit = ExprCondition(cond_bit, type=cond_bit.type)
            if mnemonic == "bc1t":
                branch_condition = cond_bit
            elif mnemonic == "bc1f":
                branch_condition = cond_bit.negated()

        elif mnemonic in CASES_JUMPS:
            assert mnemonic == "jr"
            if args.reg_ref(0) == Register("ra"):
                # Return from the function.
                assert isinstance(node, ReturnNode)
            else:
                # Switch jump.
                assert isinstance(node, SwitchNode)
                switch_value = args.reg(0)

        elif mnemonic in CASES_FN_CALL:
            if mnemonic == "jal":
                fn_target = args.imm(0)
                if isinstance(fn_target, AddressOf) and isinstance(
                    fn_target.expr, GlobalSymbol
                ):
                    pass
                elif isinstance(fn_target, Literal):
                    pass
                else:
                    raise DecompFailure("The target of jal must be a label, not {arg}")
            else:
                assert mnemonic == "jalr"
                if args.count() == 1:
                    fn_target = args.reg(0)
                elif args.count() == 2:
                    if args.reg_ref(0) != Register("ra"):
                        raise DecompFailure(
                            "Two-argument form of jalr is not supported."
                        )
                    fn_target = args.reg(1)
                else:
                    raise DecompFailure(f"jalr takes 2 arguments, {args.count()} given")

            fn_target = as_function_ptr(fn_target)
            fn_sig = fn_target.type.get_function_pointer_signature()
            assert fn_sig is not None, "known function pointers must have a signature"
            abi_slots, possible_regs = function_abi(fn_sig, for_call=True)

            func_args: List[Expression] = []
            for slot in abi_slots:
                if slot.reg:
                    func_args.append(as_type(regs[slot.reg], slot.type, True))

            valid_extra_regs: Set[str] = set()
            for register in possible_regs:
                expr = regs.get_raw(register)
                if expr is None:
                    continue

                # Don't pass this register if lower numbered ones are undefined.
                # Following the o32 ABI, register order can be a prefix of either:
                # a0, a1, a2, a3
                # f12, a1, a2, a3
                # f12, f14, a2, a3
                # f12, f13, a2, a3
                # f12, f13, f14, f15
                require: Optional[List[str]] = None
                if register == possible_regs[0]:
                    # For varargs, a subset of a0 .. a3 may be used. Don't check
                    # earlier registers for the first member of that subset.
                    pass
                elif register == Register("f13") or register == Register("f14"):
                    require = ["f12"]
                elif register == Register("a1"):
                    require = ["a0", "f12"]
                elif register == Register("a2"):
                    require = ["a1", "f13", "f14"]
                elif register == Register("a3"):
                    require = ["a2"]
                if require and not any(r in valid_extra_regs for r in require):
                    continue

                valid_extra_regs.add(register.register_name)

                if register == Register("f13"):
                    # We don't pass in f13 or f15 because they will often only
                    # contain SecondF64Half(), and otherwise would need to be
                    # merged with f12/f14 which we don't have logic for right
                    # now. However, f13 can still matter for whether a2 should
                    # be passed, and so is kept in possible_regs.
                    continue

                # Skip registers that are untouched from our initial parameter
                # list. This is sometimes wrong (can give both false positives
                # and negatives), but having a heuristic here is unavoidable
                # without access to function signatures, or when dealing with
                # varargs functions. Decompiling multiple functions at once
                # would help. TODO: don't do this in the middle of the argument
                # list, except for f12 if a0 is passed and such.
                if isinstance(expr, PassedInArg) and not expr.copied:
                    continue

                func_args.append(expr)

            # Add the arguments after a3.
            # TODO: limit this and unify types based on abi_slots
            subroutine_args.sort(key=lambda a: a[1].value)
            for arg in subroutine_args:
                func_args.append(arg[0])

            if not fn_sig.params_known:
                while len(func_args) > len(fn_sig.params):
                    fn_sig.params.append(FunctionParam())
            for i, (arg_expr, param) in enumerate(zip(func_args, fn_sig.params)):
                func_args[i] = as_type(arg_expr, param.type, True)

            # Reset subroutine_args, for the next potential function call.
            subroutine_args.clear()

            call: Expression = FuncCall(fn_target, func_args, fn_sig.return_type)
            call = eval_once(call, emit_exactly_once=True, trivial=False, prefix="ret")

            # Clear out caller-save registers, for clarity and to ensure that
            # argument regs don't get passed into the next function.
            clear_caller_save_regs()

            # Prevent reads and function calls from moving across this call.
            # This isn't really right, because this call might be moved later,
            # and then this prevention should also be... but it's the best we
            # can do with the current code architecture.
            prevent_later_function_calls()
            prevent_later_reads()

            # We may not know what this function's return registers are --
            # $f0, $v0 or ($v0,$v1) or $f0 -- but we don't really care,
            # it's fine to be liberal here and put the return value in all
            # of them. (It's not perfect for u64's, but that's rare anyway.)
            # However, if we have type information that says the function is
            # void, then don't set any of these -- it might cause us to
            # believe the function we're decompiling is non-void.
            # Note that this logic is duplicated in output_regs_for_instr.
            if not fn_sig.return_type.is_void():
                regs[Register("f0")] = eval_once(
                    Cast(
                        expr=call, reinterpret=True, silent=True, type=Type.floatish()
                    ),
                    emit_exactly_once=False,
                    trivial=False,
                    prefix="f0",
                )
                regs[Register("f1")] = SecondF64Half()
                regs[Register("v0")] = eval_once(
                    Cast(expr=call, reinterpret=True, silent=True, type=Type.intptr()),
                    emit_exactly_once=False,
                    trivial=False,
                    prefix="v0",
                )
                regs[Register("v1")] = eval_once(
                    as_u32(
                        Cast(expr=call, reinterpret=True, silent=False, type=Type.u64())
                    ),
                    emit_exactly_once=False,
                    trivial=False,
                    prefix="v1",
                )
                regs[Register("return")] = call

            has_custom_return = False
            has_function_call = True

        elif mnemonic in CASES_FLOAT_COMP:
            expr = CASES_FLOAT_COMP[mnemonic](args)
            regs[Register("condition_bit")] = expr

        elif mnemonic in CASES_HI_LO:
            hi, lo = CASES_HI_LO[mnemonic](args)
            set_reg(Register("hi"), hi)
            set_reg(Register("lo"), lo)

        elif mnemonic in CASES_NO_DEST:
            expr = CASES_NO_DEST[mnemonic](args)
            expr.use()
            to_write.append(ExprStmt(expr))

        elif mnemonic in CASES_DESTINATION_FIRST:
            target = args.reg_ref(0)
            val = CASES_DESTINATION_FIRST[mnemonic](args)
            if False and target in args.raw_args[1:]:
                # IDO tends to keep variables within single registers. Thus,
                # if source = target, overwrite that variable instead of
                # creating a new one.
                # XXX This code path is disabled due to known bugs, and kept
                # only to make it easy to experiment with. It should be removed
                # entirely at some point, hopefully to be replaced by some more
                # stable alternative.
                overwrite_reg(target, val)
            else:
                set_reg(target, val)
            mn_parts = mnemonic.split(".")
            if (len(mn_parts) >= 2 and mn_parts[1] == "d") or mnemonic == "ldc1":
                set_reg(target.other_f64_reg(), SecondF64Half())

        elif mnemonic in CASES_LWR:
            assert mnemonic == "lwr"
            target = args.reg_ref(0)
            old_value = args.reg(0)
            val = CASES_LWR[mnemonic](args, old_value)
            set_reg(target, val)

        else:
            expr = ErrorExpr(f"unknown instruction: {instr}")
            if args.count() >= 1 and isinstance(args.raw_arg(0), Register):
                reg = args.reg_ref(0)
                expr = eval_once(
                    expr,
                    emit_exactly_once=True,
                    trivial=False,
                    prefix=reg.register_name,
                )
                if reg != Register("zero"):
                    set_reg_maybe_return(reg, expr)
            else:
                to_write.append(ExprStmt(expr))

    for instr in node.block.instructions:
        with current_instr(instr):
            process_instr(instr)

    if branch_condition is not None:
        branch_condition.use()
    if switch_value is not None:
        switch_value.use()
    return_value: Optional[Expression] = None
    if isinstance(node, ReturnNode):
        return_value = regs.get_raw(Register("return"))
    return BlockInfo(
        to_write,
        return_value,
        switch_value,
        branch_condition,
        regs,
        has_custom_return=has_custom_return,
        has_function_call=has_function_call,
    )


def translate_graph_from_block(
    node: Node,
    regs: RegInfo,
    stack_info: StackInfo,
    used_phis: List[PhiExpr],
    return_blocks: List[BlockInfo],
    options: Options,
) -> None:
    """
    Given a FlowGraph node and a dictionary of register contents, give that node
    its appropriate BlockInfo (which contains the AST of its code).
    """

    if options.debug:
        print(f"\nNode in question: {node}")

    # Translate the given node and discover final register states.
    try:
        block_info = translate_node_body(node, regs, stack_info)
        if options.debug:
            print(block_info)
    except Exception as e:  # TODO: handle issues better
        if options.stop_on_error:
            raise

        instr: Optional[Instruction] = None
        if isinstance(e, InstrProcessingFailure) and isinstance(e.__cause__, Exception):
            instr = e.instr
            e = e.__cause__

        if isinstance(e, DecompFailure):
            emsg = str(e)
            print(emsg)
        else:
            tb = e.__traceback__
            traceback.print_exception(None, e, tb)
            emsg = str(e) or traceback.format_tb(tb)[-1]
            emsg = emsg.strip().split("\n")[-1].strip()

        error_stmts: List[Statement] = [CommentStmt(f"Error: {emsg}")]
        if instr is not None:
            print(
                f"Error occurred while processing instruction: {instr}", file=sys.stderr
            )
            error_stmts.append(CommentStmt(f"At instruction: {instr}"))
        print(file=sys.stderr)
        block_info = BlockInfo(
            error_stmts,
            None,
            None,
            ErrorExpr(),
            regs,
            has_custom_return=False,
            has_function_call=False,
        )

    node.block.add_block_info(block_info)
    if isinstance(node, ReturnNode):
        return_blocks.append(block_info)

    # Translate everything dominated by this node, now that we know our own
    # final register state. This will eventually reach every node.
    typemap = stack_info.global_info.typemap
    for child in node.immediately_dominates:
        if isinstance(child, TerminalNode):
            continue
        new_contents = regs.contents.copy()
        phi_regs = regs_clobbered_until_dominator(child, typemap)
        for reg in phi_regs:
            if reg_always_set(child, reg, typemap, dom_set=(reg in regs)):
                var = stack_info.maybe_get_register_var(reg)
                if var is not None:
                    new_contents[reg] = var
                else:
                    new_contents[reg] = PhiExpr(
                        reg=reg, node=child, used_phis=used_phis, type=Type.any_reg()
                    )
            elif reg in new_contents:
                del new_contents[reg]
        new_regs = RegInfo(contents=new_contents, stack_info=stack_info)
        translate_graph_from_block(
            child, new_regs, stack_info, used_phis, return_blocks, options
        )


def resolve_types_late(stack_info: StackInfo) -> None:
    """
    After translating a function, perform a final type-resolution pass.
    """
    # Use dereferences to determine pointer types
    struct_type_map = stack_info.get_struct_type_map()
    for var, offset_type_map in struct_type_map.items():
        if len(offset_type_map) == 1 and 0 in offset_type_map:
            # var was probably a plain pointer, not a struct
            # Try to unify it with the appropriate pointer type,
            # to fill in the type if it does not already have one
            type = offset_type_map[0]
            var.type.unify(Type.ptr(type))


@attr.s
class GlobalInfo:
    asm_data: AsmData = attr.ib()
    local_functions: Set[str] = attr.ib()
    typemap: Optional[TypeMap] = attr.ib()
    symbol_type_map: Dict[str, "Type"] = attr.ib(factory=dict)

    def asm_data_value(self, sym_name: str) -> Optional[AsmDataEntry]:
        entry = self.asm_data.values.get(sym_name)
        if entry and entry.is_readonly:
            return entry
        return None

    def symbol_type(self, sym_name: str) -> Type:
        if sym_name not in self.symbol_type_map:
            self.symbol_type_map[sym_name] = Type.any()
        return self.symbol_type_map[sym_name]

    def global_symbol(self, sym_name: str) -> "GlobalSymbol":
        return GlobalSymbol(
            symbol_name=sym_name,
            type=self.symbol_type(sym_name),
        )

    def address_of_gsym(self, sym_name: str) -> Expression:
        ent = self.asm_data_value(sym_name)
        if ent and ent.is_string and ent.data and isinstance(ent.data[0], bytes):
            return StringLiteral(ent.data[0], type=Type.ptr(Type.s8()))
        sym = self.global_symbol(sym_name)
        type = Type.ptr(sym.type)
        if self.typemap:
            ctype = self.typemap.var_types.get(sym_name)
            if ctype:
                ctype_type, is_ptr = ptr_type_from_ctype(ctype, self.typemap)
                if is_ptr:
                    return as_type(sym, ctype_type, True)
                else:
                    type.unify(ctype_type)
                    type = ctype_type
        return AddressOf(sym, type=type)

    def data_to_initializer(
        self, type: Type, data: List[Union[str, bytes]], fmt: Formatter
    ) -> Optional[str]:
        data = data[:]

        def read_uint(n: int) -> Optional[int]:
            """Read the next `n` bytes from `data` as an (long) integer"""
            assert 0 < n <= 8
            if not data or not isinstance(data[0], bytes):
                return None
            if len(data[0]) < n:
                return None
            bs = data[0][:n]
            data[0] = data[0][n:]
            if not data[0]:
                del data[0]
            value = 0
            for b in bs:
                value = (value << 8) | b
            return value

        def read_pointer() -> Optional[Expression]:
            """Read the next label from `data`"""
            if not data:
                return None

            if not isinstance(data[0], str):
                # Bare pointer
                value = read_uint(4)
                if value is None:
                    return None
                return Literal(value=value)

            # Pointer label
            label = data.pop(0)
            assert isinstance(label, str)
            return self.address_of_gsym(label)

        if type.is_int() or type.is_float():
            size_bits = type.get_size_bits()
            if size_bits == 0:
                return None
            elif size_bits is None:
                # Unknown size; guess 32 bits
                size_bits = 32
            value = read_uint(size_bits // 8)
            if value is not None:
                return Literal(value, type).format(fmt)

        if type.is_pointer():
            ptr = read_pointer()
            if ptr is not None:
                return as_type(ptr, type, True).format(fmt)

        if type.is_ctype():
            # TODO: Generate initializers for structs/arrays/etc.
            return None

        # Type kinds K_FN and K_VOID do not have initializers
        return None

    def global_decls(self, fmt: Formatter) -> str:
        # Format labels from symbol_type_map into global declarations.
        # As the initializers are formatted, this may cause more symbols
        # to be added to the symbol_type_map.
        lines = []
        processed_names: Set[str] = set()
        while True:
            names = self.symbol_type_map.keys() - processed_names
            if not names:
                break
            for name in sorted(names):
                processed_names.add(name)
                type = self.symbol_type_map[name]

                # Is the label defined in this unit (in the active AsmData file(s))
                is_in_file = (
                    name in self.asm_data.values or name in self.local_functions
                )
                # Is the label externally visible (mentioned in the context file)
                is_global = self.typemap and name in self.typemap.var_types
                if not is_in_file and is_global:
                    # Skip externally-declared symbols that are defined in other files
                    continue
                if is_in_file and is_global and type.is_function():
                    # Skip externally-declared functions that are defined here
                    continue

                sort_order = (not type.is_function(), is_global, is_in_file, name)
                qualifier = ""
                initializer = ""
                comments = []

                # Determine type qualifier: static, extern, or neither
                if is_in_file and is_global:
                    qualifier = ""
                elif is_in_file:
                    qualifier = "static"
                else:
                    qualifier = "extern"

                if type.is_function():
                    comments.append(qualifier)
                    qualifier = ""

                # Try to convert the data from .data/.rodata into an initializer
                entry = self.asm_data.values.get(name)
                if entry is not None:
                    if entry.is_readonly:
                        comments.append("const")
                    data_size = entry.size_bytes()
                    type_size_bits = type.get_size_bits()
                    if type_size_bits and data_size != type_size_bits // 8:
                        # TODO: Use heuristic to detect padding
                        type_size = type_size_bits // 8
                        if data_size // type_size > 1:
                            comments.append(f"array: [{data_size // type_size}]")
                        if data_size % type_size != 0:
                            comments.append(f"extra bytes: {data_size % type_size}")
                    elif entry.is_readonly and type.is_float():
                        # Float constants are almost always inlined and can be skipped
                        continue
                    initializer = self.data_to_initializer(type, entry.data, fmt) or ""

                qualifier = f"{qualifier} " if qualifier else ""
                initializer = f" = {initializer}" if initializer else ""
                comment = f" // {'; '.join(comments)}" if comments else ""
                lines.append(
                    (
                        sort_order,
                        f"{qualifier}{type.to_decl(name, fmt)}{initializer};{comment}\n",
                    )
                )
        lines.sort()
        return "".join(line for _, line in lines)


@attr.s
class FunctionInfo:
    stack_info: StackInfo = attr.ib()
    flow_graph: FlowGraph = attr.ib()
    return_type: Type = attr.ib()


def translate_to_ast(
    function: Function,
    flow_graph: FlowGraph,
    options: Options,
    global_info: GlobalInfo,
) -> FunctionInfo:
    """
    Given a function, produce a FlowGraph that both contains control-flow
    information and has AST transformations for each block of code and
    branch condition.
    """
    # Initialize info about the function.
    stack_info = get_stack_info(function, global_info, flow_graph)
    typemap = global_info.typemap

    initial_regs: Dict[Register, Expression] = {
        Register("sp"): GlobalSymbol("sp", type=Type.ptr()),
        **{reg: stack_info.saved_reg_symbol(reg.register_name) for reg in SAVED_REGS},
    }

    def make_arg(offset: int, type: Type) -> PassedInArg:
        assert offset % 4 == 0
        return PassedInArg(offset, copied=False, stack_info=stack_info, type=type)

    if typemap and function.name in typemap.functions:
        fn_type = type_from_ctype(typemap.functions[function.name].type, typemap)
        fn_decl_provided = True
    else:
        fn_type = Type.function()
        fn_decl_provided = False
    fn_type.unify(global_info.symbol_type(function.name))

    fn_sig = Type.ptr(fn_type).get_function_pointer_signature()
    assert fn_sig is not None, "fn_type is known to be a function"
    return_type = fn_sig.return_type
    stack_info.is_variadic = fn_sig.is_variadic

    if fn_sig.params_known:
        abi_slots, possible_regs = function_abi(fn_sig, for_call=False)
        for slot in abi_slots:
            stack_info.add_known_param(slot.offset, slot.name, slot.type)
            if slot.reg is not None:
                initial_regs[slot.reg] = make_arg(slot.offset, slot.type)
        for reg in possible_regs:
            offset = 4 * int(reg.register_name[1])
            initial_regs[reg] = make_arg(offset, Type.any_reg())
    else:
        initial_regs.update(
            {
                Register("a0"): make_arg(0, Type.intptr()),
                Register("a1"): make_arg(4, Type.any_reg()),
                Register("a2"): make_arg(8, Type.any_reg()),
                Register("a3"): make_arg(12, Type.any_reg()),
                Register("f12"): make_arg(0, Type.floatish()),
                Register("f14"): make_arg(4, Type.floatish()),
            }
        )

    if options.reg_vars == ["saved"]:
        reg_vars = SAVED_REGS
    elif options.reg_vars == ["most"]:
        reg_vars = SAVED_REGS + SIMPLE_TEMP_REGS
    elif options.reg_vars == ["all"]:
        reg_vars = SAVED_REGS + SIMPLE_TEMP_REGS + ARGUMENT_REGS
    else:
        reg_vars = list(map(Register, options.reg_vars))
    for reg in reg_vars:
        stack_info.add_register_var(reg)

    if options.debug:
        print(stack_info)
        print("\nNow, we attempt to translate:")

    start_reg: RegInfo = RegInfo(contents=initial_regs, stack_info=stack_info)
    used_phis: List[PhiExpr] = []
    return_blocks: List[BlockInfo] = []
    translate_graph_from_block(
        flow_graph.entry_node(),
        start_reg,
        stack_info,
        used_phis,
        return_blocks,
        options,
    )

    # A function can have a return type if all return nodes have return values.
    # But without a provided signature, we only mark a function has having a
    # return type if there is a nontrivial ("custom") return value (e.g. one
    # not from a function call)
    # TODO: check that the values aren't read from for some other purpose.
    compute_has_custom_return(flow_graph.nodes)
    could_have_return = all(b.return_value is not None for b in return_blocks)
    has_custom_return = any(b.has_custom_return for b in return_blocks)

    if options.void or return_type.is_void():
        has_return = False
    elif could_have_return and (fn_decl_provided or has_custom_return):
        has_return = True
    else:
        has_return = False

    if has_return:
        for b in return_blocks:
            if b.return_value is not None:
                ret_val = as_type(b.return_value, return_type, True)
                b.return_value = ret_val
                ret_val.use()
    else:
        for b in return_blocks:
            b.return_value = None
        return_type.unify(Type.void())

    if not fn_sig.params_known:
        while len(fn_sig.params) < len(stack_info.arguments):
            fn_sig.params.append(FunctionParam())
        for param, arg in zip(fn_sig.params, stack_info.arguments):
            param.type.unify(arg.type)
            if not param.name:
                param.name = arg.format(Formatter())

    assign_phis(used_phis, stack_info)
    resolve_types_late(stack_info)

    if options.pdb_translate:
        import pdb

        v: Dict[str, object] = {}
        fmt = Formatter()
        for local in stack_info.local_vars:
            var_name = local.format(fmt)
            v[var_name] = local
        for temp in stack_info.temp_vars:
            if temp.need_decl():
                var_name = temp.expr.var.format(fmt)
                v[var_name] = temp.expr
        for phi in stack_info.phi_vars:
            assert phi.name is not None
            v[phi.name] = phi
        pdb.set_trace()

    return FunctionInfo(stack_info, flow_graph, return_type)
