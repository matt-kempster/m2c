import struct
import sys
import traceback
import typing
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import attr

from .c_types import Function as CFunction
from .c_types import (
    TypeMap,
    function_arg_size_align,
    get_primitive_list,
    is_struct_type,
)
from .error import DecompFailure
from .flow_graph import (
    FlowGraph,
    Function,
    Node,
    ReturnNode,
    SwitchNode,
    build_flowgraph,
)
from .options import Options
from .parse_file import Rodata
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
    Type,
    find_substruct_array,
    get_field,
    get_pointer_target,
    ptr_type_from_ctype,
    type_from_ctype,
)

ARGUMENT_REGS = list(map(Register, ["a0", "a1", "a2", "a3", "f12", "f14"]))

TEMP_REGS = ARGUMENT_REGS + list(
    map(
        Register,
        [
            "at",
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
            "hi",
            "lo",
            "condition_bit",
            "return",
        ],
    )
)

SAVED_REGS = list(
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
        if not silent:
            return Cast(expr=expr, reinterpret=True, silent=False, type=type)
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


def as_intish(expr: "Expression") -> "Expression":
    return as_type(expr, Type.intish(), True)


def as_intptr(expr: "Expression") -> "Expression":
    return as_type(expr, Type.intptr(), True)


def as_ptr(expr: "Expression") -> "Expression":
    return as_type(expr, Type.ptr(), True)


@attr.s
class StackInfo:
    function: Function = attr.ib()
    rodata: Rodata = attr.ib()
    typemap: Optional[TypeMap] = attr.ib()
    allocated_stack_size: int = attr.ib(default=0)
    is_leaf: bool = attr.ib(default=True)
    is_variadic: bool = attr.ib(default=False)
    uses_framepointer: bool = attr.ib(default=False)
    local_vars_region_bottom: int = attr.ib(default=0)
    return_addr_location: int = attr.ib(default=0)
    callee_save_reg_locations: Dict[Register, int] = attr.ib(factory=dict)
    unique_type_map: Dict[Any, "Type"] = attr.ib(factory=dict)
    local_vars: List["LocalVar"] = attr.ib(factory=list)
    temp_vars: List["EvalOnceStmt"] = attr.ib(factory=list)
    phi_vars: List["PhiExpr"] = attr.ib(factory=list)
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
        if self.callee_save_reg_locations:
            subroutine_arg_top = min(self.callee_save_reg_locations.values())
            assert self.return_addr_location > subroutine_arg_top
        else:
            subroutine_arg_top = self.return_addr_location

        return location < subroutine_arg_top

    def in_local_var_region(self, location: int) -> bool:
        return self.local_vars_region_bottom <= location < self.allocated_stack_size

    def location_above_stack(self, location: int) -> bool:
        return location >= self.allocated_stack_size

    def set_param_name(self, offset: int, name: str) -> None:
        self.param_names[offset] = name

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
        ret = PassedInArg(
            real_location,
            copied=True,
            stack_info=self,
            type=self.unique_type_for("arg", real_location),
        )
        if real_location == location - 3:
            return as_type(ret, Type.of_size(8), True), ret
        if real_location == location - 2:
            return as_type(ret, Type.of_size(16), True), ret
        return ret, ret

    def record_struct_access(self, ptr: "Expression", location: int) -> None:
        if location:
            self.nonzero_accesses.add(unwrap_deep(ptr))

    def has_nonzero_access(self, ptr: "Expression") -> bool:
        return unwrap_deep(ptr) in self.nonzero_accesses

    def unique_type_for(self, category: str, key: Any) -> "Type":
        key = (category, key)
        if key not in self.unique_type_map:
            self.unique_type_map[key] = Type.any()
        return self.unique_type_map[key]

    def global_symbol(self, sym: AsmGlobalSymbol) -> "GlobalSymbol":
        return GlobalSymbol(
            symbol_name=sym.symbol_name,
            type=self.unique_type_for("symbol", sym.symbol_name),
        )

    def saved_reg_symbol(self, reg_name: str) -> "GlobalSymbol":
        sym_name = "saved_reg_" + reg_name
        return self.global_symbol(AsmGlobalSymbol(sym_name))

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
        if self.in_local_var_region(location):
            return LocalVar(location, type=self.unique_type_for("stack", location))
        elif self.location_above_stack(location):
            ret, arg = self.get_argument(location - self.allocated_stack_size)
            if not store:
                self.add_argument(arg)
            return ret
        elif self.in_subroutine_arg_region(location):
            return SubroutineArg(location, type=Type.any())
        else:
            # Some annoying bookkeeping instruction. To avoid
            # further special-casing, just return whatever - it won't matter.
            return LocalVar(location, type=Type.any())

    def is_stack_reg(self, reg: Register) -> bool:
        if reg.register_name == "sp":
            return True
        if reg.register_name == "fp":
            return self.uses_framepointer
        return False

    def __str__(self) -> str:
        return "\n".join(
            [
                f"Stack info for function {self.function.name}:",
                f"Allocated stack size: {self.allocated_stack_size}",
                f"Leaf? {self.is_leaf}",
                f"Bottom of local vars region: {self.local_vars_region_bottom}",
                f"Location of return addr: {self.return_addr_location}",
                f"Locations of callee save registers: {self.callee_save_reg_locations}",
            ]
        )


def get_stack_info(
    function: Function, rodata: Rodata, start_node: Node, typemap: Optional[TypeMap]
) -> StackInfo:
    info = StackInfo(function, rodata, typemap)

    # The goal here is to pick out special instructions that provide information
    # about this function's stack setup.
    for inst in start_node.block.instructions:
        if not inst.args:
            continue

        destination = typing.cast(Register, inst.args[0])

        if inst.mnemonic == "addiu" and destination.register_name == "sp":
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
        elif inst.mnemonic == "sw" and destination.register_name == "ra":
            # Saving the return address on the stack.
            assert isinstance(inst.args[1], AsmAddressMode)
            assert inst.args[1].rhs.register_name == "sp"
            info.is_leaf = False
            if inst.args[1].lhs:
                assert isinstance(inst.args[1].lhs, AsmLiteral)
                info.return_addr_location = inst.args[1].lhs.signed_value()
            else:
                # Note that this should only happen in the rare case that
                # this function only calls subroutines with no arguments.
                info.return_addr_location = 0
        elif (
            inst.mnemonic in ["sw", "swc1", "sdc1"]
            and destination.is_callee_save()
            and isinstance(inst.args[1], AsmAddressMode)
            and inst.args[1].rhs.register_name == "sp"
        ):
            # Initial saving of callee-save register onto the stack.
            assert isinstance(inst.args[1].rhs, Register)
            if inst.args[1].lhs:
                assert isinstance(inst.args[1].lhs, AsmLiteral)
                info.callee_save_reg_locations[destination] = inst.args[
                    1
                ].lhs.signed_value()
            else:
                info.callee_save_reg_locations[destination] = 0

    # Find the region that contains local variables.
    if info.is_leaf and info.callee_save_reg_locations:
        # In a leaf with callee-save registers, the local variables
        # lie directly above those registers.
        info.local_vars_region_bottom = max(info.callee_save_reg_locations.values()) + 4
    elif info.is_leaf:
        # In a leaf without callee-save registers, the local variables
        # lie directly at the bottom of the stack.
        info.local_vars_region_bottom = 0
    else:
        # In a non-leaf, the local variables lie above the location of the
        # return address.
        info.local_vars_region_bottom = info.return_addr_location + 4

    # Done.
    return info


def format_hex(val: int) -> str:
    return format(val, "x").upper()


def escape_char(ch: str) -> str:
    table = {
        "\0": "\\0",
        "\b": "\\b",
        "\f": "\\f",
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\v": "\\v",
        "\\": "\\\\",
        '"': '\\"',
    }
    if ch in table:
        return table[ch]
    if ord(ch) < 0x20 or ord(ch) in [0xFF, 0x7F]:
        return "\\x{:02x}".format(ord(ch))
    return ch


@attr.s(eq=False)
class Var:
    stack_info: StackInfo = attr.ib(repr=False)
    prefix: str = attr.ib()
    num_usages: int = attr.ib(default=0)
    name: Optional[str] = attr.ib(default=None)

    def __str__(self) -> str:
        if self.name is None:
            self.name = self.stack_info.temp_var(self.prefix)
        return self.name


@attr.s(frozen=True, eq=False)
class ErrorExpr:
    desc: Optional[str] = attr.ib(default=None)
    type: Type = attr.ib(factory=Type.any)

    def dependencies(self) -> List["Expression"]:
        return []

    def negated(self) -> "Condition":
        return self

    def __str__(self) -> str:
        if self.desc is not None:
            return f"ERROR({self.desc})"
        return "ERROR"


@attr.s(frozen=True, eq=False)
class SecondF64Half:
    type: Type = attr.ib(factory=Type.any)

    def dependencies(self) -> List["Expression"]:
        return []

    def __str__(self) -> str:
        return "(second half of f64)"


@attr.s(frozen=True, eq=False)
class BinaryOp:
    left: "Expression" = attr.ib()
    op: str = attr.ib()
    right: "Expression" = attr.ib()
    type: Type = attr.ib()
    floating: bool = attr.ib(default=False)

    @staticmethod
    def int(left: "Expression", op: str, right: "Expression") -> "BinaryOp":
        return BinaryOp(
            left=as_intish(left), op=op, right=as_intish(right), type=Type.intish()
        )

    @staticmethod
    def intptr(left: "Expression", op: str, right: "Expression") -> "BinaryOp":
        return BinaryOp(
            left=as_intptr(left), op=op, right=as_intptr(right), type=Type.intptr()
        )

    @staticmethod
    def icmp(left: "Expression", op: str, right: "Expression") -> "BinaryOp":
        return BinaryOp(
            left=as_intptr(left), op=op, right=as_intptr(right), type=Type.bool()
        )

    @staticmethod
    def scmp(left: "Expression", op: str, right: "Expression") -> "BinaryOp":
        return BinaryOp(
            left=as_s32(left, silent=True),
            op=op,
            right=as_s32(right, silent=True),
            type=Type.bool(),
        )

    @staticmethod
    def ucmp(left: "Expression", op: str, right: "Expression") -> "BinaryOp":
        return BinaryOp(left=as_u32(left), op=op, right=as_u32(right), type=Type.bool())

    @staticmethod
    def fcmp(left: "Expression", op: str, right: "Expression") -> "BinaryOp":
        return BinaryOp(
            left=as_f32(left),
            op=op,
            right=as_f32(right),
            type=Type.bool(),
            floating=True,
        )

    @staticmethod
    def dcmp(left: "Expression", op: str, right: "Expression") -> "BinaryOp":
        return BinaryOp(
            left=as_f64(left),
            op=op,
            right=as_f64(right),
            type=Type.bool(),
            floating=True,
        )

    @staticmethod
    def s32(left: "Expression", op: str, right: "Expression") -> "BinaryOp":
        return BinaryOp(left=as_s32(left), op=op, right=as_s32(right), type=Type.s32())

    @staticmethod
    def u32(left: "Expression", op: str, right: "Expression") -> "BinaryOp":
        return BinaryOp(left=as_u32(left), op=op, right=as_u32(right), type=Type.u32())

    @staticmethod
    def f32(left: "Expression", op: str, right: "Expression") -> "BinaryOp":
        return BinaryOp(
            left=as_f32(left),
            op=op,
            right=as_f32(right),
            type=Type.f32(),
            floating=True,
        )

    @staticmethod
    def f64(left: "Expression", op: str, right: "Expression") -> "BinaryOp":
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
        assert self.is_boolean()
        if self.floating and self.op in ["<", ">", "<=", ">="]:
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

    def dependencies(self) -> List["Expression"]:
        return [self.left, self.right]

    def __str__(self) -> str:
        if (
            self.op == "+"
            and not self.floating
            and isinstance(self.right, Literal)
            and self.right.value < 0
        ):
            neg = Literal(value=-self.right.value, type=self.right.type)
            sub = BinaryOp(op="-", left=self.left, right=neg, type=self.type)
            return str(sub)
        return f"({self.left} {self.op} {self.right})"


@attr.s(frozen=True, eq=False)
class UnaryOp:
    op: str = attr.ib()
    expr: "Expression" = attr.ib()
    type: Type = attr.ib()

    def dependencies(self) -> List["Expression"]:
        return [self.expr]

    def negated(self) -> "Condition":
        if self.op == "!" and isinstance(self.expr, (UnaryOp, BinaryOp)):
            return self.expr
        return UnaryOp("!", self, type=Type.bool())

    def __str__(self) -> str:
        return f"{self.op}{self.expr}"


@attr.s(frozen=True, eq=False)
class CommaConditionExpr:
    statements: List["Statement"] = attr.ib()
    condition: "Condition" = attr.ib()
    type: Type = Type.bool()

    def dependencies(self) -> List["Expression"]:
        assert False, "CommaConditionExpr should not be used within translate.py"
        return []

    def negated(self) -> "Condition":
        return CommaConditionExpr(self.statements, self.condition.negated())

    def __str__(self) -> str:
        comma_joined = ", ".join(str(stmt).rstrip(";") for stmt in self.statements)
        return f"({comma_joined}, {self.condition})"


@attr.s(frozen=True, eq=False)
class Cast:
    expr: "Expression" = attr.ib()
    type: Type = attr.ib()
    reinterpret: bool = attr.ib(default=False)
    silent: bool = attr.ib(default=True)

    def dependencies(self) -> List["Expression"]:
        return [self.expr]

    def use(self) -> None:
        # Try to unify, to make stringification output better.
        self.expr.type.unify(self.type)

    def __str__(self) -> str:
        if self.reinterpret and self.expr.type.is_float() != self.type.is_float():
            # This shouldn't happen, but mark it in the output if it does.
            return f"(bitwise {self.type}) {self.expr}"
        if self.reinterpret and (
            self.silent
            or (is_type_obvious(self.expr) and self.expr.type.unify(self.type))
        ):
            return str(self.expr)
        return f"({self.type}) {self.expr}"


@attr.s(frozen=True, eq=False)
class FuncCall:
    function: "Expression" = attr.ib()
    args: List["Expression"] = attr.ib()
    type: Type = attr.ib()

    def dependencies(self) -> List["Expression"]:
        return self.args + [self.function]

    def __str__(self) -> str:
        args = ", ".join(stringify_expr(arg) for arg in self.args)
        return f"{self.function}({args})"


@attr.s(frozen=True, eq=True)
class LocalVar:
    value: int = attr.ib()
    type: Type = attr.ib(eq=False)

    def dependencies(self) -> List["Expression"]:
        return []

    def __str__(self) -> str:
        return f"sp{format_hex(self.value)}"


@attr.s(frozen=True, eq=True)
class PassedInArg:
    value: int = attr.ib()
    copied: bool = attr.ib(eq=False)
    stack_info: StackInfo = attr.ib(eq=False, repr=False)
    type: Type = attr.ib(eq=False)

    def dependencies(self) -> List["Expression"]:
        return []

    def __str__(self) -> str:
        assert self.value % 4 == 0
        name = self.stack_info.get_param_name(self.value)
        return name or f"arg{format_hex(self.value // 4)}"


@attr.s(frozen=True, eq=True)
class SubroutineArg:
    value: int = attr.ib()
    type: Type = attr.ib(eq=False)

    def dependencies(self) -> List["Expression"]:
        return []

    def __str__(self) -> str:
        return f"subroutine_arg{format_hex(self.value // 4)}"


@attr.s(frozen=True, eq=True)
class StructAccess:
    # Represents struct_var->offset.
    # This has eq=True since it represents a live expression and not
    # an access at a certain point in time -- this sometimes helps get rid of phi nodes.
    # Really it should represent the latter, but making that so is hard.
    struct_var: "Expression" = attr.ib()
    offset: int = attr.ib()
    target_size: Optional[int] = attr.ib()
    field_name: Optional[str] = attr.ib(eq=False)
    stack_info: StackInfo = attr.ib(eq=False, repr=False)
    type: Type = attr.ib(eq=False)

    def dependencies(self) -> List["Expression"]:
        return [self.struct_var]

    def __str__(self) -> str:
        var = late_unwrap(self.struct_var)
        has_nonzero_access = self.stack_info.has_nonzero_access(var)

        field_name: Optional[str] = None
        if self.field_name is not None:
            field_name = self.field_name
        elif self.stack_info.typemap:
            # If we didn't have a type at the struct access was constructed,
            # but now we do, compute field name late.
            field_name = get_field(
                var.type,
                self.offset,
                self.stack_info.typemap,
                target_size=self.target_size,
            )[0]

        if field_name:
            has_nonzero_access = True
        else:
            field_name = "unk" + format_hex(self.offset)

        if isinstance(var, AddressOf):
            if self.offset == 0 and not has_nonzero_access:
                return f"{var.expr}"
            else:
                return f"{parenthesize_for_struct_access(var.expr)}.{field_name}"
        else:
            if self.offset == 0 and not has_nonzero_access:
                return f"*{var}"
            else:
                return f"{parenthesize_for_struct_access(var)}->{field_name}"


@attr.s(frozen=True, eq=True)
class ArrayAccess:
    # Represents ptr[index]. eq=True for symmetry with StructAccess.
    ptr: "Expression" = attr.ib()
    index: "Expression" = attr.ib()
    type: Type = attr.ib(eq=False)

    def dependencies(self) -> List["Expression"]:
        return [self.ptr, self.index]

    def __str__(self) -> str:
        return f"{parenthesize_for_struct_access(self.ptr)}[{self.index}]"


@attr.s(frozen=True, eq=True)
class GlobalSymbol:
    symbol_name: str = attr.ib()
    type: Type = attr.ib(eq=False)

    def dependencies(self) -> List["Expression"]:
        return []

    def __str__(self) -> str:
        return self.symbol_name


@attr.s(frozen=True, eq=True)
class Literal:
    value: int = attr.ib()
    type: Type = attr.ib(eq=False, factory=Type.any)

    def dependencies(self) -> List["Expression"]:
        return []

    def __str__(self) -> str:
        if self.type.is_float():
            if self.type.get_size_bits() == 32:
                return format_f32_imm(self.value) + "f"
            else:
                return format_f64_imm(self.value)
        prefix = ""
        if self.type.is_pointer():
            if self.value == 0:
                return "NULL"
            else:
                prefix = "(void *)"
        elif self.type.get_size_bits() == 8:
            prefix = "(u8)"
        elif self.type.get_size_bits() == 16:
            prefix = "(u16)"
        suffix = "U" if self.type.is_unsigned() else ""
        mid = (
            str(self.value)
            if abs(self.value) < 10
            else hex(self.value).upper().replace("X", "x")
        )
        return prefix + mid + suffix


@attr.s(frozen=True)
class StringLiteral:
    data: bytes = attr.ib()
    type: Type = attr.ib()

    def dependencies(self) -> List["Expression"]:
        return []

    def __str__(self) -> str:
        has_trailing_null = False
        strdata: str
        try:
            strdata = self.data.decode("utf-8")
        except UnicodeDecodeError:
            strdata = self.data.decode("latin1")
        while strdata and strdata[-1] == "\0":
            strdata = strdata[:-1]
            has_trailing_null = True
        ret = '"' + "".join(map(escape_char, strdata)) + '"'
        if not has_trailing_null:
            ret += " /* not null-terminated */"
        return ret


@attr.s(frozen=True, eq=True)
class AddressOf:
    expr: "Expression" = attr.ib()
    type: Type = attr.ib(eq=False, factory=Type.ptr)

    def dependencies(self) -> List["Expression"]:
        return [self.expr]

    def __str__(self) -> str:
        return f"&{self.expr}"


@attr.s(frozen=True)
class AddressMode:
    offset: int = attr.ib()
    rhs: Register = attr.ib()

    def __str__(self) -> str:
        if self.offset:
            return f"{self.offset}({self.rhs})"
        else:
            return f"({self.rhs})"


@attr.s(frozen=False, eq=False)
class EvalOnceExpr:
    wrapped_expr: "Expression" = attr.ib()
    var: Var = attr.ib()
    type: Type = attr.ib()

    # Mutable state:

    # True for function calls/errors, and may be set to true dynamically by the hack
    # in RegInfo.__getitem__ that deals with code that does not understand ForceVarExpr.
    # This is a mess, sorry. :(
    always_emit: bool = attr.ib()

    # True if this EvalOnceExpr should be totally transparent and not emit a variable,
    # It may dynamically change from true to false due to forced emissions.
    # Initially, it is based on is_trivial_expression.
    trivial: bool = attr.ib()

    # The number of expressions that depend on this EvalOnceExpr; we emit a variable
    # if this is > 1.
    num_usages: int = attr.ib(default=0)

    def dependencies(self) -> List["Expression"]:
        return [self.wrapped_expr]

    def use(self) -> None:
        self.num_usages += 1
        if self.trivial or (self.num_usages == 1 and not self.always_emit):
            mark_used(self.wrapped_expr)

    def need_decl(self) -> bool:
        return self.num_usages > 1 and not self.trivial

    def __str__(self) -> str:
        if not self.need_decl():
            return str(self.wrapped_expr)
        else:
            return str(self.var)


@attr.s(eq=False)
class ForceVarExpr:
    wrapped_expr: EvalOnceExpr = attr.ib()
    type: Type = attr.ib()

    def dependencies(self) -> List["Expression"]:
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
        self.wrapped_expr.use()
        self.wrapped_expr.use()

    def __str__(self) -> str:
        return str(self.wrapped_expr)


@attr.s(frozen=False, eq=False)
class PhiExpr:
    reg: Register = attr.ib()
    node: Node = attr.ib()
    type: Type = attr.ib()
    used_phis: List["PhiExpr"] = attr.ib()
    name: Optional[str] = attr.ib(default=None)
    num_usages: int = attr.ib(default=0)
    replacement_expr: Optional["Expression"] = attr.ib(default=None)
    used_by: Optional["PhiExpr"] = attr.ib(default=None)

    def dependencies(self) -> List["Expression"]:
        return []

    def get_var_name(self) -> str:
        return self.name or f"unnamed-phi({self.reg.register_name})"

    def use(self, from_phi: Optional["PhiExpr"] = None) -> None:
        if self.num_usages == 0:
            self.used_phis.append(self)
        self.num_usages += 1
        self.used_by = from_phi

    def propagates_to(self) -> "PhiExpr":
        if self.num_usages != 1 or self.used_by is None:
            return self
        return self.used_by.propagates_to()

    def __str__(self) -> str:
        if self.replacement_expr:
            return str(self.replacement_expr)
        return self.get_var_name()


@attr.s
class EvalOnceStmt:
    expr: EvalOnceExpr = attr.ib()

    def need_decl(self) -> bool:
        return self.expr.need_decl()

    def should_write(self) -> bool:
        if self.expr.always_emit:
            return self.expr.num_usages != 1
        else:
            return self.need_decl()

    def __str__(self) -> str:
        val_str = stringify_expr(self.expr.wrapped_expr)
        if self.expr.always_emit and self.expr.num_usages == 0:
            return f"{val_str};"
        return f"{self.expr.var} = {val_str};"


@attr.s
class SetPhiStmt:
    phi: PhiExpr = attr.ib()
    expr: "Expression" = attr.ib()

    def should_write(self) -> bool:
        expr = self.expr
        if isinstance(expr, PhiExpr) and expr.propagates_to() != expr:
            assert expr.propagates_to() == self.phi.propagates_to()
            return False
        return True

    def __str__(self) -> str:
        val_str = stringify_expr(self.expr)
        return f"{self.phi.propagates_to().get_var_name()} = {val_str};"


@attr.s
class ExprStmt:
    expr: "Expression" = attr.ib()

    def should_write(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"{stringify_expr(self.expr)};"


@attr.s
class StoreStmt:
    source: "Expression" = attr.ib()
    dest: "Expression" = attr.ib()

    def should_write(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"{self.dest} = {stringify_expr(self.source)};"


@attr.s
class CommentStmt:
    contents: str = attr.ib()

    def should_write(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"// {self.contents}"


Expression = Union[
    BinaryOp,
    UnaryOp,
    CommaConditionExpr,
    Cast,
    FuncCall,
    GlobalSymbol,
    Literal,
    StringLiteral,
    AddressOf,
    LocalVar,
    PassedInArg,
    StructAccess,
    ArrayAccess,
    SubroutineArg,
    EvalOnceExpr,
    ForceVarExpr,
    PhiExpr,
    ErrorExpr,
    SecondF64Half,
]

Condition = Union[BinaryOp, UnaryOp, ErrorExpr, CommaConditionExpr]

Statement = Union[StoreStmt, EvalOnceStmt, SetPhiStmt, ExprStmt, CommentStmt]


@attr.s
class RegInfo:
    contents: Dict[Register, Expression] = attr.ib()
    stack_info: StackInfo = attr.ib(repr=False)

    def __getitem__(self, key: Register) -> Expression:
        if key == Register("zero"):
            return Literal(0)
        ret = self.get_raw(key)
        if ret is None:
            raise DecompFailure(f"Read from unset register {key}")
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
            ret.always_emit = True
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

    def clear_caller_save_regs(self) -> None:
        for reg in TEMP_REGS:
            assert reg != Register("zero")
            if reg in self.contents:
                del self.contents[reg]

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

    def reg_ref(self, index: int) -> Register:
        ret = self.raw_args[index]
        assert isinstance(ret, Register)
        return ret

    def reg(self, index: int) -> Expression:
        return self.regs[self.reg_ref(index)]

    def dreg(self, index: int) -> Expression:
        """Extract a double from a register. This may involve reading both the
        mentioned register and the next."""
        reg = self.reg_ref(index)
        assert reg.is_float()
        ret = self.regs[reg]
        if not isinstance(ret, Literal) or ret.type.get_size_bits() == 64:
            return ret
        reg_num = int(reg.register_name[1:])
        assert reg_num % 2 == 0
        other = self.regs[Register(f"f{reg_num+1}")]
        assert isinstance(other, Literal) and other.type.get_size_bits() != 64
        value = ret.value | (other.value << 32)
        return Literal(value, type=Type.f64())

    def address_of_gsym(self, sym: GlobalSymbol) -> Expression:
        ent = self.stack_info.rodata.values.get(sym.symbol_name)
        if ent and ent.is_string and ent.data and isinstance(ent.data[0], bytes):
            return StringLiteral(ent.data[0], type=Type.ptr(Type.s8()))
        type = Type.ptr()
        typemap = self.stack_info.typemap
        if typemap:
            ctype = typemap.var_types.get(sym.symbol_name)
            if ctype:
                type, is_ptr = ptr_type_from_ctype(ctype, typemap)
                if is_ptr:
                    return as_type(sym, type, True)
        return AddressOf(sym, type=type)

    def imm(self, index: int) -> Expression:
        arg = strip_macros(self.raw_args[index])
        ret = literal_expr(arg, self.stack_info)
        if isinstance(ret, GlobalSymbol):
            return self.address_of_gsym(ret)
        return ret

    def hi_imm(self, index: int) -> Expression:
        arg = self.raw_args[index]
        assert isinstance(arg, Macro) and arg.macro_name == "hi"
        ret = literal_expr(arg.argument, self.stack_info)
        if isinstance(ret, GlobalSymbol):
            return self.address_of_gsym(ret)
        return ret

    def memory_ref(self, index: int) -> AddressMode:
        ret = strip_macros(self.raw_args[index])
        assert isinstance(ret, AsmAddressMode)
        if ret.lhs is None:
            return AddressMode(offset=0, rhs=ret.rhs)
        assert isinstance(ret.lhs, AsmLiteral)  # macros were removed
        return AddressMode(offset=ret.lhs.signed_value(), rhs=ret.rhs)

    def count(self) -> int:
        return len(self.raw_args)


def deref(
    arg: AddressMode,
    regs: RegInfo,
    stack_info: StackInfo,
    *,
    size: int,
    store: bool = False,
) -> Expression:
    offset = arg.offset
    if stack_info.is_stack_reg(arg.rhs):
        return stack_info.get_stack_var(offset, store=store)

    # Struct member is being dereferenced.
    var = regs[arg.rhs]

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
                and addend.value % 2 ** 16 == 0
                and addend.value < 0x1000000
            ):
                offset += addend.value
                var = base
                break

    var.type.unify(Type.ptr())
    stack_info.record_struct_access(var, offset)
    field_name: Optional[str] = None
    type: Type = stack_info.unique_type_for("struct", (var, offset))

    # Struct access with type information.
    typemap = stack_info.typemap
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
    target = get_pointer_target(var.type, typemap)
    if field_name is None and target is not None:
        sub_size, sub_type = target
        if sub_size == size and offset % size == 0:
            if offset != 0:
                index = Literal(value=offset // size, type=Type.s32())
                return ArrayAccess(var, index, type=sub_type)
            type = sub_type

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
    # TODO: Some of this logic is sketchy, saying that it's fine to repeat e.g.
    # reads even though there might have been e.g. sets or function calls in
    # between. It should really take into account what has changed since the
    # expression was created and when it's used. For now, though, we make this
    # naive guess at the creation. (Another signal we could potentially use is
    # whether the expression is stored in a callee-save register.)
    if expr is None or isinstance(
        expr,
        (
            EvalOnceExpr,
            ForceVarExpr,
            Literal,
            GlobalSymbol,
            LocalVar,
            PassedInArg,
            SubroutineArg,
        ),
    ):
        return True
    if isinstance(expr, AddressOf):
        return is_trivial_expression(expr.expr)
    if isinstance(expr, StructAccess):
        return is_trivial_expression(expr.struct_var)
    if isinstance(expr, ArrayAccess):
        return is_trivial_expression(expr.ptr) and is_trivial_expression(expr.index)
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
    if isinstance(expr, EvalOnceExpr) and expr.num_usages <= 1:
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


def stringify_expr(expr: Expression) -> str:
    """
    Stringify an expression, stripping unnecessary parentheses around it.
    """
    ret = str(expr)
    if ret.startswith("(") and balanced_parentheses(ret[1:-1]):
        return ret[1:-1]
    return ret


def parenthesize_for_struct_access(expr: Expression) -> str:
    # Nested dereferences may need to be parenthesized. All other
    # expressions will already have adequate parentheses added to them.
    # (Except Cast's, TODO...)
    s = str(expr)
    return f"({s})" if s.startswith("*") else s


def mark_used(expr: Expression) -> None:
    if isinstance(expr, (PhiExpr, EvalOnceExpr, ForceVarExpr, Cast)):
        expr.use()
    if not isinstance(expr, (EvalOnceExpr, ForceVarExpr)):
        for sub_expr in expr.dependencies():
            mark_used(sub_expr)


def uses_expr(expr: Expression, sub_expr: Expression) -> bool:
    if expr == sub_expr:
        return True
    for e in expr.dependencies():
        if uses_expr(e, sub_expr):
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
    return expr


def early_unwrap(expr: Expression) -> Expression:
    """
    Unwrap EvalOnceExpr's, even past variable boundaries.

    This is fine to use even while code is being generated, but disrespects decisions
    to use a temp for a value, so use with care.

    TODO: unwrap ForceVarExpr as well when safe, pushing the forces down into the
    expression tree.
    """
    if isinstance(expr, EvalOnceExpr) and not expr.always_emit:
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
        return stack_info.global_symbol(arg)
    if isinstance(arg, AsmLiteral):
        return Literal(arg.value)
    assert isinstance(arg, BinOp), f"argument {arg} must be a literal"
    lhs = literal_expr(arg.lhs, stack_info)
    rhs = literal_expr(arg.rhs, stack_info)
    return BinaryOp.int(left=lhs, op=arg.op, right=rhs)


def fn_op(fn_name: str, args: List[Expression], type: Type) -> FuncCall:
    return FuncCall(
        function=GlobalSymbol(symbol_name=fn_name, type=Type.any()),
        args=args,
        type=type,
    )


def load_upper(args: InstrArgs) -> Expression:
    if not isinstance(args.raw_args[1], Macro):
        assert not isinstance(
            args.raw_args[1], Literal
        ), "normalize_instruction should convert lui <literal> to li"
        raise DecompFailure("lui argument must be a literal or %hi macro")
    return args.hi_imm(1)


def handle_ori(args: InstrArgs) -> Expression:
    imm = args.imm(2)
    r = args.reg(1)
    if isinstance(r, Literal) and isinstance(imm, Literal):
        return Literal(value=(r.value | imm.value))
    # Regular bitwise OR.
    return BinaryOp.int(left=r, op="|", right=imm)


def handle_addi(args: InstrArgs) -> Expression:
    stack_info = args.stack_info
    source_reg = args.reg_ref(1)
    source = args.reg(1)
    imm = args.imm(2)
    if imm == Literal(0):
        # addiu $reg1, $reg2, 0 is a move
        # (this happens when replacing %lo(...) by 0)
        return source
    elif stack_info.is_stack_reg(source_reg):
        # Adding to sp, i.e. passing an address.
        assert isinstance(imm, Literal)
        if stack_info.is_stack_reg(args.reg_ref(0)):
            # Changing sp. Just ignore that.
            return source
        # Keep track of all local variables that we take addresses of.
        var = stack_info.get_stack_var(imm.value, store=False)
        if isinstance(var, LocalVar):
            stack_info.add_local_var(var)
        return AddressOf(var, type=Type.ptr(var.type))
    elif source.type.is_pointer():
        # Pointer addition (this may miss some pointers that get detected later;
        # unfortunately that's hard to do anything about with mips_to_c's single-pass
        # architecture.
        typemap = stack_info.typemap
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
            target = get_pointer_target(source.type, typemap)
            if target and imm.value % target[0] == 0:
                # Pointer addition.
                return BinaryOp(
                    left=source, op="+", right=as_intish(imm), type=source.type
                )
        return BinaryOp(left=source, op="+", right=as_intish(imm), type=Type.ptr())
    else:
        # Regular binary addition.
        return BinaryOp.intptr(left=source, op="+", right=imm)


def handle_load(args: InstrArgs, type: Type) -> Expression:
    # For now, make the cast silent so that output doesn't become cluttered.
    # Though really, it would be great to expose the load types somehow...
    size = type.get_size_bits() // 8
    expr = deref(args.memory_ref(1), args.regs, args.stack_info, size=size)

    # Detect rodata constants
    if isinstance(expr, StructAccess):
        target = early_unwrap(expr.struct_var)
        if (
            isinstance(target, AddressOf)
            and isinstance(target.expr, GlobalSymbol)
            and type.is_float()
        ):
            sym_name = target.expr.symbol_name
            ent = args.stack_info.rodata.values.get(sym_name)
            if (
                ent
                and ent.data
                and isinstance(ent.data[0], bytes)
                and len(ent.data[0]) >= size
            ):
                data = ent.data[0][:size]
                val: int
                if size == 4:
                    (val,) = struct.unpack(">I", data)
                else:
                    (val,) = struct.unpack(">Q", data)
                return Literal(value=val, type=type)

    return as_type(expr, type, silent=True)


def make_store(args: InstrArgs, type: Type) -> Optional[StoreStmt]:
    size = type.get_size_bits() // 8
    stack_info = args.stack_info
    source_reg = args.reg_ref(0)
    source_val = args.reg(0)
    source_raw = args.regs.get_raw(source_reg)
    target = args.memory_ref(1)
    if (
        stack_info.is_stack_reg(target.rhs)
        and source_raw is not None
        and stack_info.should_save(source_raw, target.offset)
    ):
        # Elide register preserval.
        return None
    dest = deref(target, args.regs, stack_info, size=size, store=True)
    dest.type.unify(type)
    silent = stack_info.is_stack_reg(target.rhs)
    return StoreStmt(source=as_type(source_val, type, silent=silent), dest=dest)


def format_f32_imm(num: int) -> str:
    (num,) = struct.unpack(">f", struct.pack(">I", num & (2 ** 32 - 1)))
    return str(num)


def format_f64_imm(num: int) -> str:
    (num,) = struct.unpack(">d", struct.pack(">Q", num & (2 ** 64 - 1)))
    return str(num)


def fold_mul_chains(expr: Expression) -> Expression:
    def fold(expr: Expression, toplevel: bool) -> Tuple[Expression, int]:
        if isinstance(expr, BinaryOp):
            lbase, lnum = fold(expr.left, False)
            rbase, rnum = fold(expr.right, False)
            if expr.op == "<<" and isinstance(expr.right, Literal):
                # Left-shifts by small numbers are easier to understand if
                # written as multiplications (they compile to the same thing).
                if toplevel and lnum == 1 and not (1 <= expr.right.value <= 4):
                    return (expr, 1)
                return (lbase, lnum << expr.right.value)
            if expr.op == "*" and isinstance(expr.right, Literal):
                return (lbase, lnum * expr.right.value)
            if expr.op == "+" and lbase == rbase:
                return (lbase, lnum + rnum)
            if expr.op == "-" and lbase == rbase:
                return (lbase, lnum - rnum)
        if isinstance(expr, UnaryOp) and not toplevel:
            base, num = fold(expr.expr, False)
            return (base, -num)
        if isinstance(expr, EvalOnceExpr):
            base, num = fold(expr.wrapped_expr, False)
            if num != 1 and is_trivial_expression(base):
                return (base, num)
        return (expr, 1)

    base, num = fold(expr, True)
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

    target = get_pointer_target(base.type, typemap)
    if target is None:
        return None

    if target[0] == scale:
        # base[index]
        target_type = target[1]
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
        target_type = type_from_ctype(elem_type, typemap)

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


def handle_add(lhs: Expression, rhs: Expression, stack_info: StackInfo) -> Expression:
    type = Type.intptr()
    if lhs.type.is_pointer():
        type = Type.ptr()
    elif rhs.type.is_pointer():
        type = Type.ptr()
    expr = BinaryOp(left=as_intptr(lhs), op="+", right=as_intptr(rhs), type=type)
    folded_expr = fold_mul_chains(expr)
    if folded_expr is not expr:
        return folded_expr
    typemap = stack_info.typemap
    if typemap is not None:
        array_expr = array_access_from_add(
            expr, 0, stack_info, typemap, target_size=None, ptr=True
        )
        if array_expr is not None:
            return array_expr
    return expr


def strip_macros(arg: Argument) -> Argument:
    """Replace %lo(...) by 0, and assert that there are no %hi(...). We assume
    that %hi's only ever occur in lui, where we expand them to an entire value,
    and not just the upper part. This ought to preserve semantics in all
    reasonable cases."""
    if isinstance(arg, Macro):
        assert arg.macro_name == "lo"
        return AsmLiteral(0)
    elif isinstance(arg, AsmAddressMode) and isinstance(arg.lhs, Macro):
        assert arg.lhs.macro_name == "lo"
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
    fn: CFunction, typemap: TypeMap, *, for_call: bool
) -> Tuple[List[AbiStackSlot], List[Register]]:
    """Compute stack positions/registers used by a function according to the o32 ABI,
    based on C type information. Additionally computes a list of registers that might
    contain arguments, if the function is a varargs function. (Additional varargs
    arguments may be passed on the stack; we could compute the offset at which that
    would start but right now don't care -- we just slurp up everything.)"""
    assert fn.params is not None, "checked by caller"
    offset = 0
    only_floats = True
    slots: List[AbiStackSlot] = []
    possible: List[Register] = []
    if fn.ret_type is not None and is_struct_type(fn.ret_type, typemap):
        # The ABI for struct returns is to pass a pointer to where it should be written
        # as the first argument.
        slots.append(
            AbiStackSlot(
                offset=0, reg=Register("a0"), name="__return__", type=Type.ptr()
            )
        )
        offset = 4
        only_floats = False

    for ind, param in enumerate(fn.params):
        size, align = function_arg_size_align(param.type, typemap)
        size = max(size, 4)
        primitive_list = get_primitive_list(param.type, typemap)
        only_floats = only_floats and (primitive_list in [["float"], ["double"]])
        offset = (offset + align - 1) & -align
        name = param.name
        if ind < 2 and only_floats:
            reg = Register("f12" if ind == 0 else "f14")
            is_double = primitive_list == ["double"]
            type = Type.f64() if is_double else Type.f32()
            slots.append(AbiStackSlot(offset=offset, reg=reg, name=name, type=type))
            if is_double and not for_call:
                name2 = f"{name}_lo" if name else None
                reg2 = Register("f13" if ind == 0 else "f15")
                slots.append(
                    AbiStackSlot(
                        offset=offset + 4, reg=reg2, name=name2, type=Type.any()
                    )
                )
        else:
            for i in range(offset // 4, min((offset + size) // 4, 4)):
                unk_offset = 4 * i - offset
                name2 = f"{name}_unk{unk_offset:X}" if name and unk_offset else name
                reg2 = Register(f"a{i}")
                type2 = type_from_ctype(param.type, typemap)
                slots.append(
                    AbiStackSlot(offset=4 * i, reg=reg2, name=name2, type=type2)
                )
        offset += size

    if fn.is_variadic:
        for i in range(offset // 4, 4):
            possible.append(Register(f"a{i}"))

    return slots, possible


InstrSet = Set[str]
InstrMap = Dict[str, Callable[[InstrArgs], Expression]]
CmpInstrMap = Dict[str, Callable[[InstrArgs], BinaryOp]]
StoreInstrMap = Dict[str, Callable[[InstrArgs], Optional[StoreStmt]]]
MaybeInstrMap = Dict[str, Callable[[InstrArgs], Optional[Expression]]]
PairInstrMap = Dict[
    str, Callable[[InstrArgs], Tuple[Optional[Expression], Optional[Expression]]]
]

CASES_IGNORE: InstrSet = {
    # Ignore FCSR sets; they are leftovers from float->unsigned conversions.
    # FCSR gets are as well, but it's fine to read ERROR for those.
    "ctc1",
    "nop",
    "b",
}
CASES_STORE: StoreInstrMap = {
    # Storage instructions
    "sb": lambda a: make_store(a, type=Type.of_size(8)),
    "sh": lambda a: make_store(a, type=Type.of_size(16)),
    "sw": lambda a: make_store(a, type=Type.of_size(32)),
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
    "bgez": lambda a: BinaryOp.scmp(a.reg(0), ">=", Literal(0)),
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
CASES_FLOAT_COMP: CmpInstrMap = {
    # Floating point comparisons
    "c.eq.s": lambda a: BinaryOp.fcmp(a.reg(0), "==", a.reg(1)),
    "c.le.s": lambda a: BinaryOp.fcmp(a.reg(0), "<=", a.reg(1)),
    "c.lt.s": lambda a: BinaryOp.fcmp(a.reg(0), "<", a.reg(1)),
    "c.eq.d": lambda a: BinaryOp.dcmp(a.dreg(0), "==", a.dreg(1)),
    "c.le.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<=", a.dreg(1)),
    "c.lt.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<", a.dreg(1)),
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
    # The high part of multiplication cannot be directly represented in C
    "mult": lambda a: (None, BinaryOp.int(a.reg(0), "*", a.reg(1))),
    "multu": lambda a: (None, BinaryOp.int(a.reg(0), "*", a.reg(1))),
}
CASES_SOURCE_FIRST: InstrMap = {
    # Floating point moving instruction
    "mtc1": lambda a: a.reg(0)
}
CASES_DESTINATION_FIRST: InstrMap = {
    # Flag-setting instructions
    "slt": lambda a: BinaryOp.scmp(a.reg(1), "<", a.reg(2)),
    "slti": lambda a: BinaryOp.scmp(a.reg(1), "<", a.imm(2)),
    "sltu": lambda a: BinaryOp.ucmp(a.reg(1), "<", a.reg(2)),
    "sltiu": lambda a: BinaryOp.ucmp(a.reg(1), "<", a.imm(2)),
    # Integer arithmetic
    "addi": lambda a: handle_addi(a),
    "addiu": lambda a: handle_addi(a),
    "addu": lambda a: handle_add(a.reg(1), a.reg(2), a.stack_info),
    "subu": lambda a: fold_mul_chains(BinaryOp.intptr(a.reg(1), "-", a.reg(2))),
    "negu": lambda a: fold_mul_chains(
        UnaryOp(op="-", expr=as_s32(a.reg(1)), type=Type.s32())
    ),
    "neg": lambda a: fold_mul_chains(
        UnaryOp(op="-", expr=as_s32(a.reg(1)), type=Type.s32())
    ),
    # Hi/lo register uses (used after division/multiplication)
    "mfhi": lambda a: a.regs[Register("hi")],
    "mflo": lambda a: a.regs[Register("lo")],
    # Floating point arithmetic
    "add.s": lambda a: BinaryOp.f32(a.reg(1), "+", a.reg(2)),
    "sub.s": lambda a: BinaryOp.f32(a.reg(1), "-", a.reg(2)),
    "neg.s": lambda a: UnaryOp("-", as_f32(a.reg(1)), type=Type.f32()),
    "abs.s": lambda a: fn_op("fabsf", [as_f32(a.reg(1))], Type.f32()),
    "sqrt.s": lambda a: fn_op("sqrtf", [as_f32(a.reg(1))], Type.f32()),
    "div.s": lambda a: BinaryOp.f32(a.reg(1), "/", a.reg(2)),
    "mul.s": lambda a: BinaryOp.f32(a.reg(1), "*", a.reg(2)),
    # Double-precision arithmetic
    "add.d": lambda a: BinaryOp.f64(a.dreg(1), "+", a.dreg(2)),
    "sub.d": lambda a: BinaryOp.f64(a.dreg(1), "-", a.dreg(2)),
    "neg.d": lambda a: UnaryOp("-", as_f64(a.dreg(1)), type=Type.f64()),
    "abs.d": lambda a: fn_op("fabs", [as_f64(a.dreg(1))], Type.f64()),
    "sqrt.d": lambda a: fn_op("sqrt", [as_f64(a.dreg(1))], Type.f64()),
    "div.d": lambda a: BinaryOp.f64(a.dreg(1), "/", a.dreg(2)),
    "mul.d": lambda a: BinaryOp.f64(a.dreg(1), "*", a.dreg(2)),
    # Floating point conversions
    "cvt.d.s": lambda a: Cast(expr=as_f32(a.reg(1)), type=Type.f64()),
    "cvt.d.w": lambda a: Cast(expr=as_intish(a.reg(1)), type=Type.f64()),
    "cvt.s.d": lambda a: Cast(expr=as_f64(a.dreg(1)), type=Type.f32()),
    "cvt.s.u": lambda a: Cast(expr=as_u32(a.reg(1)), type=Type.f32()),
    "cvt.s.w": lambda a: Cast(expr=as_intish(a.reg(1)), type=Type.f32()),
    "cvt.w.d": lambda a: Cast(expr=as_f64(a.dreg(1)), type=Type.s32()),
    "cvt.w.s": lambda a: Cast(expr=as_f32(a.reg(1)), type=Type.s32()),
    "cvt.u.d": lambda a: Cast(expr=as_f64(a.dreg(1)), type=Type.u32()),
    "cvt.u.s": lambda a: Cast(expr=as_f32(a.reg(1)), type=Type.u32()),
    "trunc.w.s": lambda a: Cast(expr=as_f32(a.reg(1)), type=Type.s32()),
    "trunc.w.d": lambda a: Cast(expr=as_f64(a.dreg(1)), type=Type.s32()),
    # Bit arithmetic
    "ori": lambda a: handle_ori(a),
    "and": lambda a: BinaryOp.int(left=a.reg(1), op="&", right=a.reg(2)),
    "or": lambda a: BinaryOp.int(left=a.reg(1), op="|", right=a.reg(2)),
    "not": lambda a: UnaryOp("~", a.reg(1), type=Type.intish()),
    "nor": lambda a: UnaryOp(
        "~", BinaryOp.int(left=a.reg(1), op="|", right=a.reg(2)), type=Type.intish()
    ),
    "xor": lambda a: BinaryOp.int(left=a.reg(1), op="^", right=a.reg(2)),
    "andi": lambda a: BinaryOp.int(left=a.reg(1), op="&", right=a.imm(2)),
    "xori": lambda a: BinaryOp.int(left=a.reg(1), op="^", right=a.imm(2)),
    "sll": lambda a: fold_mul_chains(
        BinaryOp.int(left=a.reg(1), op="<<", right=a.imm(2))
    ),
    "sllv": lambda a: BinaryOp.int(left=a.reg(1), op="<<", right=a.reg(2)),
    "srl": lambda a: BinaryOp(
        left=as_u32(a.reg(1)), op=">>", right=as_intish(a.imm(2)), type=Type.u32()
    ),
    "srlv": lambda a: BinaryOp(
        left=as_u32(a.reg(1)), op=">>", right=as_intish(a.reg(2)), type=Type.u32()
    ),
    "sra": lambda a: BinaryOp(
        left=as_s32(a.reg(1)), op=">>", right=as_intish(a.imm(2)), type=Type.s32()
    ),
    "srav": lambda a: BinaryOp(
        left=as_s32(a.reg(1)), op=">>", right=as_intish(a.reg(2)), type=Type.s32()
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
    "li": lambda a: a.imm(1),
    "lui": lambda a: load_upper(a),
    # Loading instructions
    "lb": lambda a: handle_load(a, type=Type.s8()),
    "lbu": lambda a: handle_load(a, type=Type.u8()),
    "lh": lambda a: handle_load(a, type=Type.s16()),
    "lhu": lambda a: handle_load(a, type=Type.u16()),
    "lw": lambda a: handle_load(a, type=Type.intptr32()),
    "lwu": lambda a: handle_load(a, type=Type.u32()),
    "lwc1": lambda a: handle_load(a, type=Type.f32()),
    "ldc1": lambda a: handle_load(a, type=Type.f64()),
}


def output_regs_for_instr(
    instr: Instruction, typemap: Optional[TypeMap]
) -> List[Register]:
    def reg_at(index: int) -> List[Register]:
        reg = instr.args[index]
        assert isinstance(reg, Register)
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
    seen = set([node.immediate_dominator])
    stack = node.parents[:]
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
    seen = set([node.immediate_dominator])
    stack = node.parents[:]
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
        assert len(phi.node.parents) >= 2
        exprs = []
        for node in phi.node.parents:
            block_info = node.block.block_info
            assert isinstance(block_info, BlockInfo)
            exprs.append(block_info.final_register_states[phi.reg])

        first_uw = early_unwrap(exprs[0])
        if all(early_unwrap(e) == first_uw for e in exprs[1:]):
            # All the phis have the same value (e.g. because we recomputed an
            # expression after a store, or restored a register after a function
            # call). Just use that value instead of introducing a phi node.
            phi.replacement_expr = as_type(first_uw, phi.type, silent=True)
            for e in exprs[1:]:
                e.type.unify(phi.type)
            for _ in range(phi.num_usages):
                mark_used(exprs[0])
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
                    mark_used(expr)
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
            block_info = n.block.block_info
            assert isinstance(block_info, BlockInfo)
            if block_info.has_custom_return or block_info.has_function_call:
                continue
            for p in n.parents:
                block_info2 = p.block.block_info
                assert isinstance(block_info2, BlockInfo)
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
        always_emit: bool,
        trivial: bool,
        prefix: str = "",
        reuse_var: Optional[Var] = None,
    ) -> EvalOnceExpr:
        if always_emit:
            # (otherwise this will be marked used once num_usages reaches 1)
            mark_used(expr)
        assert reuse_var or prefix
        var = reuse_var or Var(stack_info, "temp_" + prefix)
        expr = EvalOnceExpr(
            wrapped_expr=expr,
            var=var,
            type=expr.type,
            always_emit=always_emit,
            trivial=trivial,
        )
        var.num_usages += 1
        stmt = EvalOnceStmt(expr)
        to_write.append(stmt)
        stack_info.temp_vars.append(stmt)
        return expr

    def prevent_later_uses(sub_expr: Expression, avoid_reg: Optional[Register]) -> None:
        for r in regs.contents.keys():
            if r == avoid_reg:
                # For debugging sanity, don't modify registers that we're just about
                # to overwrite.
                continue
            e = regs.get_raw(r)
            assert e is not None
            if not isinstance(e, ForceVarExpr) and uses_expr(e, sub_expr):
                # Mark the register as "if used, emit the expression's once
                # var". I think we should always have a once var at this point,
                # but if we don't, create one.
                # Exception: unused PassedInArg, which can pass the uses_expr
                # test simply based on having the same variable name.
                if not isinstance(e, EvalOnceExpr):
                    if isinstance(e, PassedInArg) and not e.copied:
                        continue
                    e = eval_once(
                        e, always_emit=False, trivial=False, prefix=r.register_name
                    )
                regs[r] = ForceVarExpr(e, type=e.type)

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

        if isinstance(expr, LocalVar) and expr in local_var_writes:
            # Elide register restores (only for the same register for now, to
            # be conversative).
            orig_reg, orig_expr = local_var_writes[expr]
            if orig_reg == reg:
                expr = orig_expr
        if not isinstance(expr, Literal):
            expr = eval_once(
                expr,
                always_emit=False,
                trivial=is_trivial_expression(expr),
                prefix=reg.register_name,
            )
        if reg == Register("zero"):
            # Emit the expression as is. It's probably a volatile load.
            mark_used(expr)
            to_write.append(ExprStmt(expr))
        else:
            set_reg_maybe_return(reg, expr)

    def overwrite_reg(reg: Register, expr: Expression) -> None:
        prev = regs.get_raw(reg)
        if isinstance(prev, ForceVarExpr):
            prev = prev.wrapped_expr
        if (
            not isinstance(prev, EvalOnceExpr)
            or isinstance(expr, Literal)
            or reg == Register("sp")
            or not prev.type.unify(expr.type)
        ):
            set_reg(reg, expr)
        else:
            # TODO: This is a bit heavy-handed: we're preventing later uses
            # even though we are not sure whether we will actually emit the
            # overwrite. Doing this properly is hard, however -- it would
            # involve tracking "time" for uses, and sometimes moving timestamps
            # backwards when EvalOnceExpr's get emitted as vars.
            prevent_later_uses(prev, avoid_reg=reg)
            set_reg_maybe_return(
                reg,
                eval_once(
                    expr,
                    always_emit=False,
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
                # Note that the prevent_later_uses step happens after mark_used, since
                # the stored expression is allowed to reference its destination var,
                # but before the write is written, since prevent_later_uses might emit
                # writes of its own that should go before this write. In practice that
                # probably never occurs -- all relevant register contents should be
                # EvalOnceExpr's that can be emitted at their point of creation, but
                # I'm not 100% certain that that's always the case and will remain so.
                mark_used(to_store.source)
                mark_used(to_store.dest)
                prevent_later_uses(to_store.dest, avoid_reg=None)
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
            assert isinstance(cond_bit, BinaryOp)
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
                if isinstance(fn_target, AddressOf):
                    fn_target = fn_target.expr
                    assert isinstance(fn_target, GlobalSymbol)
                else:
                    assert isinstance(fn_target, Literal)
                    fn_target = as_ptr(fn_target)
            else:
                assert mnemonic == "jalr"
                if args.count() == 1:
                    fn_target = as_ptr(args.reg(0))
                else:
                    assert args.count() == 2
                    if args.reg_ref(0) != Register("ra"):
                        raise DecompFailure(
                            "Two-argument form of jalr is not supported."
                        )
                    fn_target = as_ptr(args.reg(1))

            # At most one of $f12 and $a0 may be passed, and at most one of
            # $f14 and $a1. We could try to figure out which ones, and cap
            # the function call at the point where a register is empty, but
            # for now we'll leave that for manual fixup.
            typemap = stack_info.typemap
            c_fn: Optional[CFunction] = None
            if typemap and isinstance(fn_target, GlobalSymbol):
                c_fn = typemap.functions.get(fn_target.symbol_name)

            func_args: List[Expression] = []
            if typemap and c_fn and c_fn.params is not None:
                abi_slots, possible_regs = function_abi(c_fn, typemap, for_call=True)
                for slot in abi_slots:
                    if slot.reg:
                        func_args.append(as_type(regs[slot.reg], slot.type, True))
            else:
                possible_regs = list(
                    map(Register, ["f12", "f14", "a0", "a1", "a2", "a3"])
                )

            for register in possible_regs:
                # The latter check verifies that the register is not just
                # meant for us. This might give false positives for the
                # first function call if an argument passed in the same
                # position as we received it, but that's impossible to do
                # anything about without access to function signatures.
                expr = regs.get_raw(register)
                if expr is not None and (
                    not isinstance(expr, PassedInArg) or expr.copied
                ):
                    func_args.append(expr)

            # Add the arguments after a3.
            # TODO: limit this and unify types based on abi_slots
            subroutine_args.sort(key=lambda a: a[1].value)
            for arg in subroutine_args:
                func_args.append(arg[0])

            # Reset subroutine_args, for the next potential function call.
            subroutine_args.clear()

            if c_fn and c_fn.ret_type and typemap:
                known_ret_type = type_from_ctype(c_fn.ret_type, typemap)
            else:
                known_ret_type = Type.any()

            call: Expression = FuncCall(fn_target, func_args, known_ret_type)
            call = eval_once(call, always_emit=True, trivial=False, prefix="ret")

            # Clear out caller-save registers, for clarity and to ensure
            # that argument regs don't get passed into the next function.
            regs.clear_caller_save_regs()

            # We may not know what this function's return registers are --
            # $f0, $v0 or ($v0,$v1) or $f0 -- but we don't really care,
            # it's fine to be liberal here and put the return value in all
            # of them. (It's not perfect for u64's, but that's rare anyway.)
            # However, if we have type information that says the function is
            # void, then don't set any of these -- it might cause us to
            # believe the function we're decompiling is non-void.
            # Note that this logic is duplicated in output_regs_for_instr.
            if not c_fn or c_fn.ret_type:
                regs[Register("f0")] = Cast(
                    expr=call, reinterpret=True, silent=True, type=Type.f32()
                )
                regs[Register("v0")] = Cast(
                    expr=call, reinterpret=True, silent=True, type=Type.intptr()
                )
                regs[Register("v1")] = as_u32(
                    Cast(expr=call, reinterpret=True, silent=False, type=Type.u64())
                )
                regs[Register("return")] = call

            has_custom_return = False
            has_function_call = True

        elif mnemonic in CASES_FLOAT_COMP:
            expr = CASES_FLOAT_COMP[mnemonic](args)
            assert expr is not None
            regs[Register("condition_bit")] = expr

        elif mnemonic in CASES_HI_LO:
            hi, lo = CASES_HI_LO[mnemonic](args)
            set_reg(Register("hi"), hi)
            set_reg(Register("lo"), lo)

        elif mnemonic in CASES_DESTINATION_FIRST:
            target = args.reg_ref(0)
            val = CASES_DESTINATION_FIRST[mnemonic](args)
            if target in args.raw_args[1:]:
                # IRIX tends to keep variables within single registers. Thus,
                # if source = target, overwrite that variable instead of
                # creating a new one.
                overwrite_reg(target, val)
            else:
                set_reg(target, val)
            mn_parts = mnemonic.split(".")
            if (len(mn_parts) >= 2 and mn_parts[1] == "d") or mnemonic == "ldc1":
                set_reg(target.other_f64_reg(), SecondF64Half())

        else:
            expr = ErrorExpr(f"unknown instruction: {instr}")
            if args.count() >= 1 and isinstance(args.raw_args[0], Register):
                reg = args.reg_ref(0)
                expr = eval_once(
                    expr, always_emit=True, trivial=False, prefix=reg.register_name
                )
                if reg != Register("zero"):
                    set_reg_maybe_return(reg, expr)
            else:
                to_write.append(ExprStmt(expr))

    for instr in node.block.instructions:
        with current_instr(instr):
            process_instr(instr)

    if branch_condition is not None:
        mark_used(branch_condition)
    if switch_value is not None:
        mark_used(switch_value)
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
        print(f"\nNode in question: {node.block}")

    # Translate the given node and discover final register states.
    try:
        block_info = translate_node_body(node, regs, stack_info)
        if options.debug:
            print(block_info)
    except Exception as e:  # TODO: handle issues better
        if options.stop_on_error:
            raise e

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
    typemap = stack_info.typemap
    for child in node.immediately_dominates:
        new_contents = regs.contents.copy()
        phi_regs = regs_clobbered_until_dominator(child, typemap)
        for reg in phi_regs:
            if reg_always_set(child, reg, typemap, dom_set=(reg in regs)):
                new_contents[reg] = PhiExpr(
                    reg=reg, node=child, used_phis=used_phis, type=Type.any()
                )
            elif reg in new_contents:
                del new_contents[reg]
        new_regs = RegInfo(contents=new_contents, stack_info=stack_info)
        translate_graph_from_block(
            child, new_regs, stack_info, used_phis, return_blocks, options
        )


@attr.s
class FunctionInfo:
    stack_info: StackInfo = attr.ib()
    flow_graph: FlowGraph = attr.ib()
    return_type: Type = attr.ib()


def translate_to_ast(
    function: Function, options: Options, rodata: Rodata, typemap: Optional[TypeMap]
) -> FunctionInfo:
    """
    Given a function, produce a FlowGraph that both contains control-flow
    information and has AST transformations for each block of code and
    branch condition.
    """
    # Initialize info about the function.
    flow_graph: FlowGraph = build_flowgraph(function, rodata)
    start_node = flow_graph.entry_node()
    stack_info = get_stack_info(function, rodata, start_node, typemap)

    initial_regs: Dict[Register, Expression] = {
        Register("sp"): GlobalSymbol("sp", type=Type.ptr()),
        **{reg: stack_info.saved_reg_symbol(reg.register_name) for reg in SAVED_REGS},
    }

    def make_arg(offset: int, type: Type) -> PassedInArg:
        return PassedInArg(offset, copied=False, stack_info=stack_info, type=type)

    c_fn: Optional[CFunction] = None
    known_params = False
    variadic = False
    return_type = Type.any()
    if typemap and function.name in typemap.functions:
        c_fn = typemap.functions[function.name]
        if c_fn.ret_type is not None:
            return_type = type_from_ctype(c_fn.ret_type, typemap)
        if c_fn.is_variadic:
            stack_info.is_variadic = True
        if c_fn.params is not None:
            abi_slots, possible_regs = function_abi(c_fn, typemap, for_call=False)
            for slot in abi_slots:
                if slot.name is not None:
                    stack_info.set_param_name(slot.offset, slot.name)
                if slot.reg is not None:
                    initial_regs[slot.reg] = make_arg(slot.offset, slot.type)
            for reg in possible_regs:
                offset = 4 * int(reg.register_name[1])
                initial_regs[reg] = make_arg(offset, Type.any())
            known_params = True

    if not known_params:
        initial_regs.update(
            {
                Register("a0"): make_arg(0, Type.intptr()),
                Register("a1"): make_arg(4, Type.any()),
                Register("a2"): make_arg(8, Type.any()),
                Register("a3"): make_arg(12, Type.any()),
                Register("f12"): make_arg(0, Type.f32()),
                Register("f14"): make_arg(4, Type.f32()),
            }
        )

    if options.debug:
        print(stack_info)
        print("\nNow, we attempt to translate:")

    start_reg: RegInfo = RegInfo(contents=initial_regs, stack_info=stack_info)
    used_phis: List[PhiExpr] = []
    return_blocks: List[BlockInfo] = []
    translate_graph_from_block(
        start_node, start_reg, stack_info, used_phis, return_blocks, options
    )

    # We mark the function as having a return type if all return nodes have
    # return values, and not all those values are trivial (e.g. from function
    # calls). TODO: check that the values aren't read from for some other purpose.
    compute_has_custom_return(flow_graph.nodes)
    has_return = all(b.return_value is not None for b in return_blocks) and any(
        b.has_custom_return for b in return_blocks
    )

    if options.void:
        has_return = False
    elif c_fn is not None:
        if c_fn.ret_type is None:
            has_return = False
        else:
            has_return = True

    if has_return:
        for b in return_blocks:
            if b.return_value is not None:
                ret_val = as_type(b.return_value, return_type, True)
                b.return_value = ret_val
                mark_used(ret_val)
    else:
        for b in return_blocks:
            b.return_value = None

    assign_phis(used_phis, stack_info)
    return FunctionInfo(stack_info, flow_graph, return_type)
