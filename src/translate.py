import abc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import math
import struct
import sys
import traceback
import typing
from typing import (
    AbstractSet,
    Callable,
    Collection,
    DefaultDict,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from .c_types import CType, TypeMap
from .demangle_codewarrior import parse as demangle_codewarrior_parse, CxxSymbol
from .error import DecompFailure, static_assert_unreachable
from .flow_graph import (
    ArchFlowGraph,
    FlowGraph,
    Function,
    Node,
    ReturnNode,
    SwitchNode,
    TerminalNode,
)
from .options import CodingStyle, Formatter, Options, Target
from .parse_file import AsmData, AsmDataEntry
from .parse_instruction import (
    ArchAsm,
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmLiteral,
    BinOp,
    Instruction,
    InstrProcessingFailure,
    MemoryAccess,
    Macro,
    Register,
    current_instr,
)
from .types import (
    AccessPath,
    FunctionParam,
    FunctionSignature,
    StructDeclaration,
    Type,
    TypePool,
)

InstrSet = Collection[str]
InstrMap = Mapping[str, Callable[["InstrArgs"], "Expression"]]
StmtInstrMap = Mapping[str, Callable[["InstrArgs"], "Statement"]]
CmpInstrMap = Mapping[str, Callable[["InstrArgs"], "Condition"]]
StoreInstrMap = Mapping[str, Callable[["InstrArgs"], Optional["StoreStmt"]]]
MaybeInstrMap = Mapping[str, Callable[["InstrArgs"], Optional["Expression"]]]
PairInstrMap = Mapping[str, Callable[["InstrArgs"], Tuple["Expression", "Expression"]]]
ImplicitInstrMap = Mapping[str, Tuple[Register, Callable[["InstrArgs"], "Expression"]]]
PpcCmpInstrMap = Mapping[str, Callable[["InstrArgs", str], "Expression"]]


class Arch(ArchFlowGraph):
    instrs_ignore: InstrSet = set()
    instrs_store: StoreInstrMap = {}
    instrs_store_update: StoreInstrMap = {}
    instrs_load_update: InstrMap = {}

    instrs_branches: CmpInstrMap = {}
    instrs_float_branches: InstrSet = set()
    instrs_float_comp: CmpInstrMap = {}
    instrs_ppc_compare: PpcCmpInstrMap = {}
    instrs_jumps: InstrSet = set()
    instrs_fn_call: InstrSet = set()

    instrs_no_dest: StmtInstrMap = {}
    instrs_hi_lo: PairInstrMap = {}
    instrs_source_first: InstrMap = {}
    instrs_destination_first: InstrMap = {}
    instrs_implicit_destination: ImplicitInstrMap = {}

    @abc.abstractmethod
    def function_abi(
        self,
        fn_sig: FunctionSignature,
        likely_regs: Dict[Register, bool],
        *,
        for_call: bool,
    ) -> "Abi":
        """
        Compute stack positions/registers used by a function based on its type
        information. Also computes a list of registers that may contain arguments,
        if the function has varargs or an unknown/incomplete type.
        """
        ...

    @abc.abstractmethod
    def function_return(self, expr: "Expression") -> Dict[Register, "Expression"]:
        """
        Compute register location(s) & values that will hold the return value
        of the function call `expr`.
        This must have a value for each register in `all_return_regs` in order to stay
        consistent with `Instruction.outputs`. This is why we can't use the
        function's return type, even though it may be more accurate.
        """
        ...


ASSOCIATIVE_OPS: Set[str] = {"+", "&&", "||", "&", "|", "^", "*"}
COMPOUND_ASSIGNMENT_OPS: Set[str] = {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"}
PSEUDO_FUNCTION_OPS: Set[str] = {"MULT_HI", "MULTU_HI", "DMULT_HI", "DMULTU_HI", "CLZ"}


def as_type(expr: "Expression", type: Type, silent: bool) -> "Expression":
    type = type.weaken_void_ptr()
    ptr_target_type = type.get_pointer_target()
    if expr.type.unify(type):
        if silent or isinstance(expr, Literal):
            return expr
    elif ptr_target_type is not None:
        ptr_target_type_size = ptr_target_type.get_size_bytes()
        field_path, field_type, _ = expr.type.get_deref_field(
            0, target_size=ptr_target_type_size
        )
        if field_path is not None and field_type.unify(ptr_target_type):
            expr = AddressOf(
                StructAccess(
                    struct_var=expr,
                    offset=0,
                    target_size=ptr_target_type_size,
                    field_path=field_path,
                    stack_info=None,
                    type=field_type,
                ),
                type=type,
            )
            if silent:
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


@dataclass
class StackInfo:
    function: Function
    global_info: "GlobalInfo"
    flow_graph: FlowGraph
    allocated_stack_size: int = 0
    is_leaf: bool = True
    is_variadic: bool = False
    uses_framepointer: bool = False
    subroutine_arg_top: int = 0
    callee_save_reg_locations: Dict[Register, int] = field(default_factory=dict)
    callee_save_reg_region: Tuple[int, int] = (0, 0)
    unique_type_map: Dict[Tuple[str, object], "Type"] = field(default_factory=dict)
    local_vars: List["LocalVar"] = field(default_factory=list)
    temp_vars: List["EvalOnceStmt"] = field(default_factory=list)
    phi_vars: List["PhiExpr"] = field(default_factory=list)
    reg_vars: Dict[Register, "RegisterVar"] = field(default_factory=dict)
    used_reg_vars: Set[Register] = field(default_factory=set)
    arguments: List["PassedInArg"] = field(default_factory=list)
    temp_name_counter: Dict[str, int] = field(default_factory=dict)
    nonzero_accesses: Set["Expression"] = field(default_factory=set)
    param_names: Dict[int, str] = field(default_factory=dict)
    stack_pointer_type: Optional[Type] = None
    replace_first_arg: Optional[Tuple[str, Type]] = None
    weak_stack_var_types: Dict[int, Type] = field(default_factory=dict)
    weak_stack_var_locations: Set[int] = field(default_factory=set)

    def temp_var(self, prefix: str) -> str:
        counter = self.temp_name_counter.get(prefix, 0) + 1
        self.temp_name_counter[prefix] = counter
        return prefix + (f"_{counter}" if counter > 1 else "")

    def in_subroutine_arg_region(self, location: int) -> bool:
        if self.global_info.arch.arch == Target.ArchEnum.PPC:
            return False
        if self.is_leaf:
            return False
        assert self.subroutine_arg_top is not None
        return location < self.subroutine_arg_top

    def in_callee_save_reg_region(self, location: int) -> bool:
        lower_bound, upper_bound = self.callee_save_reg_region
        if lower_bound <= location < upper_bound:
            return True
        # PPC saves LR in the header of the previous stack frame
        if (
            self.global_info.arch.arch == Target.ArchEnum.PPC
            and location == self.allocated_stack_size + 4
        ):
            return True
        return False

    def location_above_stack(self, location: int) -> bool:
        return location >= self.allocated_stack_size

    def add_known_param(self, offset: int, name: Optional[str], type: Type) -> None:
        # A common pattern in C for OOP-style polymorphism involves casting a general "base" struct
        # to a specific "class" struct, where the first member of the class struct is the base struct.
        #
        # For the first argument of the function, if it is a pointer to a base struct, and there
        # exists a class struct named after the first part of the function name, assume that
        # this pattern is being used. Internally, treat the argument as a pointer to the *class*
        # struct, even though it is only a pointer to the *base* struct in the provided context.
        if offset == 0 and type.is_pointer() and self.replace_first_arg is None:
            namespace = self.function.name.partition("_")[0]
            base_struct_type = type.get_pointer_target()
            self_struct = self.global_info.typepool.get_struct_by_tag_name(
                namespace, self.global_info.typemap
            )
            if (
                self_struct is not None
                and base_struct_type is not None
                and base_struct_type.is_struct()
            ):
                # Check if `self_struct_type` contains a `base_struct_type` at offset 0
                self_struct_type = Type.struct(self_struct)
                field_path, field_type, _ = self_struct_type.get_field(
                    offset=0, target_size=base_struct_type.get_size_bytes()
                )
                if (
                    field_path is not None
                    and field_type.unify(base_struct_type)
                    and not self_struct_type.unify(base_struct_type)
                ):
                    # Success, it looks like `self_struct_type` extends `base_struct_type`.
                    # By default, name the local var `self`, unless the argument name is `thisx` then use `this`
                    self.replace_first_arg = (name or "_self", type)
                    name = "this" if name == "thisx" else "self"
                    type = Type.ptr(Type.struct(self_struct))
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
            return as_type(arg, Type.int_of_size(8), True), arg
        if real_location == location - 2:
            return as_type(arg, Type.int_of_size(16), True), arg
        return arg, arg

    def record_struct_access(self, ptr: "Expression", location: int) -> None:
        if location:
            self.nonzero_accesses.add(unwrap_deep(ptr))

    def has_nonzero_access(self, ptr: "Expression") -> bool:
        return unwrap_deep(ptr) in self.nonzero_accesses

    def unique_type_for(self, category: str, key: object, default: Type) -> "Type":
        key = (category, key)
        if key not in self.unique_type_map:
            self.unique_type_map[key] = default
        return self.unique_type_map[key]

    def saved_reg_symbol(self, reg_name: str) -> "GlobalSymbol":
        sym_name = "saved_reg_" + reg_name
        type = self.unique_type_for("saved_reg", sym_name, Type.any_reg())
        return GlobalSymbol(symbol_name=sym_name, type=type)

    def should_save(self, expr: "Expression", offset: Optional[int]) -> bool:
        expr = early_unwrap(expr)
        if isinstance(expr, GlobalSymbol) and (
            expr.symbol_name.startswith("saved_reg_") or expr.symbol_name == "sp"
        ):
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
        if self.in_callee_save_reg_region(location):
            # Some annoying bookkeeping instruction. To avoid
            # further special-casing, just return whatever - it won't matter.
            return LocalVar(location, type=Type.any_reg(), path=None)
        elif self.location_above_stack(location):
            ret, arg = self.get_argument(location - self.allocated_stack_size)
            if not store:
                self.add_argument(arg)
            return ret
        elif self.in_subroutine_arg_region(location):
            return SubroutineArg(location, type=Type.any_reg())
        else:
            # Local variable
            assert self.stack_pointer_type is not None
            field_path, field_type, _ = self.stack_pointer_type.get_deref_field(
                location, target_size=None
            )

            # Some variables on the stack are compiler-managed, and aren't declared
            # in the original source. These variables can have different types inside
            # different blocks, so we track their types but assume that they may change
            # on each store.
            # TODO: Because the types are tracked in StackInfo instead of RegInfo, it is
            # possible that a load could incorrectly use a weak type from a sibling node
            # instead of a parent node. A more correct implementation would use similar
            # logic to the PhiExpr system. In practice however, storing types in StackInfo
            # works well enough because nodes are traversed approximately depth-first.
            # TODO: Maybe only do this for certain configurable regions?

            # Get the previous type stored in `location`
            previous_stored_type = self.weak_stack_var_types.get(location)
            if previous_stored_type is not None:
                # Check if the `field_type` is compatible with the type of the last store
                if not previous_stored_type.unify(field_type):
                    # The types weren't compatible: mark this `location` as "weak"
                    # This marker is only used to annotate the output
                    self.weak_stack_var_locations.add(location)

                if store:
                    # If there's already been a store to `location`, then return a fresh type
                    field_type = Type.any_field()
                else:
                    # Use the type of the last store instead of the one from `get_deref_field()`
                    field_type = previous_stored_type

            # Track the type last stored at `location`
            if store:
                self.weak_stack_var_types[location] = field_type

            return LocalVar(location, type=field_type, path=field_path)

    def maybe_get_register_var(self, reg: Register) -> Optional["RegisterVar"]:
        return self.reg_vars.get(reg)

    def add_register_var(self, reg: Register) -> None:
        type = Type.floatish() if reg.is_float() else Type.intptr()
        self.reg_vars[reg] = RegisterVar(reg=reg, type=type)

    def use_register_var(self, var: "RegisterVar") -> None:
        self.used_reg_vars.add(var.reg)

    def is_stack_reg(self, reg: Register) -> bool:
        if reg == self.global_info.arch.stack_pointer_reg:
            return True
        if reg == self.global_info.arch.frame_pointer_reg:
            return self.uses_framepointer
        return False

    def get_struct_type_map(self) -> Dict["Expression", Dict[int, Type]]:
        """Reorganize struct information in unique_type_map by var & offset"""
        struct_type_map: Dict[Expression, Dict[int, Type]] = {}
        for (category, key), type in self.unique_type_map.items():
            if category != "struct":
                continue
            var, offset = typing.cast(Tuple[Expression, int], key)
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
                f"Bounds of callee-saved vars region: {self.callee_save_reg_region}",
                f"Locations of callee save registers: {self.callee_save_reg_locations}",
            ]
        )


def get_stack_info(
    function: Function,
    global_info: "GlobalInfo",
    flow_graph: FlowGraph,
) -> StackInfo:
    arch = global_info.arch
    info = StackInfo(function, global_info, flow_graph)

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
    # Track simple literal values stored into registers: MIPS compilers need a temp
    # reg to move the stack pointer more than 0x7FFF bytes.
    temp_reg_values: Dict[Register, int] = {}
    for inst in flow_graph.entry_node().block.instructions:
        arch_mnemonic = inst.arch_mnemonic(arch)
        if inst.mnemonic in arch.instrs_fn_call:
            break
        elif arch_mnemonic == "mips:addiu" and inst.args[0] == arch.stack_pointer_reg:
            # Moving the stack pointer on MIPS
            assert isinstance(inst.args[2], AsmLiteral)
            info.allocated_stack_size = abs(inst.args[2].signed_value())
        elif (
            arch_mnemonic == "mips:subu"
            and inst.args[0] == arch.stack_pointer_reg
            and inst.args[1] == arch.stack_pointer_reg
            and inst.args[2] in temp_reg_values
        ):
            # Moving the stack pointer more than 0x7FFF on MIPS
            # TODO: This instruction needs to be ignored later in translation, in the
            # same way that `addiu $sp, $sp, N` is ignored in handle_addi_real
            assert isinstance(inst.args[2], Register)
            info.allocated_stack_size = temp_reg_values[inst.args[2]]
        elif arch_mnemonic == "ppc:stwu" and inst.args[0] == arch.stack_pointer_reg:
            # Moving the stack pointer on PPC
            assert isinstance(inst.args[1], AsmAddressMode)
            assert isinstance(inst.args[1].lhs, AsmLiteral)
            info.allocated_stack_size = abs(inst.args[1].lhs.signed_value())
        elif (
            arch_mnemonic == "mips:move"
            and inst.args[0] == arch.frame_pointer_reg
            and inst.args[1] == arch.stack_pointer_reg
        ):
            # "move fp, sp" very likely means the code is compiled with frame
            # pointers enabled; thus fp should be treated the same as sp.
            info.uses_framepointer = True
        elif (
            arch_mnemonic
            in ["mips:sw", "mips:swc1", "mips:sdc1", "ppc:stw", "ppc:stmw", "ppc:stfd"]
            and isinstance(inst.args[0], Register)
            and inst.args[0] in arch.saved_regs
            and isinstance(inst.args[1], AsmAddressMode)
            and inst.args[1].rhs == arch.stack_pointer_reg
            and inst.args[0] not in info.callee_save_reg_locations
        ):
            # Initial saving of callee-save register onto the stack.
            if inst.args[0] in (arch.return_address_reg, Register("r0")):
                # Saving the return address on the stack.
                info.is_leaf = False
            # The registers & their stack accesses must be matched up in ArchAsm.parse
            for reg, mem in zip(inst.inputs, inst.outputs):
                if (
                    isinstance(reg, Register)
                    and isinstance(mem, MemoryAccess)
                    and mem.base_reg == arch.stack_pointer_reg
                    and isinstance(mem.offset, AsmLiteral)
                ):
                    stack_offset = mem.offset.value
                    info.callee_save_reg_locations[reg] = stack_offset
                    callee_saved_offset_and_size.append((stack_offset, mem.size))
        elif arch_mnemonic == "ppc:mflr" and inst.args[0] == Register("r0"):
            info.is_leaf = False
        elif arch_mnemonic == "mips:li" and inst.args[0] in arch.temp_regs:
            assert isinstance(inst.args[0], Register)
            assert isinstance(inst.args[1], AsmLiteral)
            temp_reg_values[inst.args[0]] = inst.args[1].value
        elif (
            arch_mnemonic == "mips:ori"
            and inst.args[0] == inst.args[1]
            and inst.args[0] in temp_reg_values
        ):
            assert isinstance(inst.args[0], Register)
            assert isinstance(inst.args[2], AsmLiteral)
            temp_reg_values[inst.args[0]] |= inst.args[2].value

    if not info.is_leaf:
        # Iterate over the whole function, not just the first basic block,
        # to estimate the boundary for the subroutine argument region
        info.subroutine_arg_top = info.allocated_stack_size
        for node in flow_graph.nodes:
            for inst in node.block.instructions:
                arch_mnemonic = inst.arch_mnemonic(arch)
                if (
                    arch_mnemonic in ["mips:lw", "mips:lwc1", "mips:ldc1", "ppc:lwz"]
                    and isinstance(inst.args[1], AsmAddressMode)
                    and inst.args[1].rhs == arch.stack_pointer_reg
                    and inst.args[1].lhs_as_literal() >= 16
                ):
                    info.subroutine_arg_top = min(
                        info.subroutine_arg_top, inst.args[1].lhs_as_literal()
                    )
                elif (
                    arch_mnemonic == "mips:addiu"
                    and inst.args[0] != arch.stack_pointer_reg
                    and inst.args[1] == arch.stack_pointer_reg
                    and isinstance(inst.args[2], AsmLiteral)
                    and inst.args[2].value < info.allocated_stack_size
                ):
                    info.subroutine_arg_top = min(
                        info.subroutine_arg_top, inst.args[2].value
                    )

        # Compute the bounds of the callee-saved register region, including padding
        if callee_saved_offset_and_size:
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
            info.callee_save_reg_region = (bottom, top)

            # Subroutine arguments must be at the very bottom of the stack, so they
            # must come after the callee-saved region
            info.subroutine_arg_top = min(info.subroutine_arg_top, bottom)

    # Use a struct to represent the stack layout. If the struct is provided in the context,
    # its fields will be used for variable types & names.
    stack_struct_name = f"_mips2c_stack_{function.name}"
    stack_struct = global_info.typepool.get_struct_by_tag_name(
        stack_struct_name, global_info.typemap
    )
    if stack_struct is not None:
        if stack_struct.size != info.allocated_stack_size:
            raise DecompFailure(
                f"Function {function.name} has a provided stack type {stack_struct_name} "
                f"with size {stack_struct.size}, but the detected stack size was "
                f"{info.allocated_stack_size}."
            )
    else:
        stack_struct = StructDeclaration.unknown_of_size(
            global_info.typepool,
            size=info.allocated_stack_size,
            tag_name=stack_struct_name,
        )
    # Mark the struct as a stack struct so we never try to use a reference to the struct itself
    stack_struct.is_stack = True
    stack_struct.new_field_prefix = "sp"

    # This acts as the type of the $sp register
    info.stack_pointer_type = Type.ptr(Type.struct(stack_struct))

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


@dataclass(eq=False)
class Var:
    stack_info: StackInfo = field(repr=False)
    prefix: str
    num_usages: int = 0
    name: Optional[str] = None

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


@dataclass(frozen=True, eq=False)
class ErrorExpr(Condition):
    desc: Optional[str] = None
    type: Type = field(default_factory=Type.any_reg)

    def dependencies(self) -> List[Expression]:
        return []

    def negated(self) -> "Condition":
        return self

    def format(self, fmt: Formatter) -> str:
        if self.desc is not None:
            return f"MIPS2C_ERROR({self.desc})"
        return "MIPS2C_ERROR()"


@dataclass(frozen=True)
class CommentExpr(Expression):
    expr: Expression
    type: Type = field(compare=False)
    prefix: Optional[str] = None
    suffix: Optional[str] = None

    def dependencies(self) -> List[Expression]:
        return [self.expr]

    def format(self, fmt: Formatter) -> str:
        expr_str = self.expr.format(fmt)

        if fmt.coding_style.comment_style == CodingStyle.CommentStyle.NONE:
            return expr_str

        prefix_str = f"/* {self.prefix} */ " if self.prefix is not None else ""
        suffix_str = f" /* {self.suffix} */" if self.suffix is not None else ""
        return f"{prefix_str}{expr_str}{suffix_str}"

    @staticmethod
    def wrap(
        expr: Expression, prefix: Optional[str] = None, suffix: Optional[str] = None
    ) -> Expression:
        if prefix is None and suffix is None:
            return expr
        return CommentExpr(expr=expr, type=expr.type, prefix=prefix, suffix=suffix)


@dataclass(frozen=True, eq=False)
class SecondF64Half(Expression):
    type: Type = field(default_factory=Type.any_reg)

    def dependencies(self) -> List[Expression]:
        return []

    def format(self, fmt: Formatter) -> str:
        return "(second half of f64)"


@dataclass(frozen=True, eq=False)
class CarryBit(Expression):
    type: Type = field(default_factory=Type.intish)

    def dependencies(self) -> List[Expression]:
        return []

    def format(self, fmt: Formatter) -> str:
        return "MIPS2C_CARRY"

    @staticmethod
    def add_to(expr: Expression) -> "BinaryOp":
        return fold_divmod(BinaryOp.intptr(expr, "+", CarryBit()))

    @staticmethod
    def sub_from(expr: Expression) -> "BinaryOp":
        return BinaryOp.intptr(expr, "-", UnaryOp("!", CarryBit(), type=Type.intish()))


@dataclass(frozen=True, eq=False)
class BinaryOp(Condition):
    left: Expression
    op: str
    right: Expression
    type: Type

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
    def sintptr_cmp(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(
            left=as_type(left, Type.sintptr(), False),
            op=op,
            right=as_type(right, Type.sintptr(), False),
            type=Type.bool(),
        )

    @staticmethod
    def ucmp(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(left=as_u32(left), op=op, right=as_u32(right), type=Type.bool())

    @staticmethod
    def uintptr_cmp(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(
            left=as_type(left, Type.uintptr(), False),
            op=op,
            right=as_type(right, Type.uintptr(), False),
            type=Type.bool(),
        )

    @staticmethod
    def fcmp(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(
            left=as_f32(left),
            op=op,
            right=as_f32(right),
            type=Type.bool(),
        )

    @staticmethod
    def dcmp(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(
            left=as_f64(left),
            op=op,
            right=as_f64(right),
            type=Type.bool(),
        )

    @staticmethod
    def s32(
        left: Expression, op: str, right: Expression, silent: bool = False
    ) -> "BinaryOp":
        return BinaryOp(
            left=as_s32(left, silent=silent),
            op=op,
            right=as_s32(right, silent=silent),
            type=Type.s32(),
        )

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
        )

    @staticmethod
    def f64(left: Expression, op: str, right: Expression) -> "BinaryOp":
        return BinaryOp(
            left=as_f64(left),
            op=op,
            right=as_f64(right),
            type=Type.f64(),
        )

    def is_comparison(self) -> bool:
        return self.op in ["==", "!=", ">", "<", ">=", "<="]

    def is_floating(self) -> bool:
        return self.left.type.is_float() and self.right.type.is_float()

    def negated(self) -> "Condition":
        if (
            self.op in ["&&", "||"]
            and isinstance(self.left, Condition)
            and isinstance(self.right, Condition)
        ):
            # DeMorgan's Laws
            return BinaryOp(
                left=self.left.negated(),
                op={"&&": "||", "||": "&&"}[self.op],
                right=self.right.negated(),
                type=Type.bool(),
            )
        if not self.is_comparison() or (
            self.is_floating() and self.op in ["<", ">", "<=", ">="]
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
        left_expr = late_unwrap(self.left)
        right_expr = late_unwrap(self.right)
        if (
            self.is_comparison()
            and isinstance(left_expr, Literal)
            and not isinstance(right_expr, Literal)
        ):
            return BinaryOp(
                left=right_expr,
                op=self.op.translate(str.maketrans("<>", "><")),
                right=left_expr,
                type=self.type,
            ).format(fmt)

        if (
            not self.is_floating()
            and isinstance(right_expr, Literal)
            and right_expr.value < 0
        ):
            if self.op == "+":
                neg = Literal(value=-right_expr.value, type=right_expr.type)
                sub = BinaryOp(op="-", left=left_expr, right=neg, type=self.type)
                return sub.format(fmt)
            if self.op in ("&", "|"):
                neg = Literal(value=~right_expr.value, type=right_expr.type)
                right = UnaryOp("~", neg, type=Type.any_reg())
                expr = BinaryOp(op=self.op, left=left_expr, right=right, type=self.type)
                return expr.format(fmt)

        # For commutative, left-associative operations, strip unnecessary parentheses.
        lhs = left_expr.format(fmt)
        if (
            isinstance(left_expr, BinaryOp)
            and left_expr.op == self.op
            and self.op in ASSOCIATIVE_OPS
        ):
            lhs = lhs[1:-1]

        # For certain operators, use base-10 (decimal) for the RHS
        if self.op in ("/", "%") and isinstance(right_expr, Literal):
            rhs = right_expr.format(fmt, force_dec=True)
        else:
            rhs = right_expr.format(fmt)

        # These aren't real operators (or functions); format them as a fn call
        if self.op in PSEUDO_FUNCTION_OPS:
            return f"{self.op}({lhs}, {rhs})"

        return f"({lhs} {self.op} {rhs})"


@dataclass(frozen=True, eq=False)
class TernaryOp(Expression):
    cond: Condition
    left: Expression
    right: Expression
    type: Type

    def dependencies(self) -> List[Expression]:
        return [self.cond, self.left, self.right]

    def format(self, fmt: Formatter) -> str:
        cond_str = simplify_condition(self.cond).format(fmt)
        left_str = self.left.format(fmt)
        right_str = self.right.format(fmt)
        return f"({cond_str} ? {left_str} : {right_str})"


@dataclass(frozen=True, eq=False)
class UnaryOp(Condition):
    op: str
    expr: Expression
    type: Type

    def dependencies(self) -> List[Expression]:
        return [self.expr]

    def negated(self) -> "Condition":
        if self.op == "!" and isinstance(self.expr, (UnaryOp, BinaryOp)):
            return self.expr
        return UnaryOp("!", self, type=Type.bool())

    def format(self, fmt: Formatter) -> str:
        # These aren't real operators (or functions); format them as a fn call
        if self.op in PSEUDO_FUNCTION_OPS:
            return f"{self.op}({self.expr.format(fmt)})"

        return f"{self.op}{self.expr.format(fmt)}"


@dataclass(frozen=True, eq=False)
class ExprCondition(Condition):
    expr: Expression
    type: Type
    is_negated: bool = False

    def dependencies(self) -> List[Expression]:
        return [self.expr]

    def negated(self) -> "Condition":
        return ExprCondition(self.expr, self.type, not self.is_negated)

    def format(self, fmt: Formatter) -> str:
        neg = "!" if self.is_negated else ""
        return f"{neg}{self.expr.format(fmt)}"


@dataclass(frozen=True, eq=False)
class CommaConditionExpr(Condition):
    statements: List["Statement"]
    condition: "Condition"
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


@dataclass(frozen=True, eq=False)
class Cast(Expression):
    expr: Expression
    type: Type
    reinterpret: bool = False
    silent: bool = True

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


@dataclass(frozen=True, eq=False)
class FuncCall(Expression):
    function: Expression
    args: List[Expression]
    type: Type

    def dependencies(self) -> List[Expression]:
        return self.args + [self.function]

    def format(self, fmt: Formatter) -> str:
        # TODO: The function type may have a different number of params than it had
        # when the FuncCall was created. Should we warn that there may be the wrong
        # number of arguments at this callsite?
        args = ", ".join(format_expr(arg, fmt) for arg in self.args)
        return f"{self.function.format(fmt)}({args})"


@dataclass(frozen=True, eq=True)
class LocalVar(Expression):
    value: int
    type: Type = field(compare=False)
    path: Optional[AccessPath] = field(compare=False)

    def dependencies(self) -> List[Expression]:
        return []

    def format(self, fmt: Formatter) -> str:
        fallback_name = f"unksp{format_hex(self.value)}"
        if self.path is None:
            return fallback_name

        name = StructAccess.access_path_to_field_name(self.path, fmt)
        if name.startswith("->"):
            return name[2:]
        return fallback_name

    def toplevel_decl(self, fmt: Formatter) -> Optional[str]:
        """Return a declaration for this LocalVar, if required."""
        # If len(self.path) > 2, then this local is an inner field of another
        # local, so it doesn't need to be declared.
        if (
            self.path is None
            or len(self.path) != 2
            or not isinstance(self.path[1], str)
        ):
            return None
        return self.type.to_decl(self.path[1], fmt)


@dataclass(frozen=True, eq=False)
class RegisterVar(Expression):
    reg: Register
    type: Type

    def dependencies(self) -> List[Expression]:
        return []

    def format(self, fmt: Formatter) -> str:
        return self.reg.register_name


@dataclass(frozen=True, eq=True)
class PassedInArg(Expression):
    value: int
    copied: bool = field(compare=False)
    stack_info: StackInfo = field(compare=False, repr=False)
    type: Type = field(compare=False)

    def dependencies(self) -> List[Expression]:
        return []

    def format(self, fmt: Formatter) -> str:
        assert self.value % 4 == 0
        name = self.stack_info.get_param_name(self.value)
        return name or f"arg{format_hex(self.value // 4)}"


@dataclass(frozen=True, eq=True)
class SubroutineArg(Expression):
    value: int
    type: Type = field(compare=False)

    def dependencies(self) -> List[Expression]:
        return []

    def format(self, fmt: Formatter) -> str:
        return f"subroutine_arg{format_hex(self.value // 4)}"


@dataclass(eq=True, unsafe_hash=True)
class StructAccess(Expression):
    # Represents struct_var->offset.
    # This has eq=True since it represents a live expression and not an access
    # at a certain point in time -- this sometimes helps get rid of phi nodes.
    # prevent_later_uses makes sure it's not used after writes/function calls
    # that may invalidate it.
    struct_var: Expression
    offset: int
    target_size: Optional[int]
    field_path: Optional[AccessPath] = field(compare=False)
    stack_info: Optional[StackInfo] = field(compare=False, repr=False)
    type: Type = field(compare=False)
    checked_late_field_path: bool = field(default=False, compare=False)

    def __post_init__(self) -> None:
        # stack_info is used to resolve field_path late
        assert (
            self.stack_info is not None or self.field_path is not None
        ), "Must provide at least one of (stack_info, field_path)"
        self.assert_valid_field_path(self.field_path)

    @staticmethod
    def assert_valid_field_path(path: Optional[AccessPath]) -> None:
        assert path is None or (
            path and isinstance(path[0], int)
        ), "The first element of the field path, if present, must be an int"

    @classmethod
    def access_path_to_field_name(cls, path: AccessPath, fmt: Formatter) -> str:
        """
        Convert an access path into a dereferencing field name, like the following examples:
            - `[0, "foo", 3, "bar"]` into `"->foo[3].bar"`
            - `[0, 3, "bar"]` into `"[0][3].bar"`
            - `[0, 1, 2]` into `"[0][1][2]"
            - `[0]` into `"[0]"`
        The path must have at least one element, and the first element must be an int.
        """
        cls.assert_valid_field_path(path)
        output = ""

        # Replace an initial "[0]." with "->"
        if len(path) >= 2 and path[0] == 0 and isinstance(path[1], str):
            output += f"->{path[1]}"
            path = path[2:]

        for p in path:
            if isinstance(p, str):
                output += f".{p}"
            elif isinstance(p, int):
                output += f"[{fmt.format_int(p)}]"
            else:
                static_assert_unreachable(p)
        return output

    def dependencies(self) -> List[Expression]:
        return [self.struct_var]

    def make_reference(self) -> Optional["StructAccess"]:
        field_path = self.late_field_path()
        if field_path and len(field_path) >= 2 and field_path[-1] == 0:
            return replace(self, field_path=field_path[:-1])
        return None

    def late_field_path(self) -> Optional[AccessPath]:
        # If we didn't have a type at the time when the struct access was
        # constructed, but now we do, compute field name.

        if self.field_path is None and not self.checked_late_field_path:
            var = late_unwrap(self.struct_var)
            field_path, field_type, _ = var.type.get_deref_field(
                self.offset, target_size=self.target_size
            )
            if field_path is not None:
                self.assert_valid_field_path(field_path)
                self.field_path = field_path
                self.type.unify(field_type)

            self.checked_late_field_path = True
        return self.field_path

    def late_has_known_type(self) -> bool:
        if self.late_field_path() is not None:
            return True
        assert (
            self.stack_info is not None
        ), "StructAccess must have stack_info if field_path isn't set"
        if self.offset == 0:
            var = late_unwrap(self.struct_var)
            if (
                not self.stack_info.has_nonzero_access(var)
                and isinstance(var, AddressOf)
                and isinstance(var.expr, GlobalSymbol)
                and var.expr.type_provided
            ):
                return True
        return False

    def format(self, fmt: Formatter) -> str:
        var = late_unwrap(self.struct_var)
        has_nonzero_access = False
        if self.stack_info is not None:
            has_nonzero_access = self.stack_info.has_nonzero_access(var)

        field_path = self.late_field_path()

        if field_path is not None and field_path != [0]:
            has_nonzero_access = True
        elif fmt.valid_syntax and (self.offset != 0 or has_nonzero_access):
            offset_str = fmt.format_int(self.offset)
            return f"MIPS2C_FIELD({var.format(fmt)}, {Type.ptr(self.type).format(fmt)}, {offset_str})"
        else:
            prefix = "unk" + ("_" if fmt.coding_style.unknown_underscore else "")
            field_path = [0, prefix + format_hex(self.offset)]
        field_name = self.access_path_to_field_name(field_path, fmt)

        # Rewrite `(&x)->y` to `x.y` by stripping `AddressOf` & setting deref=False
        deref = True
        if (
            isinstance(var, AddressOf)
            and not var.expr.type.is_array()
            and field_name.startswith("->")
        ):
            var = var.expr
            field_name = field_name.replace("->", ".", 1)
            deref = False

        # Rewrite `x->unk0` to `*x` and `x.unk0` to `x`, unless has_nonzero_access
        if self.offset == 0 and not has_nonzero_access:
            return f"{'*' if deref else ''}{var.format(fmt)}"

        return f"{parenthesize_for_struct_access(var, fmt)}{field_name}"


@dataclass(frozen=True, eq=True)
class ArrayAccess(Expression):
    # Represents ptr[index]. eq=True for symmetry with StructAccess.
    ptr: Expression
    index: Expression
    type: Type = field(compare=False)

    def dependencies(self) -> List[Expression]:
        return [self.ptr, self.index]

    def format(self, fmt: Formatter) -> str:
        base = parenthesize_for_struct_access(self.ptr, fmt)
        index = format_expr(self.index, fmt)
        return f"{base}[{index}]"


@dataclass(eq=False)
class GlobalSymbol(Expression):
    symbol_name: str
    type: Type
    asm_data_entry: Optional[AsmDataEntry] = None
    symbol_in_context: bool = False
    type_provided: bool = False
    initializer_in_typemap: bool = False
    demangled_str: Optional[str] = None

    def dependencies(self) -> List[Expression]:
        return []

    def is_string_constant(self) -> bool:
        ent = self.asm_data_entry
        if not ent or not ent.is_string:
            return False
        return len(ent.data) == 1 and isinstance(ent.data[0], bytes)

    def format_string_constant(self, fmt: Formatter) -> str:
        assert self.is_string_constant(), "checked by caller"
        assert self.asm_data_entry and isinstance(self.asm_data_entry.data[0], bytes)

        has_trailing_null = False
        data = self.asm_data_entry.data[0]
        while data and data[-1] == 0:
            data = data[:-1]
            has_trailing_null = True
        data = b"".join(map(escape_byte, data))

        strdata = data.decode("utf-8", "backslashreplace")
        ret = f'"{strdata}"'
        if not has_trailing_null:
            ret += " /* not null-terminated */"
        return ret

    def format(self, fmt: Formatter) -> str:
        return self.symbol_name

    def potential_array_dim(self, element_size: int) -> Tuple[int, int]:
        """
        Using the size of the symbol's `asm_data_entry` and a potential array element
        size, return the corresponding array dimension and number of "extra" bytes left
        at the end of the symbol's data.
        If the extra bytes are nonzero, then it's likely that `element_size` is incorrect.
        """
        # If we don't have the .data/.rodata entry for this symbol, we can't guess
        # its array dimension. Jump tables are ignored and not treated as arrays.
        if self.asm_data_entry is None or self.asm_data_entry.is_jtbl:
            return 0, element_size

        min_data_size, max_data_size = self.asm_data_entry.size_range_bytes()
        if element_size > max_data_size:
            # The type is too big for the data (not an array)
            return 0, max_data_size

        # Check if it's possible that this symbol is not an array, and is just 1 element
        if min_data_size <= element_size <= max_data_size and not self.type.is_array():
            return 1, 0

        array_dim, extra_bytes = divmod(min_data_size, element_size)
        if extra_bytes != 0:
            # If it's not possible to make an exact multiple of element_size by incorporating
            # bytes from the padding, then indicate that in the return value.
            padding_bytes = element_size - extra_bytes
            if min_data_size + padding_bytes > max_data_size:
                return array_dim, extra_bytes

        # Include potential padding in the array. Although this is unlikely to match the original C,
        # it's much easier to manually remove all or some of these elements than to add them back in.
        return max_data_size // element_size, 0


@dataclass(frozen=True, eq=True)
class Literal(Expression):
    value: int
    type: Type = field(compare=False, default_factory=Type.any)
    elide_cast: bool = field(compare=False, default=False)

    def dependencies(self) -> List[Expression]:
        return []

    def format(self, fmt: Formatter, force_dec: bool = False) -> str:
        if self.type.is_likely_float():
            if self.type.get_size_bits() == 64:
                return format_f64_imm(self.value)
            else:
                return format_f32_imm(self.value) + "f"
        if self.type.is_pointer() and self.value == 0:
            return "NULL"

        prefix = ""
        suffix = ""
        if not fmt.skip_casts and not self.elide_cast:
            if self.type.is_pointer():
                prefix = f"({self.type.format(fmt)})"
            if self.type.is_unsigned():
                suffix = "U"

        if force_dec:
            value = str(self.value)
        else:
            size_bits = self.type.get_size_bits()
            v = self.value

            # The top 2 bits are tested rather than just the sign bit
            # to help prevent N64 VRAM pointers (0x80000000+) turning negative
            if (
                self.type.is_signed()
                and size_bits
                and v & (1 << (size_bits - 1))
                and v > (3 << (size_bits - 2))
                and v < 2 ** size_bits
            ):
                v -= 1 << size_bits
            value = fmt.format_int(v, size_bits=size_bits)

        return prefix + value + suffix

    def likely_partial_offset(self) -> bool:
        return self.value % 2 ** 15 in (0, 2 ** 15 - 1) and self.value < 0x1000000


@dataclass(frozen=True, eq=True)
class AddressOf(Expression):
    expr: Expression
    type: Type = field(compare=False, default_factory=Type.ptr)

    def dependencies(self) -> List[Expression]:
        return [self.expr]

    def format(self, fmt: Formatter) -> str:
        if isinstance(self.expr, GlobalSymbol):
            if self.expr.is_string_constant():
                return self.expr.format_string_constant(fmt)
        if self.expr.type.is_array():
            return f"{self.expr.format(fmt)}"
        if self.expr.type.is_function():
            # Functions are automatically converted to function pointers
            # without an explicit `&` by the compiler
            return f"{self.expr.format(fmt)}"
        if isinstance(self.expr, StructAccess):
            # Simplify `&x[0]` into `x`
            ref = self.expr.make_reference()
            if ref:
                return f"{ref.format(fmt)}"
        return f"&{self.expr.format(fmt)}"


@dataclass(frozen=True)
class Lwl(Expression):
    load_expr: Expression
    key: Tuple[int, object]
    type: Type = field(compare=False, default_factory=Type.any_reg)

    def dependencies(self) -> List[Expression]:
        return [self.load_expr]

    def format(self, fmt: Formatter) -> str:
        return f"MIPS2C_LWL({self.load_expr.format(fmt)})"


@dataclass(frozen=True)
class Load3Bytes(Expression):
    load_expr: Expression
    type: Type = field(compare=False, default_factory=Type.any_reg)

    def dependencies(self) -> List[Expression]:
        return [self.load_expr]

    def format(self, fmt: Formatter) -> str:
        if fmt.valid_syntax:
            return f"MIPS2C_FIRST3BYTES({self.load_expr.format(fmt)})"
        return f"(first 3 bytes) {self.load_expr.format(fmt)}"


@dataclass(frozen=True)
class UnalignedLoad(Expression):
    load_expr: Expression
    type: Type = field(compare=False, default_factory=Type.any_reg)

    def dependencies(self) -> List[Expression]:
        return [self.load_expr]

    def format(self, fmt: Formatter) -> str:
        if fmt.valid_syntax:
            return f"MIPS2C_UNALIGNED32({self.load_expr.format(fmt)})"
        return f"(unaligned s32) {self.load_expr.format(fmt)}"


@dataclass(frozen=False, eq=False)
class EvalOnceExpr(Expression):
    wrapped_expr: Expression
    var: Var
    type: Type

    # True for function calls/errors
    emit_exactly_once: bool

    # Mutable state:

    # True if this EvalOnceExpr should be totally transparent and not emit a variable,
    # It may dynamically change from true to false due to forced emissions.
    # Initially, it is based on is_trivial_expression.
    trivial: bool

    # True if this EvalOnceExpr must emit a variable (see RegMeta.force)
    forced_emit: bool = False

    # The number of expressions that depend on this EvalOnceExpr; we emit a variable
    # if this is > 1.
    num_usages: int = 0

    def dependencies(self) -> List[Expression]:
        # (this is a bit iffy since state can change over time, but improves uses_expr)
        if self.need_decl():
            return []
        return [self.wrapped_expr]

    def use(self) -> None:
        self.num_usages += 1
        if self.trivial or (self.num_usages == 1 and not self.emit_exactly_once):
            self.wrapped_expr.use()

    def force(self) -> None:
        # Transition to non-trivial, and mark as used multiple times to force a var.
        # TODO: If it was originally trivial, we may previously have marked its
        # wrappee used multiple times, even though we now know that it should
        # have been marked just once... We could fix that by moving marking of
        # trivial EvalOnceExpr's to the very end. At least the consequences of
        # getting this wrong are pretty mild -- it just causes extraneous var
        # emission in rare cases.
        self.trivial = False
        self.forced_emit = True
        self.use()
        self.use()

    def need_decl(self) -> bool:
        return self.num_usages > 1 and not self.trivial

    def format(self, fmt: Formatter) -> str:
        if not self.need_decl():
            return self.wrapped_expr.format(fmt)
        else:
            return self.var.format(fmt)


@dataclass(frozen=False, eq=False)
class PhiExpr(Expression):
    reg: Register
    node: Node
    type: Type
    used_phis: List["PhiExpr"]
    name: Optional[str] = None
    num_usages: int = 0
    replacement_expr: Optional[Expression] = None
    used_by: Optional["PhiExpr"] = None

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
        if self.used_by is None or self.replacement_expr is not None:
            return self
        return self.used_by.propagates_to()

    def format(self, fmt: Formatter) -> str:
        if self.replacement_expr:
            return self.replacement_expr.format(fmt)
        return self.get_var_name()


@dataclass
class SwitchControl:
    control_expr: Expression
    jump_table: Optional[GlobalSymbol] = None
    offset: int = 0
    is_irregular: bool = False

    def matches_guard_condition(self, cond: Condition) -> bool:
        """
        Return True if `cond` is one of:
            - `((control_expr + (-offset)) >= len(jump_table))`, if `offset != 0`
            - `(control_expr >= len(jump_table))`, if `offset == 0`
        These are the appropriate bounds checks before using `jump_table`.
        """
        cmp_expr = simplify_condition(cond)
        if not isinstance(cmp_expr, BinaryOp) or cmp_expr.op not in (">=", ">"):
            return False
        cmp_exclusive = cmp_expr.op == ">"

        # The LHS may have been wrapped in a u32 cast
        left_expr = late_unwrap(cmp_expr.left)
        if isinstance(left_expr, Cast):
            left_expr = late_unwrap(left_expr.expr)

        if self.offset != 0:
            if (
                not isinstance(left_expr, BinaryOp)
                or late_unwrap(left_expr.left) != late_unwrap(self.control_expr)
                or left_expr.op != "+"
                or late_unwrap(left_expr.right) != Literal(-self.offset)
            ):
                return False
        elif left_expr != late_unwrap(self.control_expr):
            return False

        right_expr = late_unwrap(cmp_expr.right)
        if (
            self.jump_table is None
            or self.jump_table.asm_data_entry is None
            or not self.jump_table.asm_data_entry.is_jtbl
            or not isinstance(right_expr, Literal)
        ):
            return False

        # Count the number of labels (exclude padding bytes)
        jump_table_len = sum(
            isinstance(e, str) for e in self.jump_table.asm_data_entry.data
        )
        return right_expr.value + int(cmp_exclusive) == jump_table_len

    @staticmethod
    def irregular_from_expr(control_expr: Expression) -> "SwitchControl":
        """
        Return a SwitchControl representing a "irregular" switch statement.
        The switch does not have a single jump table; instead it is a series of
        if statements & other switches.
        """
        return SwitchControl(
            control_expr=control_expr,
            jump_table=None,
            offset=0,
            is_irregular=True,
        )

    @staticmethod
    def from_expr(expr: Expression) -> "SwitchControl":
        """
        Try to convert `expr` into a SwitchControl from one of the following forms:
            - `*(&jump_table + (control_expr * 4))`
            - `*(&jump_table + ((control_expr + (-offset)) * 4))`
        If `offset` is not present, it defaults to 0.
        If `expr` does not match, return a thin wrapper around the input expression,
        with `jump_table` set to `None`.
        """
        # The "error" expression we use if we aren't able to parse `expr`
        error_expr = SwitchControl(expr)

        # Match `*(&jump_table + (control_expr * 4))`
        struct_expr = early_unwrap(expr)
        if not isinstance(struct_expr, StructAccess) or struct_expr.offset != 0:
            return error_expr
        add_expr = early_unwrap(struct_expr.struct_var)
        if not isinstance(add_expr, BinaryOp) or add_expr.op != "+":
            return error_expr

        # Check for either `*(&jump_table + (control_expr * 4))` and `*((control_expr * 4) + &jump_table)`
        left_expr, right_expr = early_unwrap(add_expr.left), early_unwrap(
            add_expr.right
        )
        if isinstance(left_expr, AddressOf) and isinstance(
            left_expr.expr, GlobalSymbol
        ):
            jtbl_addr_expr, mul_expr = left_expr, right_expr
        elif isinstance(right_expr, AddressOf) and isinstance(
            right_expr.expr, GlobalSymbol
        ):
            mul_expr, jtbl_addr_expr = left_expr, right_expr
        else:
            return error_expr

        jump_table = jtbl_addr_expr.expr
        assert isinstance(jump_table, GlobalSymbol)
        if (
            not isinstance(mul_expr, BinaryOp)
            or mul_expr.op != "*"
            or early_unwrap(mul_expr.right) != Literal(4)
        ):
            return error_expr
        control_expr = mul_expr.left

        # Optionally match `control_expr + (-offset)`
        offset = 0
        uw_control_expr = early_unwrap(control_expr)
        if isinstance(uw_control_expr, BinaryOp) and uw_control_expr.op == "+":
            offset_lit = early_unwrap(uw_control_expr.right)
            if isinstance(offset_lit, Literal):
                control_expr = uw_control_expr.left
                offset = -offset_lit.value

        # Check that it is really a jump table
        if jump_table.asm_data_entry is None or not jump_table.asm_data_entry.is_jtbl:
            return error_expr

        return SwitchControl(control_expr, jump_table, offset)


@dataclass
class EvalOnceStmt(Statement):
    expr: EvalOnceExpr

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


@dataclass
class SetPhiStmt(Statement):
    phi: PhiExpr
    expr: Expression

    def should_write(self) -> bool:
        expr = self.expr
        if isinstance(expr, PhiExpr) and expr.propagates_to() != expr:
            # When we have phi1 = phi2, and phi2 is only used in this place,
            # the SetPhiStmt for phi2 will store directly to phi1 and we can
            # skip this store.
            assert expr.propagates_to() == self.phi.propagates_to()
            return False
        if late_unwrap(expr) == self.phi.propagates_to():
            # Elide "phi = phi".
            return False
        return True

    def format(self, fmt: Formatter) -> str:
        return format_assignment(self.phi.propagates_to(), self.expr, fmt)


@dataclass
class ExprStmt(Statement):
    expr: Expression

    def should_write(self) -> bool:
        return True

    def format(self, fmt: Formatter) -> str:
        return f"{format_expr(self.expr, fmt)};"


@dataclass
class StoreStmt(Statement):
    source: Expression
    dest: Expression

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


@dataclass
class CommentStmt(Statement):
    contents: str

    def should_write(self) -> bool:
        return True

    def format(self, fmt: Formatter) -> str:
        return f"// {self.contents}"


@dataclass(frozen=True)
class AddressMode:
    offset: int
    rhs: Register

    def __str__(self) -> str:
        if self.offset:
            return f"{self.offset}({self.rhs})"
        else:
            return f"({self.rhs})"


@dataclass(frozen=True)
class RawSymbolRef:
    offset: int
    sym: AsmGlobalSymbol

    def __str__(self) -> str:
        if self.offset:
            return f"{self.sym.symbol_name} + {self.offset}"
        else:
            return self.sym.symbol_name


@dataclass
class RegMeta:
    # True if this regdata is unchanged from the start of the block
    inherited: bool = False

    # True if this regdata is read by some later node
    is_read: bool = False

    # True if the value derives solely from function call return values
    function_return: bool = False

    # True if the value derives solely from regdata's with is_read = True,
    # function_return = True, or is a passed in argument
    uninteresting: bool = False

    # True if the regdata must be replaced by variable if it is ever read
    force: bool = False


@dataclass
class RegData:
    value: Expression
    meta: RegMeta


@dataclass
class RegInfo:
    stack_info: StackInfo = field(repr=False)
    contents: Dict[Register, RegData] = field(default_factory=dict)
    read_inherited: Set[Register] = field(default_factory=set)
    _active_instr: Optional[Instruction] = None

    def __getitem__(self, key: Register) -> Expression:
        if self._active_instr is not None and key not in self._active_instr.inputs:
            lineno = self._active_instr.meta.lineno
            return ErrorExpr(f"Read from unset register {key} on line {lineno}")
        if key == Register("zero"):
            return Literal(0)
        data = self.contents.get(key)
        if data is None:
            return ErrorExpr(f"Read from unset register {key}")
        ret = data.value
        data.meta.is_read = True
        if data.meta.inherited:
            self.read_inherited.add(key)
        if isinstance(ret, PassedInArg) and not ret.copied:
            # Create a new argument object to better distinguish arguments we
            # are called with from arguments passed to subroutines. Also, unify
            # the argument's type with what we can guess from the register used.
            val, arg = self.stack_info.get_argument(ret.value)
            self.stack_info.add_argument(arg)
            val.type.unify(ret.type)
            return val
        if data.meta.force:
            assert isinstance(ret, EvalOnceExpr)
            ret.force()
        return ret

    def __contains__(self, key: Register) -> bool:
        return key in self.contents

    def __setitem__(self, key: Register, value: Expression) -> None:
        self.set_with_meta(key, value, RegMeta())

    def set_with_meta(self, key: Register, value: Expression, meta: RegMeta) -> None:
        if self._active_instr is not None and key not in self._active_instr.outputs:
            raise DecompFailure(f"Undeclared write to {key} in {self._active_instr}")
        self.unchecked_set_with_meta(key, value, meta)

    def unchecked_set_with_meta(
        self, key: Register, value: Expression, meta: RegMeta
    ) -> None:
        assert key != Register("zero")
        self.contents[key] = RegData(value, meta)

    def __delitem__(self, key: Register) -> None:
        assert key != Register("zero")
        del self.contents[key]

    def get_raw(self, key: Register) -> Optional[Expression]:
        data = self.contents.get(key)
        return data.value if data is not None else None

    def get_meta(self, key: Register) -> Optional[RegMeta]:
        data = self.contents.get(key)
        return data.meta if data is not None else None

    @contextmanager
    def current_instr(self, instr: Instruction) -> Iterator[None]:
        self._active_instr = instr
        try:
            with current_instr(instr):
                yield
        finally:
            self._active_instr = None

    def __str__(self) -> str:
        return ", ".join(
            f"{k}: {v.value}"
            for k, v in sorted(self.contents.items(), key=lambda x: x[0].register_name)
            if not self.stack_info.should_save(v.value, None)
        )


@dataclass
class BlockInfo:
    """
    Contains translated assembly code (to_write), the block's branch condition,
    and block's final register states.
    """

    to_write: List[Statement]
    return_value: Optional[Expression]
    switch_control: Optional[SwitchControl]
    branch_condition: Optional[Condition]
    final_register_states: RegInfo
    has_function_call: bool

    def __str__(self) -> str:
        newline = "\n\t"
        return "\n".join(
            [
                f"Statements: {newline.join(str(w) for w in self.statements_to_write())}",
                f"Branch condition: {self.branch_condition}",
                f"Final register states: {self.final_register_states}",
            ]
        )

    def statements_to_write(self) -> List[Statement]:
        return [st for st in self.to_write if st.should_write()]


def get_block_info(node: Node) -> BlockInfo:
    ret = node.block.block_info
    assert isinstance(ret, BlockInfo)
    return ret


@dataclass
class InstrArgs:
    raw_args: List[Argument]
    regs: RegInfo = field(repr=False)
    stack_info: StackInfo = field(repr=False)

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

    def imm_value(self, index: int) -> int:
        arg = self.full_imm(index)
        assert isinstance(arg, Literal)
        return arg.value

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

        # PPC: FPR's hold doubles (64 bits), so we don't need to do anything special
        if self.stack_info.global_info.arch.arch == Target.ArchEnum.PPC:
            return ret

        # MIPS: Look at the paired FPR to get the full 64-bit value
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

    def cmp_reg(self, key: str) -> Condition:
        cond = self.regs[Register(key)]
        if not isinstance(cond, Condition):
            cond = BinaryOp.icmp(cond, "!=", Literal(0))
        return cond

    def full_imm(self, index: int) -> Expression:
        arg = strip_macros(self.raw_arg(index))
        ret = literal_expr(arg, self.stack_info)
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
        if not isinstance(arg, Macro) or arg.macro_name not in ("hi", "ha", "h"):
            raise DecompFailure(
                f"Got lui/lis instruction with macro other than %hi/@ha/@h: {arg}"
            )
        return arg.argument

    def shifted_imm(self, index: int) -> Expression:
        # TODO: Should this be part of hi_imm? Do we need to handle @ha?
        raw_imm = self.unsigned_imm(index)
        assert isinstance(raw_imm, Literal)
        return Literal(raw_imm.value << 16)

    def memory_ref(self, index: int) -> Union[AddressMode, RawSymbolRef]:
        ret = strip_macros(self.raw_arg(index))

        # In MIPS, we want to allow "lw $v0, symbol + 4", which is outputted by
        # some disassemblers (like IDA) even though it isn't valid assembly.
        # For PPC, we want to allow "lwz $r1, symbol@sda21($r13)" where $r13 is
        # assumed to point to the start of a small data area (SDA).
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
        if not isinstance(ret.lhs, AsmLiteral):
            raise DecompFailure(
                f"Unable to parse offset for instruction argument {ret}. "
                "Expected a constant or a %lo macro."
            )
        return AddressMode(offset=ret.lhs.signed_value(), rhs=ret.rhs)

    def count(self) -> int:
        return len(self.raw_args)


def deref(
    arg: Union[AddressMode, RawSymbolRef, Expression],
    regs: RegInfo,
    stack_info: StackInfo,
    *,
    size: int,
    store: bool = False,
) -> Expression:
    if isinstance(arg, Expression):
        offset = 0
        var = arg
    elif isinstance(arg, AddressMode):
        offset = arg.offset
        if stack_info.is_stack_reg(arg.rhs):
            return stack_info.get_stack_var(offset, store=store)
        var = regs[arg.rhs]
    else:
        offset = arg.offset
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
            if isinstance(addend, Literal) and addend.likely_partial_offset():
                offset += addend.value
                var = base
                uw_var = early_unwrap(var)
                break

    var.type.unify(Type.ptr())
    stack_info.record_struct_access(var, offset)
    field_name: Optional[str] = None
    type: Type = stack_info.unique_type_for("struct", (uw_var, offset), Type.any())

    # Struct access with type information.
    array_expr = array_access_from_add(
        var, offset, stack_info, target_size=size, ptr=False
    )
    if array_expr is not None:
        return array_expr
    field_path, field_type, _ = var.type.get_deref_field(offset, target_size=size)
    if field_path is not None:
        field_type.unify(type)
        type = field_type
    else:
        field_path = None

    return StructAccess(
        struct_var=var,
        offset=offset,
        target_size=size,
        field_path=field_path,
        stack_info=stack_info,
        type=type,
    )


def is_trivial_expression(expr: Expression) -> bool:
    # Determine whether an expression should be evaluated only once or not.
    if isinstance(
        expr,
        (
            EvalOnceExpr,
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
        return all(is_trivial_expression(e) for e in expr.dependencies())
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
            AddressOf,
            LocalVar,
            PhiExpr,
            PassedInArg,
            RegisterVar,
            FuncCall,
        ),
    ):
        return True
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
    if isinstance(expr, UnaryOp):
        inner = simplify_condition(expr.expr)
        if expr.op == "!" and isinstance(inner, Condition):
            return inner.negated()
        return UnaryOp(expr=inner, op=expr.op, type=expr.type)
    if isinstance(expr, BinaryOp):
        left = simplify_condition(expr.left)
        right = simplify_condition(expr.right)
        if isinstance(left, BinaryOp) and left.is_comparison() and right == Literal(0):
            if expr.op == "==":
                return simplify_condition(left.negated())
            if expr.op == "!=":
                return left
        if (
            expr.is_comparison()
            and isinstance(left, Literal)
            and not isinstance(right, Literal)
        ):
            return BinaryOp(
                left=right,
                op=expr.op.translate(str.maketrans("<>", "><")),
                right=left,
                type=expr.type,
            )
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
    s = expr.format(fmt)
    if (
        s.startswith("*")
        or s.startswith("&")
        or (isinstance(expr, Cast) and expr.needed_for_store())
    ):
        return f"({s})"
    return s


def elide_casts_for_store(expr: Expression) -> Expression:
    uw_expr = late_unwrap(expr)
    if isinstance(uw_expr, Cast) and not uw_expr.needed_for_store():
        return elide_casts_for_store(uw_expr.expr)
    if isinstance(uw_expr, Literal) and uw_expr.type.is_int():
        # Avoid suffixes for unsigned ints
        return replace(uw_expr, elide_cast=True)
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
    Unwrap EvalOnceExpr's, stopping at variable boundaries.

    This function may produce wrong results while code is being generated,
    since at that point we don't know the final status of EvalOnceExpr's.
    """
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
    """
    if (
        isinstance(expr, EvalOnceExpr)
        and not expr.forced_emit
        and not expr.emit_exactly_once
    ):
        return early_unwrap(expr.wrapped_expr)
    return expr


def early_unwrap_ints(expr: Expression) -> Expression:
    """
    Unwrap EvalOnceExpr's, even past variable boundaries or through int Cast's
    This is a bit sketchier than early_unwrap(), but can be used for pattern matching.
    """
    uw_expr = early_unwrap(expr)
    if isinstance(uw_expr, Cast) and uw_expr.reinterpret and uw_expr.type.is_int():
        return early_unwrap_ints(uw_expr.expr)
    return uw_expr


def unwrap_deep(expr: Expression) -> Expression:
    """
    Unwrap EvalOnceExpr's, even past variable boundaries.

    This is generally a sketchy thing to do, try to avoid it. In particular:
    - the returned expression is not usable for emission, because it may contain
      accesses at an earlier point in time or an expression that should not be repeated.
    - just because unwrap_deep(a) == unwrap_deep(b) doesn't mean a and b are
      interchangable, because they may be computed in different places.
    """
    if isinstance(expr, EvalOnceExpr):
        return unwrap_deep(expr.wrapped_expr)
    return expr


def literal_expr(arg: Argument, stack_info: StackInfo) -> Expression:
    if isinstance(arg, AsmGlobalSymbol):
        return stack_info.global_info.address_of_gsym(arg.symbol_name)
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


def void_fn_op(fn_name: str, args: List[Expression]) -> ExprStmt:
    fn_call = fn_op(fn_name, args, Type.any_reg())
    fn_call.use()
    return ExprStmt(fn_call)


def load_upper(args: InstrArgs) -> Expression:
    arg = args.raw_arg(1)
    if not isinstance(arg, Macro):
        assert not isinstance(
            arg, Literal
        ), "normalize_instruction should convert lui/lis <literal> to li"
        raise DecompFailure(
            f"lui/lis argument must be a literal or %hi/@ha macro, found {arg}"
        )

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
        raise DecompFailure(f"Invalid %hi/@ha argument {hi_arg}")

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


def handle_or(left: Expression, right: Expression) -> Expression:
    if left == right:
        # `or $rD, $rS, $rS` can be used to move $rS into $rD
        return left

    if isinstance(left, Literal) and isinstance(right, Literal):
        if (((left.value & 0xFFFF) == 0 and (right.value & 0xFFFF0000) == 0)) or (
            (right.value & 0xFFFF) == 0 and (left.value & 0xFFFF0000) == 0
        ):
            return Literal(value=(left.value | right.value))
    # Regular bitwise OR.
    return BinaryOp.int(left=left, op="|", right=right)


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

    # `(x + 0xEDCC)` is emitted as `((x + 0x10000) - 0x1234)`,
    # i.e. as an `addis` followed by an `addi`
    uw_source = early_unwrap(source)
    if (
        isinstance(uw_source, BinaryOp)
        and uw_source.op == "+"
        and isinstance(uw_source.right, Literal)
        and uw_source.right.value % 0x10000 == 0
        and isinstance(imm, Literal)
    ):
        return add_imm(
            uw_source.left, Literal(imm.value + uw_source.right.value), stack_info
        )
    return handle_addi_real(args.reg_ref(0), source_reg, source, imm, stack_info)


def handle_addis(args: InstrArgs) -> Expression:
    stack_info = args.stack_info
    source_reg = args.reg_ref(1)
    source = args.reg(1)
    imm = args.shifted_imm(2)
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
        return AddressOf(var, type=var.type.reference())
    else:
        return add_imm(source, imm, stack_info)


def add_imm(source: Expression, imm: Expression, stack_info: StackInfo) -> Expression:
    if imm == Literal(0):
        # addiu $reg1, $reg2, 0 is a move
        # (this happens when replacing %lo(...) by 0)
        return source
    elif source.type.is_pointer_or_array():
        # Pointer addition (this may miss some pointers that get detected later;
        # unfortunately that's hard to do anything about with mips_to_c's single-pass
        # architecture).
        if isinstance(imm, Literal) and not imm.likely_partial_offset():
            array_access = array_access_from_add(
                source, imm.value, stack_info, target_size=None, ptr=True
            )
            if array_access is not None:
                return array_access

            field_path, field_type, _ = source.type.get_deref_field(
                imm.value, target_size=None
            )
            if field_path is not None:
                return AddressOf(
                    StructAccess(
                        struct_var=source,
                        offset=imm.value,
                        target_size=None,
                        field_path=field_path,
                        stack_info=stack_info,
                        type=field_type,
                    ),
                    type=field_type.reference(),
                )
        if isinstance(imm, Literal):
            target = source.type.get_pointer_target()
            if target:
                target_size = target.get_size_bytes()
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
    size = type.get_size_bytes()
    assert size is not None
    expr = deref(args.memory_ref(1), args.regs, args.stack_info, size=size)

    # Detect rodata constants
    if isinstance(expr, StructAccess) and expr.offset == 0:
        target = early_unwrap(expr.struct_var)
        if (
            isinstance(target, AddressOf)
            and isinstance(target.expr, GlobalSymbol)
            and type.is_likely_float()
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
    # Unaligned load for the left part of a register (lwl can technically merge with
    # a pre-existing lwr, but doesn't in practice, so we treat this as a standard
    # destination-first operation)
    ref = args.memory_ref(1)
    expr = deref_unaligned(ref, args.regs, args.stack_info)
    key: Tuple[int, object]
    if isinstance(ref, AddressMode):
        key = (ref.offset, args.regs[ref.rhs])
    else:
        key = (ref.offset, ref.sym)
    return Lwl(expr, key)


def handle_lwr(args: InstrArgs) -> Expression:
    # Unaligned load for the right part of a register. This lwr may merge with an
    # existing lwl, if it loads from the same target but with an offset that's +3.
    uw_old_value = early_unwrap(args.reg(0))
    ref = args.memory_ref(1)
    lwl_key: Tuple[int, object]
    if isinstance(ref, AddressMode):
        lwl_key = (ref.offset - 3, args.regs[ref.rhs])
    else:
        lwl_key = (ref.offset - 3, ref.sym)
    if isinstance(uw_old_value, Lwl) and uw_old_value.key[0] == lwl_key[0]:
        return UnalignedLoad(uw_old_value.load_expr)
    if ref.offset % 4 == 2:
        left_mem_ref = replace(ref, offset=ref.offset - 2)
        load_expr = deref_unaligned(left_mem_ref, args.regs, args.stack_info)
        return Load3Bytes(load_expr)
    return ErrorExpr("Unable to handle lwr; missing a corresponding lwl")


def make_store(args: InstrArgs, type: Type) -> Optional[StoreStmt]:
    size = type.get_size_bytes()
    assert size is not None
    stack_info = args.stack_info
    source_reg = args.reg_ref(0)
    source_raw = args.regs.get_raw(source_reg)
    if type.is_likely_float() and size == 8:
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


def make_storex(args: InstrArgs, type: Type) -> Optional[StoreStmt]:
    # "indexed stores" like `stwx rS, rA, rB` write `rS` into `(rA + rB)`
    size = type.get_size_bytes()
    assert size is not None

    source = args.reg(0)
    ptr = BinaryOp.intptr(left=args.reg(1), op="+", right=args.reg(2))

    # TODO: Can we assume storex's are never used to save registers to the stack?
    dest = deref(ptr, args.regs, args.stack_info, size=size, store=True)
    dest.type.unify(type)
    return StoreStmt(source=as_type(source, type, silent=False), dest=dest)


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
    real_target = replace(target, offset=target.offset - 2)
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
            elif expr.op == "<<" and rhs > shift.value:
                new_shift = fold_mul_chains(
                    BinaryOp.int(expr.left, "<<", Literal(rhs - shift.value))
                )
                return as_type(new_shift, tp, silent=False)
            elif expr.op == "*" and rhs % pow2 == 0 and rhs != pow2:
                mul = BinaryOp.int(expr.left, "*", Literal(value=rhs // pow2))
                return as_type(mul, tp, silent=False)
    return fold_divmod(BinaryOp(as_s32(lhs), ">>", as_intish(shift), type=Type.s32()))


def handle_conditional_move(args: InstrArgs, nonzero: bool) -> Expression:
    op = "!=" if nonzero else "=="
    type = Type.any_reg()
    return TernaryOp(
        BinaryOp.scmp(args.reg(2), op, Literal(0)),
        as_type(args.reg(1), type, silent=True),
        as_type(args.reg(0), type, silent=True),
        type,
    )


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


def fold_divmod(original_expr: BinaryOp) -> BinaryOp:
    """
    Return a new BinaryOp instance if this one can be simplified to a single / or % op.
    This involves simplifying expressions using MULT_HI, MULTU_HI, +, -, <<, >>, and /.

    In GCC 2.7.2, the code that generates these instructions is in expmed.c.

    See also https://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html
    for a modern writeup of a similar algorithm.

    This optimization is also used by MWCC and modern compilers (but not IDO).
    """
    mult_high_ops = ("MULT_HI", "MULTU_HI")
    possible_match_ops = mult_high_ops + ("-", "+", ">>")

    # Only operate on integer expressions of certain operations
    if original_expr.is_floating() or original_expr.op not in possible_match_ops:
        return original_expr

    # Use `early_unwrap_ints` instead of `early_unwrap` to ignore Casts to integer types
    # Although this discards some extra type information, this function largely ignores
    # sign/size information to stay simpler. The result will be made with BinaryOp.int()
    # regardless of input types.
    expr = original_expr
    left_expr = early_unwrap_ints(expr.left)
    right_expr = early_unwrap_ints(expr.right)
    divisor_shift = 0

    # Detect signed power-of-two division: (x >> N) + MIPS2C_CARRY --> x / (1 << N)
    if (
        isinstance(left_expr, BinaryOp)
        and left_expr.op == ">>"
        and isinstance(left_expr.right, Literal)
        and expr.op == "+"
        and isinstance(right_expr, CarryBit)
    ):
        new_denom = 1 << left_expr.right.value
        return BinaryOp.s32(
            left=left_expr.left,
            op="/",
            right=Literal(new_denom),
            silent=True,
        )

    # Fold `/` with `>>`: ((x / N) >> M) --> x / (N << M)
    # NB: If x is signed, this is only correct if there is a sign-correcting subtraction term
    if (
        isinstance(left_expr, BinaryOp)
        and left_expr.op == "/"
        and isinstance(left_expr.right, Literal)
        and expr.op == ">>"
        and isinstance(right_expr, Literal)
    ):
        new_denom = left_expr.right.value << right_expr.value
        if new_denom < (1 << 32):
            return BinaryOp.int(
                left=left_expr.left,
                op="/",
                right=Literal(new_denom),
            )

    # Detect `%`: (x - ((x / N) * N)) --> x % N
    if expr.op == "-" and isinstance(right_expr, BinaryOp) and right_expr.op == "*":
        div_expr = early_unwrap_ints(right_expr.left)
        mod_base = early_unwrap_ints(right_expr.right)
        if (
            isinstance(div_expr, BinaryOp)
            and early_unwrap_ints(div_expr.left) == left_expr
            and isinstance(mod_base, Literal)
        ):
            # Accept either `(x / N) * N` or `(x >> N) * M` (where `1 << N == M`)
            divisor = early_unwrap_ints(div_expr.right)
            if (div_expr.op == "/" and divisor == mod_base) or (
                div_expr.op == ">>"
                and isinstance(divisor, Literal)
                and (1 << divisor.value) == mod_base.value
            ):
                return BinaryOp.int(left=left_expr, op="%", right=right_expr.right)

    # Detect dividing by a negative: ((x >> 31) - (x / N)) --> x / -N
    if (
        expr.op == "-"
        and isinstance(left_expr, BinaryOp)
        and left_expr.op == ">>"
        and early_unwrap_ints(left_expr.right) == Literal(31)
        and isinstance(right_expr, BinaryOp)
        and right_expr.op == "/"
        and isinstance(right_expr.right, Literal)
    ):
        # Swap left_expr & right_expr, but replace the N in right_expr with -N
        left_expr, right_expr = (
            replace(right_expr, right=Literal(-right_expr.right.value)),
            left_expr,
        )

    # Remove outer error term: ((x / N) + ((x / N) >> 31)) --> x / N
    # As N gets close to (1 << 30), this is no longer a negligible error term
    if (
        expr.op == "+"
        and isinstance(left_expr, BinaryOp)
        and left_expr.op == "/"
        and isinstance(left_expr.right, Literal)
        and left_expr.right.value <= (1 << 29)
        and isinstance(right_expr, BinaryOp)
        and early_unwrap_ints(right_expr.left) == left_expr
        and right_expr.op == ">>"
        and early_unwrap_ints(right_expr.right) == Literal(31)
    ):
        return left_expr

    # Remove outer error term: ((x / N) - (x >> 31)) --> x / N
    if (
        expr.op == "-"
        and isinstance(left_expr, BinaryOp)
        and left_expr.op == "/"
        and isinstance(left_expr.right, Literal)
        and isinstance(right_expr, BinaryOp)
        and right_expr.op == ">>"
        and early_unwrap_ints(right_expr.right) == Literal(31)
    ):
        div_expr = left_expr
        shift_var_expr = early_unwrap_ints(right_expr.left)
        div_var_expr = early_unwrap_ints(div_expr.left)
        # Check if the LHS of the shift is the same var that we're dividing by
        if div_var_expr == shift_var_expr:
            if isinstance(div_expr.right, Literal) and div_expr.right.value >= (
                1 << 30
            ):
                return BinaryOp.int(
                    left=div_expr.left,
                    op=div_expr.op,
                    right=div_expr.right,
                )
            return div_expr
        # If the var is under 32 bits, the error term may look like `(x << K) >> 31` instead
        if (
            isinstance(shift_var_expr, BinaryOp)
            and early_unwrap_ints(div_expr.left)
            == early_unwrap_ints(shift_var_expr.left)
            and shift_var_expr.op == "<<"
            and isinstance(shift_var_expr.right, Literal)
        ):
            return div_expr

    # Shift on the result of the mul: MULT_HI(x, N) >> M, shift the divisor by M
    if (
        isinstance(left_expr, BinaryOp)
        and expr.op == ">>"
        and isinstance(right_expr, Literal)
    ):
        divisor_shift += right_expr.value
        expr = left_expr
        left_expr = early_unwrap_ints(expr.left)
        right_expr = early_unwrap_ints(expr.right)
        # Normalize MULT_HI(N, x) to MULT_HI(x, N)
        if isinstance(left_expr, Literal) and not isinstance(right_expr, Literal):
            left_expr, right_expr = right_expr, left_expr

        # Remove inner addition: (MULT_HI(x, N) + x) >> M --> MULT_HI(x, N) >> M
        # MULT_HI performs signed multiplication, so the `+ x` acts as setting the 32nd bit
        # while having a result with the same sign as x.
        # We can ignore it because `round_div` can work with arbitrarily large constants
        if (
            isinstance(left_expr, BinaryOp)
            and left_expr.op == "MULT_HI"
            and expr.op == "+"
            and early_unwrap_ints(left_expr.left) == right_expr
        ):
            expr = left_expr
            left_expr = early_unwrap_ints(expr.left)
            right_expr = early_unwrap_ints(expr.right)

    # Shift on the LHS of the mul: MULT_HI(x >> M, N) --> MULT_HI(x, N) >> M
    if (
        expr.op in mult_high_ops
        and isinstance(left_expr, BinaryOp)
        and left_expr.op == ">>"
        and isinstance(left_expr.right, Literal)
    ):
        divisor_shift += left_expr.right.value
        left_expr = early_unwrap_ints(left_expr.left)

    # Instead of checking for the error term precisely, just check that
    # the quotient is "close enough" to the integer value
    def round_div(x: int, y: int) -> Optional[int]:
        if y <= 1:
            return None
        result = round(x / y)
        if x / (y + 1) <= result <= x / (y - 1):
            return result
        return None

    if expr.op in mult_high_ops and isinstance(right_expr, Literal):
        denom = round_div(1 << (32 + divisor_shift), right_expr.value)
        if denom is not None:
            return BinaryOp.int(
                left=left_expr,
                op="/",
                right=Literal(denom),
            )

    return original_expr


def replace_clz_shift(expr: BinaryOp) -> BinaryOp:
    """
    Simplify an expression matching `CLZ(x) >> 5` into `x == 0`,
    and further simplify `(a - b) == 0` into `a == b`.
    """
    # Check that the outer expression is `>>`
    if expr.is_floating() or expr.op != ">>":
        return expr

    # Match `CLZ(x) >> 5`, or return the original expr
    left_expr = early_unwrap_ints(expr.left)
    right_expr = early_unwrap_ints(expr.right)
    if not (
        isinstance(left_expr, UnaryOp)
        and left_expr.op == "CLZ"
        and isinstance(right_expr, Literal)
        and right_expr.value == 5
    ):
        return expr

    # If the inner `x` is `(a - b)`, return `a == b`
    sub_expr = early_unwrap(left_expr.expr)
    if (
        isinstance(sub_expr, BinaryOp)
        and not sub_expr.is_floating()
        and sub_expr.op == "-"
    ):
        return BinaryOp.icmp(sub_expr.left, "==", sub_expr.right)

    return BinaryOp.icmp(left_expr.expr, "==", Literal(0, type=left_expr.expr.type))


def replace_bitand(expr: BinaryOp) -> Expression:
    """Detect expressions using `&` for truncating integer casts"""
    if not expr.is_floating() and expr.op == "&":
        if expr.right == Literal(0xFF):
            return as_type(expr.left, Type.int_of_size(8), silent=False)
        if expr.right == Literal(0xFFFF):
            return as_type(expr.left, Type.int_of_size(16), silent=False)
    return expr


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
        if isinstance(expr, UnaryOp) and expr.op == "-" and not toplevel:
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
    *,
    target_size: Optional[int],
    ptr: bool,
) -> Optional[Expression]:
    expr = early_unwrap(expr)
    if not isinstance(expr, BinaryOp) or expr.op != "+":
        return None
    base = expr.left
    addend = expr.right
    if addend.type.is_pointer_or_array() and not base.type.is_pointer_or_array():
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

    if scale < 0:
        scale = -scale
        index = UnaryOp("-", as_s32(index, silent=True), type=Type.s32())

    target_type = base.type.get_pointer_target()
    if target_type is None:
        return None

    uw_base = early_unwrap(base)
    typepool = stack_info.global_info.typepool

    # In `&x + index * scale`, if the type of `x` is not known, try to mark it as an array.
    # Skip the `scale = 1` case because this often indicates a complex `index` expression,
    # and is not actually a 1-byte array lookup.
    if (
        scale > 1
        and offset == 0
        and isinstance(uw_base, AddressOf)
        and target_type.get_size_bytes() is None
    ):
        inner_type: Optional[Type] = None
        if (
            isinstance(uw_base.expr, GlobalSymbol)
            and uw_base.expr.potential_array_dim(scale)[1] != 0
        ):
            # For GlobalSymbols, use the size of the asm data to check the feasibility of being
            # an array with `scale`. This helps be more conservative around fake symbols.
            pass
        elif scale == 2:
            # This *could* be a struct, but is much more likely to be an int
            inner_type = Type.int_of_size(16)
        elif scale == 4:
            inner_type = Type.reg32(likely_float=False)
        elif typepool.unk_inference and isinstance(uw_base.expr, GlobalSymbol):
            # Make up a struct with a tag name based on the symbol & struct size.
            # Although `scale = 8` could indicate an array of longs/doubles, it seems more
            # common to be an array of structs.
            struct_name = f"_struct_{uw_base.expr.symbol_name}_0x{scale:X}"
            struct = typepool.get_struct_by_tag_name(
                struct_name, stack_info.global_info.typemap
            )
            if struct is None:
                struct = StructDeclaration.unknown_of_size(
                    typepool, size=scale, tag_name=struct_name
                )
            elif struct.size != scale:
                # This should only happen if there was already a struct with this name in the context
                raise DecompFailure(f"sizeof(struct {struct_name}) != {scale:#x}")
            inner_type = Type.struct(struct)

        if inner_type is not None:
            # This might fail, if `uw_base.expr.type` can't be changed to an array
            uw_base.expr.type.unify(Type.array(inner_type, dim=None))
            # This acts as a backup, and will usually succeed
            target_type.unify(inner_type)

    if target_type.get_size_bytes() == scale:
        # base[index]
        pass
    else:
        # base->subarray[index]
        sub_path, sub_type, remaining_offset = base.type.get_deref_field(
            offset, target_size=scale, exact=False
        )
        # Check if the last item in the path is `0`, which indicates the start of an array
        # If it is, remove it: it will be replaced by `[index]`
        if sub_path is None or len(sub_path) < 2 or sub_path[-1] != 0:
            return None
        sub_path.pop()
        base = StructAccess(
            struct_var=base,
            offset=offset - remaining_offset,
            target_size=None,
            field_path=sub_path,
            stack_info=stack_info,
            type=sub_type,
        )
        offset = remaining_offset
        target_type = sub_type

    ret: Expression = ArrayAccess(base, index, type=target_type)

    # Add .field if necessary by wrapping ret in StructAccess(AddressOf(...))
    ret_ref = AddressOf(ret, type=ret.type.reference())
    field_path, field_type, _ = ret_ref.type.get_deref_field(
        offset, target_size=target_size
    )

    if offset != 0 or (target_size is not None and target_size != scale):
        ret = StructAccess(
            struct_var=ret_ref,
            offset=offset,
            target_size=target_size,
            field_path=field_path,
            stack_info=stack_info,
            type=field_type,
        )

    if ptr:
        ret = AddressOf(ret, type=ret.type.reference())

    return ret


def handle_add(args: InstrArgs) -> Expression:
    lhs = args.reg(1)
    rhs = args.reg(2)
    stack_info = args.stack_info
    type = Type.intptr()
    # Because lhs & rhs are in registers, it shouldn't be possible for them to be arrays.
    # If they are, treat them the same as pointers anyways.
    if lhs.type.is_pointer_or_array():
        type = Type.ptr()
    elif rhs.type.is_pointer_or_array():
        type = Type.ptr()

    # addiu instructions can sometimes be emitted as addu instead, when the
    # offset is too large.
    if isinstance(rhs, Literal):
        return handle_addi_real(args.reg_ref(0), args.reg_ref(1), lhs, rhs, stack_info)
    if isinstance(lhs, Literal):
        return handle_addi_real(args.reg_ref(0), args.reg_ref(2), rhs, lhs, stack_info)

    expr = BinaryOp(left=as_intptr(lhs), op="+", right=as_intptr(rhs), type=type)
    folded_expr = fold_mul_chains(expr)
    if isinstance(folded_expr, BinaryOp):
        folded_expr = fold_divmod(folded_expr)
    if folded_expr is not expr:
        return folded_expr
    array_expr = array_access_from_add(expr, 0, stack_info, target_size=None, ptr=True)
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


def rlwi_mask(mask_begin: int, mask_end: int) -> int:
    # Compute the mask constant used by the rlwi* family of PPC instructions,
    # referred to as the `MASK(MB, ME)` function in the processor manual.
    # Bit 0 is the MSB, Bit 31 is the LSB
    bits_upto: Callable[[int], int] = lambda m: (1 << (32 - m)) - 1
    all_ones = 0xFFFFFFFF
    if mask_begin <= mask_end:
        # Set bits inside the range, fully inclusive
        mask = bits_upto(mask_begin) - bits_upto(mask_end + 1)
    else:
        # Set bits from [31, mask_end] and [mask_begin, 0]
        mask = (bits_upto(mask_end + 1) - bits_upto(mask_begin)) ^ all_ones
    return mask


def handle_rlwinm(
    source: Expression,
    shift: int,
    mask_begin: int,
    mask_end: int,
    simplify: bool = True,
) -> Expression:
    # TODO: Detect shift + truncate, like `(x << 2) & 0xFFF3` or `(x >> 2) & 0x3FFF`

    # The output of the rlwinm instruction is `ROTL(source, shift) & mask`. We write this as
    # ((source << shift) & mask) | ((source >> (32 - shift)) & mask)
    # and compute both OR operands (upper_bits and lower_bits respectively).
    all_ones = 0xFFFFFFFF
    mask = rlwi_mask(mask_begin, mask_end)
    left_shift = shift
    right_shift = 32 - shift
    left_mask = (all_ones << left_shift) & mask
    right_mask = (all_ones >> right_shift) & mask

    # We only simplify if the `simplify` argument is True, and there will be no `|` in the
    # resulting expression. If there is an `|`, the expression is best left as bitwise math
    simplify = simplify and not (left_mask and right_mask)

    if isinstance(source, Literal):
        upper_value = (source.value << left_shift) & mask
        lower_value = (source.value >> right_shift) & mask
        return Literal(upper_value | lower_value)

    upper_bits: Optional[Expression]
    if left_mask == 0:
        upper_bits = None
    else:
        upper_bits = source
        if left_shift != 0:
            upper_bits = BinaryOp.int(
                left=upper_bits, op="<<", right=Literal(left_shift)
            )

        if simplify:
            upper_bits = fold_mul_chains(upper_bits)

        if left_mask != (all_ones << left_shift) & all_ones:
            upper_bits = BinaryOp.int(left=upper_bits, op="&", right=Literal(left_mask))
            if simplify:
                upper_bits = replace_bitand(upper_bits)

    lower_bits: Optional[Expression]
    if right_mask == 0:
        lower_bits = None
    else:
        lower_bits = BinaryOp.u32(left=source, op=">>", right=Literal(right_shift))

        if simplify:
            lower_bits = replace_clz_shift(fold_divmod(lower_bits))

        if right_mask != (all_ones >> right_shift) & all_ones:
            lower_bits = BinaryOp.int(
                left=lower_bits, op="&", right=Literal(right_mask)
            )
            if simplify:
                lower_bits = replace_bitand(lower_bits)

    if upper_bits is None and lower_bits is None:
        return Literal(0)
    elif upper_bits is None:
        assert lower_bits is not None
        return lower_bits
    elif lower_bits is None:
        return upper_bits
    else:
        return BinaryOp.int(left=upper_bits, op="|", right=lower_bits)


def handle_rlwimi(
    base: Expression, source: Expression, shift: int, mask_begin: int, mask_end: int
) -> Expression:
    # This instruction reads from `base`, replaces some bits with values from `source`, then
    # writes the result back into the first register. This can be used to copy any contiguous
    # bitfield from `source` into `base`, and is commonly used when manipulating flags, such
    # as in `x |= 0x10` or `x &= ~0x10`.

    # It's generally more readable to write the mask with `~` (instead of computing the inverse here)
    mask_literal = Literal(rlwi_mask(mask_begin, mask_end))
    mask = UnaryOp("~", mask_literal, type=Type.u32())
    masked_base = BinaryOp.int(left=base, op="&", right=mask)
    if source == Literal(0):
        # If the source is 0, there are no bits inserted. (This may look like `x &= ~0x10`)
        return masked_base
    # Set `simplify=False` to keep the `inserted` expression as bitwise math instead of `*` or `/`
    inserted = handle_rlwinm(source, shift, mask_begin, mask_end, simplify=False)
    if inserted == mask_literal:
        # If this instruction will set all the bits in the mask, we can OR the values
        # together without masking the base. (`x |= 0xF0` instead of `x = (x & ~0xF0) | 0xF0`)
        return BinaryOp.int(left=base, op="|", right=inserted)
    return BinaryOp.int(left=masked_base, op="|", right=inserted)


def handle_loadx(args: InstrArgs, type: Type) -> Expression:
    # "indexed loads" like `lwzx rD, rA, rB` read `(rA + rB)` into `rD`
    size = type.get_size_bytes()
    assert size is not None

    ptr = BinaryOp.intptr(left=args.reg(1), op="+", right=args.reg(2))
    expr = deref(ptr, args.regs, args.stack_info, size=size)
    return as_type(expr, type, silent=True)


def strip_macros(arg: Argument) -> Argument:
    """Replace %lo(...) by 0, and assert that there are no %hi(...). We assume that
    %hi's only ever occur in lui, where we expand them to an entire value, and not
    just the upper part. This preserves semantics in most cases (though not when %hi's
    are reused for different %lo's...)"""
    if isinstance(arg, Macro):
        if arg.macro_name in ["sda2", "sda21"]:
            return arg.argument
        if arg.macro_name == "hi":
            raise DecompFailure("%hi macro outside of lui")
        if arg.macro_name not in ["lo", "l"]:
            raise DecompFailure(f"Unrecognized linker macro %{arg.macro_name}")
        # This is sort of weird; for `symbol@l` we return 0 here and assume
        # that this @l is always perfectly paired with one other @ha.
        # However, with `literal@l`, we return the literal value, and assume it is
        # paired with another `literal@ha`. This lets us reuse `literal@ha` values,
        # but assumes that we never mix literals & symbols
        if isinstance(arg.argument, AsmLiteral):
            return AsmLiteral(arg.argument.value)
        return AsmLiteral(0)
    elif isinstance(arg, AsmAddressMode) and isinstance(arg.lhs, Macro):
        if arg.lhs.macro_name in ["sda2", "sda21"]:
            return arg.lhs.argument
        if arg.lhs.macro_name not in ["lo", "l"]:
            raise DecompFailure(
                f"Bad linker macro in instruction argument {arg}, expected %lo"
            )
        return AsmAddressMode(lhs=AsmLiteral(0), rhs=arg.rhs)
    else:
        return arg


@dataclass
class AbiArgSlot:
    offset: int
    reg: Optional[Register]
    type: Type
    name: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class Abi:
    arg_slots: List[AbiArgSlot]
    possible_slots: List[AbiArgSlot]


def pick_phi_assignment_nodes(
    reg: Register, nodes: List[Node], expr: Expression
) -> List[Node]:
    """
    As part of `assign_phis()`, we need to pick a set of nodes where we can emit a
    `SetPhiStmt` that assigns the phi for `reg` to `expr`.
    The final register state for `reg` for each node in `nodes` is `expr`,
    so the best case would be finding a single dominating node for the assignment.
    """
    # Find the set of nodes which dominate *all* of `nodes`, sorted by number
    # of dominators. (This puts "earlier" nodes at the beginning of the list.)
    dominators = sorted(
        set.intersection(*(node.dominators for node in nodes)),
        key=lambda n: len(n.dominators),
    )

    # Check the dominators for a node with the correct final state for `reg`
    for node in dominators:
        regs = get_block_info(node).final_register_states
        raw = regs.get_raw(reg)
        meta = regs.get_meta(reg)
        if raw is None or meta is None or meta.force:
            continue
        if raw == expr:
            return [node]

    # We couldn't find anything, so fall back to the naive solution
    # TODO: In some cases there may be a better solution (e.g. one that requires 2 nodes)
    return nodes


def assign_phis(used_phis: List[PhiExpr], stack_info: StackInfo) -> None:
    i = 0
    # Iterate over used phis until there are no more remaining. New ones may
    # appear during iteration, hence the while loop.
    while i < len(used_phis):
        phi = used_phis[i]
        assert phi.num_usages > 0
        assert len(phi.node.parents) >= 2

        # Group parent nodes by the value of their phi register
        equivalent_nodes: DefaultDict[Expression, List[Node]] = defaultdict(list)
        for node in phi.node.parents:
            expr = get_block_info(node).final_register_states[phi.reg]
            expr.type.unify(phi.type)
            equivalent_nodes[expr].append(node)

        exprs = list(equivalent_nodes.keys())
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
            for _ in range(phi.num_usages):
                first_uw.use()
        else:
            for expr, nodes in equivalent_nodes.items():
                for node in pick_phi_assignment_nodes(phi.reg, nodes, expr):
                    block_info = get_block_info(node)
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


def propagate_register_meta(nodes: List[Node], reg: Register) -> None:
    """Propagate RegMeta bits forwards/backwards."""
    non_terminal: List[Node] = [n for n in nodes if not isinstance(n, TerminalNode)]

    # Set `is_read` based on `read_inherited`.
    for n in non_terminal:
        if reg in get_block_info(n).final_register_states.read_inherited:
            for p in n.parents:
                par_meta = get_block_info(p).final_register_states.get_meta(reg)
                if par_meta:
                    par_meta.is_read = True

    # Propagate `is_read` backwards.
    todo = non_terminal[:]
    while todo:
        n = todo.pop()
        meta = get_block_info(n).final_register_states.get_meta(reg)
        for p in n.parents:
            par_meta = get_block_info(p).final_register_states.get_meta(reg)
            if (par_meta and not par_meta.is_read) and (
                meta and meta.inherited and meta.is_read
            ):
                par_meta.is_read = True
                todo.append(p)

    # Set `uninteresting` and propagate it and `function_return` forwards. Start by
    # assuming inherited values are all set; they will get unset iteratively, but for
    # cyclic dependency purposes we want to assume them set.
    for n in non_terminal:
        meta = get_block_info(n).final_register_states.get_meta(reg)
        if meta:
            if meta.inherited:
                meta.uninteresting = True
                meta.function_return = True
            else:
                meta.uninteresting |= meta.is_read or meta.function_return

    todo = non_terminal[:]
    while todo:
        n = todo.pop()
        if isinstance(n, TerminalNode):
            continue
        meta = get_block_info(n).final_register_states.get_meta(reg)
        if not meta or not meta.inherited:
            continue
        all_uninteresting = True
        all_function_return = True
        for p in n.parents:
            par_meta = get_block_info(p).final_register_states.get_meta(reg)
            if par_meta:
                all_uninteresting &= par_meta.uninteresting
                all_function_return &= par_meta.function_return
        if meta.uninteresting and not all_uninteresting and not meta.is_read:
            meta.uninteresting = False
            todo.extend(n.children())
        if meta.function_return and not all_function_return:
            meta.function_return = False
            todo.extend(n.children())


def determine_return_register(
    return_blocks: List[BlockInfo], fn_decl_provided: bool, arch: Arch
) -> Optional[Register]:
    """Determine which of the arch's base_return_regs (i.e. v0, f0) is the most
    likely to contain the return value, or if the function is likely void."""

    def priority(block_info: BlockInfo, reg: Register) -> int:
        meta = block_info.final_register_states.get_meta(reg)
        if not meta:
            return 3
        if meta.uninteresting:
            return 1
        if meta.function_return:
            return 0
        return 2

    if not return_blocks:
        return None

    best_reg: Optional[Register] = None
    best_prio = -1
    for reg in arch.base_return_regs:
        prios = [priority(b, reg) for b in return_blocks]
        max_prio = max(prios)
        if max_prio == 3:
            # Register is not always set, skip it
            continue
        if max_prio <= 1 and not fn_decl_provided:
            # Register is always read after being written, or comes from a
            # function call; seems unlikely to be an intentional return.
            # Skip it, unless we have a known non-void return type.
            continue
        if max_prio > best_prio:
            best_prio = max_prio
            best_reg = reg
    return best_reg


def translate_node_body(node: Node, regs: RegInfo, stack_info: StackInfo) -> BlockInfo:
    """
    Given a node and current register contents, return a BlockInfo containing
    the translated AST for that node.
    """

    to_write: List[Union[Statement]] = []
    local_var_writes: Dict[LocalVar, Tuple[Register, Expression]] = {}
    subroutine_args: Dict[int, Expression] = {}
    branch_condition: Optional[Condition] = None
    switch_expr: Optional[Expression] = None
    has_custom_return: bool = False
    has_function_call: bool = False
    arch = stack_info.global_info.arch

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
            data = regs.contents.get(r)
            assert data is not None
            expr = data.value
            if not data.meta.force and expr_filter(expr):
                # Mark the register as "if used, emit the expression's once
                # var". We usually always have a once var at this point,
                # but if we don't, create one.
                if not isinstance(expr, EvalOnceExpr):
                    expr = eval_once(
                        expr,
                        emit_exactly_once=False,
                        trivial=False,
                        prefix=r.register_name,
                    )

                # This write isn't changing the value of the register; it didn't need
                # to be declared as part of the current instruction's inputs/outputs.
                regs.unchecked_set_with_meta(r, expr, replace(data.meta, force=True))

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
        regs[reg] = expr

    def set_reg(reg: Register, expr: Optional[Expression]) -> Optional[Expression]:
        if expr is None:
            if reg in regs:
                del regs[reg]
            return None

        if isinstance(expr, LocalVar):
            if (
                isinstance(node, ReturnNode)
                and stack_info.maybe_get_register_var(reg)
                and (stack_info.callee_save_reg_locations.get(reg) == expr.value)
            ):
                # Elide saved register restores with --reg-vars (it doesn't
                # matter in other cases).
                return None
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
                if not (isinstance(uw_expr, RegisterVar) and uw_expr.reg == reg):
                    source = as_type(expr, dest.type, True)
                    source.use()
                    to_write.append(StoreStmt(source=source, dest=dest))
                expr = dest
            set_reg_maybe_return(reg, expr)
        return expr

    def clear_caller_save_regs() -> None:
        for reg in arch.temp_regs:
            if reg in regs:
                del regs[reg]

    def maybe_clear_local_var_writes(func_args: List[Expression]) -> None:
        # Clear the `local_var_writes` dict if any of the `func_args` contain
        # a reference to a stack var. (The called function may modify the stack,
        # replacing the value we have in `local_var_writes`.)
        for arg in func_args:
            if uses_expr(
                arg,
                lambda expr: isinstance(expr, AddressOf)
                and isinstance(expr.expr, LocalVar),
            ):
                local_var_writes.clear()
                return

    def process_instr(instr: Instruction) -> None:
        nonlocal branch_condition, switch_expr, has_function_call

        mnemonic = instr.mnemonic
        arch_mnemonic = instr.arch_mnemonic(arch)
        args = InstrArgs(instr.args, regs, stack_info)
        expr: Expression

        # Figure out what code to generate!
        if mnemonic in arch.instrs_ignore:
            pass

        elif mnemonic in arch.instrs_store or mnemonic in arch.instrs_store_update:
            # Store a value in a permanent place.
            if mnemonic in arch.instrs_store:
                to_store = arch.instrs_store[mnemonic](args)
            else:
                # PPC specific store-and-update instructions
                # `stwu r3, 8(r4)` is equivalent to `$r3 = *($r4 + 8); $r4 += 8;`
                to_store = arch.instrs_store_update[mnemonic](args)

                # Update the register in the second argument
                update = args.memory_ref(1)
                if not isinstance(update, AddressMode):
                    raise DecompFailure(
                        f"Unhandled store-and-update arg in {instr}: {update!r}"
                    )
                set_reg(
                    update.rhs,
                    add_imm(args.regs[update.rhs], Literal(update.offset), stack_info),
                )

            if to_store is None:
                # Elided register preserval.
                pass
            elif isinstance(to_store.dest, SubroutineArg):
                # About to call a subroutine with this argument. Skip arguments for the
                # first four stack slots; they are also passed in registers.
                if to_store.dest.value >= 0x10:
                    subroutine_args[to_store.dest.value] = to_store.source
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

        elif mnemonic in arch.instrs_source_first:
            # Just 'mtc1'. It's reversed, so we have to specially handle it.
            set_reg(args.reg_ref(1), arch.instrs_source_first[mnemonic](args))

        elif mnemonic in arch.instrs_branches:
            assert branch_condition is None
            branch_condition = arch.instrs_branches[mnemonic](args)

        elif mnemonic in arch.instrs_float_branches:
            assert branch_condition is None
            cond_bit = regs[Register("condition_bit")]
            if not isinstance(cond_bit, BinaryOp):
                cond_bit = ExprCondition(cond_bit, type=cond_bit.type)
            if arch_mnemonic == "mips:bc1t":
                branch_condition = cond_bit
            elif arch_mnemonic == "mips:bc1f":
                branch_condition = cond_bit.negated()

        elif mnemonic in arch.instrs_jumps:
            if arch_mnemonic == "ppc:bctr":
                # Switch jump
                assert isinstance(node, SwitchNode)
                switch_expr = args.regs[Register("ctr")]
            elif arch_mnemonic == "mips:jr":
                # MIPS:
                if args.reg_ref(0) == arch.return_address_reg:
                    # Return from the function.
                    assert isinstance(node, ReturnNode)
                else:
                    # Switch jump.
                    assert isinstance(node, SwitchNode)
                    switch_expr = args.reg(0)
            elif arch_mnemonic == "ppc:blr":
                assert isinstance(node, ReturnNode)
            else:
                assert False, f"Unhandled jump mnemonic {arch_mnemonic}"

        elif mnemonic in arch.instrs_fn_call:
            if arch_mnemonic in ["mips:jal", "ppc:bl"]:
                fn_target = args.imm(0)
                if not (
                    (
                        isinstance(fn_target, AddressOf)
                        and isinstance(fn_target.expr, GlobalSymbol)
                    )
                    or isinstance(fn_target, Literal)
                ):
                    raise DecompFailure(
                        f"Target of function call must be a symbol, not {fn_target}"
                    )
            elif arch_mnemonic == "ppc:blrl":
                fn_target = args.regs[Register("lr")]
            elif arch_mnemonic == "ppc:bctrl":
                fn_target = args.regs[Register("ctr")]
            elif arch_mnemonic == "mips:jalr":
                fn_target = args.reg(1)
            else:
                assert False, f"Unhandled fn call mnemonic {arch_mnemonic}"

            fn_target = as_function_ptr(fn_target)
            fn_sig = fn_target.type.get_function_pointer_signature()
            assert fn_sig is not None, "known function pointers must have a signature"

            likely_regs: Dict[Register, bool] = {}
            for reg, data in regs.contents.items():
                # We use a much stricter filter for PPC than MIPS, because the same
                # registers can be used arguments & return values.
                # The ABI can also mix & match the rN & fN registers, which  makes the
                # "require" heuristic less powerful.
                #
                # - `meta.inherited` will only be False for registers set in *this* basic block
                # - `meta.function_return` will only be accurate for registers set within this
                #   basic block because we have not called `propagate_register_meta` yet.
                #   Within this block, it will be True for registers that were return values.
                if arch.arch == Target.ArchEnum.PPC and (
                    data.meta.inherited or data.meta.function_return
                ):
                    likely_regs[reg] = False
                elif isinstance(data.value, PassedInArg) and not data.value.copied:
                    likely_regs[reg] = False
                else:
                    likely_regs[reg] = True

            abi = arch.function_abi(fn_sig, likely_regs, for_call=True)

            func_args: List[Expression] = []
            for slot in abi.arg_slots:
                if slot.reg:
                    expr = regs[slot.reg]
                elif slot.offset in subroutine_args:
                    expr = subroutine_args.pop(slot.offset)
                else:
                    expr = ErrorExpr(
                        f"Unable to find stack arg {slot.offset:#x} in block"
                    )
                func_args.append(
                    CommentExpr.wrap(
                        as_type(expr, slot.type, True), prefix=slot.comment
                    )
                )

            for slot in abi.possible_slots:
                assert slot.reg is not None
                func_args.append(regs[slot.reg])

            # Add the arguments after a3.
            # TODO: limit this based on abi.arg_slots. If the function type is known
            # and not variadic, this list should be empty.
            for _, arg in sorted(subroutine_args.items()):
                if fn_sig.params_known and not fn_sig.is_variadic:
                    func_args.append(CommentExpr.wrap(arg, prefix="extra?"))
                else:
                    func_args.append(arg)

            if not fn_sig.params_known:
                while len(func_args) > len(fn_sig.params):
                    fn_sig.params.append(FunctionParam())
                # When the function signature isn't provided, the we only assume that each
                # parameter is "simple" (<=4 bytes, no return struct, etc.). This may not
                # match the actual function signature, but it's the best we can do.
                # Without that assumption, the logic from `function_abi` would be needed here.
                for i, (arg_expr, param) in enumerate(zip(func_args, fn_sig.params)):
                    func_args[i] = as_type(arg_expr, param.type.decay(), True)

            # Reset subroutine_args, for the next potential function call.
            subroutine_args.clear()

            call: Expression = FuncCall(
                fn_target, func_args, fn_sig.return_type.weaken_void_ptr()
            )
            call = eval_once(call, emit_exactly_once=True, trivial=False, prefix="ret")

            # Clear out caller-save registers, for clarity and to ensure that
            # argument regs don't get passed into the next function.
            clear_caller_save_regs()

            # Clear out local var write tracking if any argument contains a stack
            # reference. That dict is used to track register saves/restores, which
            # are unreliable if we call a function with a stack reference.
            maybe_clear_local_var_writes(func_args)

            # Prevent reads and function calls from moving across this call.
            # This isn't really right, because this call might be moved later,
            # and then this prevention should also be... but it's the best we
            # can do with the current code architecture.
            prevent_later_function_calls()
            prevent_later_reads()

            return_reg_vals = arch.function_return(call)
            for out in instr.outputs:
                if not isinstance(out, Register):
                    continue
                val = return_reg_vals[out]
                if not isinstance(val, SecondF64Half):
                    val = eval_once(
                        val,
                        emit_exactly_once=False,
                        trivial=False,
                        prefix=out.register_name,
                    )
                regs.set_with_meta(out, val, RegMeta(function_return=True))

            has_function_call = True

        elif mnemonic in arch.instrs_float_comp:
            expr = arch.instrs_float_comp[mnemonic](args)
            regs[Register("condition_bit")] = expr

        elif mnemonic in arch.instrs_hi_lo:
            hi, lo = arch.instrs_hi_lo[mnemonic](args)
            set_reg(Register("hi"), hi)
            set_reg(Register("lo"), lo)

        elif mnemonic in arch.instrs_implicit_destination:
            reg, expr_fn = arch.instrs_implicit_destination[mnemonic]
            set_reg(reg, expr_fn(args))

        elif mnemonic in arch.instrs_ppc_compare:
            if instr.args[0] != Register("cr0"):
                raise DecompFailure(
                    f"Instruction {instr} not supported (first arg is not $cr0)"
                )

            set_reg(Register("cr0_eq"), arch.instrs_ppc_compare[mnemonic](args, "=="))
            set_reg(Register("cr0_gt"), arch.instrs_ppc_compare[mnemonic](args, ">"))
            set_reg(Register("cr0_lt"), arch.instrs_ppc_compare[mnemonic](args, "<"))
            set_reg(Register("cr0_so"), Literal(0))

        elif mnemonic in arch.instrs_no_dest:
            stmt = arch.instrs_no_dest[mnemonic](args)
            to_write.append(stmt)

        elif mnemonic.rstrip(".") in arch.instrs_destination_first:
            target = args.reg_ref(0)
            val = arch.instrs_destination_first[mnemonic.rstrip(".")](args)
            # TODO: IDO tends to keep variables within single registers. Thus,
            # if source = target, maybe we could overwrite that variable instead
            # of creating a new one?
            target_val = set_reg(target, val)
            mn_parts = arch_mnemonic.split(".")
            if arch_mnemonic.startswith("ppc:") and arch_mnemonic.endswith("."):
                # PPC instructions suffixed with . set condition bits (CR0) based on the result value
                if target_val is None:
                    target_val = val
                set_reg(
                    Register("cr0_eq"),
                    BinaryOp.icmp(target_val, "==", Literal(0, type=target_val.type)),
                )
                # Use manual casts for cr0_gt/cr0_lt so that the type of target_val is not modified
                # until the resulting bit is .use()'d.
                target_s32 = Cast(
                    target_val, reinterpret=True, silent=True, type=Type.s32()
                )
                set_reg(
                    Register("cr0_gt"),
                    BinaryOp(target_s32, ">", Literal(0), type=Type.s32()),
                )
                set_reg(
                    Register("cr0_lt"),
                    BinaryOp(target_s32, "<", Literal(0), type=Type.s32()),
                )
                set_reg(
                    Register("cr0_so"),
                    fn_op("MIPS2C_OVERFLOW", [target_val], type=Type.s32()),
                )

            elif (
                len(mn_parts) >= 2
                and mn_parts[0].startswith("mips:")
                and mn_parts[1] == "d"
            ) or arch_mnemonic == "mips:ldc1":
                set_reg(target.other_f64_reg(), SecondF64Half())

        elif mnemonic in arch.instrs_load_update:
            target = args.reg_ref(0)
            val = arch.instrs_load_update[mnemonic](args)
            set_reg(target, val)

            if arch_mnemonic in ["ppc:lwzux", "ppc:lhzux", "ppc:lbzux"]:
                # In `rD, rA, rB`, update `rA = rA + rB`
                update_reg = args.reg_ref(1)
                offset = args.reg(2)
            else:
                # In `rD, rA(N)`, update `rA = rA + N`
                update = args.memory_ref(1)
                if not isinstance(update, AddressMode):
                    raise DecompFailure(
                        f"Unhandled store-and-update arg in {instr}: {update!r}"
                    )
                update_reg = update.rhs
                offset = Literal(update.offset)

            if update_reg == target:
                raise DecompFailure(
                    f"Invalid instruction, rA and rD must be different in {instr}"
                )

            set_reg(update_reg, add_imm(args.regs[update_reg], offset, stack_info))

        else:
            expr = ErrorExpr(f"unknown instruction: {instr}")
            if arch_mnemonic.startswith("ppc:") and arch_mnemonic.endswith("."):
                # Unimplemented PPC instructions that modify CR0
                set_reg(Register("cr0_eq"), expr)
                set_reg(Register("cr0_gt"), expr)
                set_reg(Register("cr0_lt"), expr)
                set_reg(Register("cr0_so"), expr)
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
        with regs.current_instr(instr):
            process_instr(instr)

    if branch_condition is not None:
        branch_condition.use()
    switch_control: Optional[SwitchControl] = None
    if switch_expr is not None:
        switch_control = SwitchControl.from_expr(switch_expr)
        switch_control.control_expr.use()
    return BlockInfo(
        to_write=to_write,
        return_value=None,
        switch_control=switch_control,
        branch_condition=branch_condition,
        final_register_states=regs,
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
            to_write=error_stmts,
            return_value=None,
            switch_control=None,
            branch_condition=ErrorExpr(),
            final_register_states=regs,
            has_function_call=False,
        )

    node.block.add_block_info(block_info)
    if isinstance(node, ReturnNode):
        return_blocks.append(block_info)

    # Translate everything dominated by this node, now that we know our own
    # final register state. This will eventually reach every node.
    for child in node.immediately_dominates:
        if isinstance(child, TerminalNode):
            continue
        new_regs = RegInfo(stack_info=stack_info)
        for reg, data in regs.contents.items():
            new_regs.set_with_meta(
                reg, data.value, RegMeta(inherited=True, force=data.meta.force)
            )

        for phi_arg, addrs in stack_info.flow_graph.node_phis[child].items():
            if not isinstance(phi_arg, Register):
                continue
            if addrs.is_valid():
                expr: Optional[Expression] = stack_info.maybe_get_register_var(phi_arg)
                if expr is None:
                    expr = PhiExpr(
                        reg=phi_arg,
                        node=child,
                        used_phis=used_phis,
                        type=Type.any_reg(),
                    )
                new_regs.set_with_meta(phi_arg, expr, RegMeta(inherited=True))
            elif phi_arg in new_regs:
                del new_regs[phi_arg]
        translate_graph_from_block(
            child, new_regs, stack_info, used_phis, return_blocks, options
        )


def resolve_types_late(stack_info: StackInfo) -> None:
    """
    After translating a function, perform a final type-resolution pass.
    """
    # Final check over stack var types. Because of delayed type unification, some
    # locations should now be marked as "weak".
    for location in stack_info.weak_stack_var_types.keys():
        stack_info.get_stack_var(location, store=False)

    # Use dereferences to determine pointer types
    struct_type_map = stack_info.get_struct_type_map()
    for var, offset_type_map in struct_type_map.items():
        if len(offset_type_map) == 1 and 0 in offset_type_map:
            # var was probably a plain pointer, not a struct
            # Try to unify it with the appropriate pointer type,
            # to fill in the type if it does not already have one
            type = offset_type_map[0]
            var.type.unify(Type.ptr(type))


@dataclass
class FunctionInfo:
    stack_info: StackInfo
    flow_graph: FlowGraph
    return_type: Type
    symbol: GlobalSymbol


@dataclass
class GlobalInfo:
    asm_data: AsmData
    arch: Arch
    target: Target
    local_functions: Set[str]
    typemap: TypeMap
    typepool: TypePool
    global_symbol_map: Dict[str, GlobalSymbol] = field(default_factory=dict)

    def asm_data_value(self, sym_name: str) -> Optional[AsmDataEntry]:
        return self.asm_data.values.get(sym_name)

    def address_of_gsym(self, sym_name: str) -> AddressOf:
        if sym_name in self.global_symbol_map:
            sym = self.global_symbol_map[sym_name]
        else:
            demangled_symbol: Optional[CxxSymbol] = None
            demangled_str: Optional[str] = None
            if self.target.language == Target.LanguageEnum.CXX:
                try:
                    demangled_symbol = demangle_codewarrior_parse(sym_name)
                except ValueError:
                    pass
                else:
                    demangled_str = str(demangled_symbol)

            sym = self.global_symbol_map[sym_name] = GlobalSymbol(
                symbol_name=sym_name,
                type=Type.any(),
                asm_data_entry=self.asm_data_value(sym_name),
                demangled_str=demangled_str,
            )

            fn = self.typemap.functions.get(sym_name)
            ctype: Optional[CType]
            if fn is not None:
                ctype = fn.type
            else:
                ctype = self.typemap.var_types.get(sym_name)

            if ctype is not None:
                sym.symbol_in_context = True
                sym.initializer_in_typemap = (
                    sym_name in self.typemap.vars_with_initializers
                )
                sym.type.unify(Type.ctype(ctype, self.typemap, self.typepool))
                if sym_name not in self.typepool.unknown_decls:
                    sym.type_provided = True
            elif sym_name in self.local_functions:
                sym.type.unify(Type.function())

            # Do this after unifying the type in the typemap, so that it has lower precedence
            if demangled_symbol is not None:
                sym.type.unify(Type.demangled_symbol(demangled_symbol))

        return AddressOf(sym, type=sym.type.reference())

    def is_function_known_void(self, sym_name: str) -> bool:
        """Return True if the function exists in the context, and has no return value"""
        fn = self.typemap.functions.get(sym_name)
        if fn is None:
            return False
        return fn.ret_type is None

    def initializer_for_symbol(
        self, sym: GlobalSymbol, fmt: Formatter
    ) -> Optional[str]:
        assert sym.asm_data_entry is not None
        data = sym.asm_data_entry.data[:]

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
            if not data or not isinstance(data[0], str):
                return None

            label = data[0]
            data.pop(0)
            return self.address_of_gsym(label)

        def for_type(type: Type) -> Optional[str]:
            """Return the initializer for a single element of type `type`"""
            if type.is_struct() or type.is_array():
                struct_fields = type.get_initializer_fields()
                if not struct_fields:
                    return None
                members = []
                for field in struct_fields:
                    if isinstance(field, int):
                        # Check that all padding bytes are 0
                        for i in range(field):
                            padding = read_uint(1)
                            if padding != 0:
                                return None
                    else:
                        m = for_type(field)
                        if m is None:
                            return None
                        members.append(m)
                return fmt.format_array(members)

            if type.is_reg():
                size = type.get_size_bytes()
                if not size:
                    return None

                if size == 4:
                    ptr = read_pointer()
                    if ptr is not None:
                        return as_type(ptr, type, silent=True).format(fmt)

                value = read_uint(size)
                if value is not None:
                    expr = as_type(Literal(value), type, True)
                    return elide_casts_for_store(expr).format(fmt)

            # Type kinds K_FN and K_VOID do not have initializers
            return None

        return for_type(sym.type)

    def find_forward_declares_needed(self, functions: List[FunctionInfo]) -> Set[str]:
        funcs_seen = set()
        forward_declares_needed = self.asm_data.mentioned_labels

        for func in functions:
            funcs_seen.add(func.stack_info.function.name)

            for instr in func.stack_info.function.body:
                if not isinstance(instr, Instruction):
                    continue

                for arg in instr.args:
                    if isinstance(arg, AsmGlobalSymbol):
                        func_name = arg.symbol_name
                    elif isinstance(arg, Macro) and isinstance(
                        arg.argument, AsmGlobalSymbol
                    ):
                        func_name = arg.argument.symbol_name
                    else:
                        continue

                    if func_name in self.local_functions:
                        if func_name not in funcs_seen:
                            forward_declares_needed.add(func_name)

        return forward_declares_needed

    def global_decls(
        self,
        fmt: Formatter,
        decls: Options.GlobalDeclsEnum,
        functions: List[FunctionInfo],
    ) -> str:
        # Format labels from symbol_type_map into global declarations.
        # As the initializers are formatted, this may cause more symbols
        # to be added to the global_symbol_map.
        forward_declares_needed = self.find_forward_declares_needed(functions)

        lines = []
        processed_names: Set[str] = set()
        while True:
            names: AbstractSet[str] = self.global_symbol_map.keys()
            if decls == Options.GlobalDeclsEnum.ALL:
                names |= self.asm_data.values.keys()
            names -= processed_names
            if not names:
                break
            for name in sorted(names):
                processed_names.add(name)
                sym = self.address_of_gsym(name).expr
                assert isinstance(sym, GlobalSymbol)
                data_entry = sym.asm_data_entry

                # Is the label defined in this unit (in the active AsmData file(s))
                is_in_file = data_entry is not None or name in self.local_functions
                # Is the label externally visible (mentioned in the context file)
                is_global = sym.symbol_in_context
                # Is the label a symbol in .rodata?
                is_const = data_entry is not None and data_entry.is_readonly

                if data_entry and data_entry.is_jtbl:
                    # Skip jump tables
                    continue
                if is_in_file and is_global and sym.type.is_function():
                    # Skip externally-declared functions that are defined here
                    continue
                if self.local_functions == {name}:
                    # Skip the function being decompiled if just a single one
                    continue
                if not is_in_file and sym.type_provided:
                    # Skip externally-declared symbols that are defined in other files
                    continue

                # TODO: Use original MIPSFile ordering for variables
                sort_order = (
                    not sym.type.is_function(),
                    is_global,
                    is_in_file,
                    is_const,
                    name,
                )
                qualifier = ""
                value: Optional[str] = None
                comments = []

                # Determine type qualifier: static, extern, or neither
                if is_in_file and is_global:
                    qualifier = ""
                elif is_in_file:
                    qualifier = "static"
                else:
                    qualifier = "extern"

                if sym.type.is_function():
                    comments.append(qualifier)
                    qualifier = ""

                # Try to guess if the symbol is an array (and if it is, its dimension) if
                # we have a data entry for it, and the symbol is either not in the typemap
                # or was a variable-length array there ("VLA", e.g. `int []`)
                # (Otherwise, if the dim is provided by the typemap, we trust it.)
                element_type, array_dim = sym.type.get_array()
                is_vla = element_type is not None and (
                    array_dim is None or array_dim <= 0
                )
                if data_entry and (not sym.type_provided or is_vla):
                    # The size of the data entry is uncertain, because of padding
                    # between sections. Generally `(max_data_size - data_size) < 16`.
                    min_data_size, max_data_size = data_entry.size_range_bytes()
                    # The size of the element type (not the size of the array type)
                    if element_type is None:
                        element_type = sym.type

                    # If we don't know the type, we can't guess the array_dim
                    type_size = element_type.get_size_bytes()
                    if type_size:
                        potential_dim, extra_bytes = sym.potential_array_dim(type_size)
                        if potential_dim == 0 and extra_bytes > 0:
                            # The type is too big for our data. (not an array)
                            comments.append(
                                f"type too large by {fmt.format_int(type_size - extra_bytes)}"
                            )
                        elif potential_dim > 1 or is_vla:
                            # NB: In general, replacing the types of Expressions can be sketchy.
                            # However, the GlobalSymbol here came from address_of_gsym(), which
                            # always returns a reference to the element_type.
                            array_dim = potential_dim
                            sym.type = Type.array(element_type, array_dim)

                        if potential_dim != 0 and extra_bytes > 0:
                            comments.append(
                                f"extra bytes: {fmt.format_int(extra_bytes)}"
                            )

                # Try to convert the data from .data/.rodata into an initializer
                if data_entry and not data_entry.is_bss:
                    value = self.initializer_for_symbol(sym, fmt)
                    if value is None:
                        # This warning helps distinguish .bss symbols from .data/.rodata,
                        # IDO only puts symbols in .bss if they don't have any initializer
                        comments.append("unable to generate initializer")

                if is_const:
                    comments.append("const")

                    # Float & string constants are almost always inlined and can be omitted
                    if sym.is_string_constant():
                        continue
                    if array_dim is None and sym.type.is_likely_float():
                        continue

                # In "none" mode, do not emit any decls
                if decls == Options.GlobalDeclsEnum.NONE:
                    continue
                # In modes except "all", skip the decl if the context file already had an initializer
                if decls != Options.GlobalDeclsEnum.ALL and sym.initializer_in_typemap:
                    continue

                if (
                    sym.type.is_function()
                    and decls != Options.GlobalDeclsEnum.ALL
                    and name in self.local_functions
                    and name not in forward_declares_needed
                ):
                    continue

                qualifier = f"{qualifier} " if qualifier else ""
                value = f" = {value}" if value else ""
                lines.append(
                    (
                        sort_order,
                        fmt.with_comments(
                            f"{qualifier}{sym.type.to_decl(name, fmt)}{value};",
                            comments,
                        )
                        + "\n",
                    )
                )
        lines.sort()
        return "".join(line for _, line in lines)


def narrow_ir_with_context(
    function: Function,
    global_info: GlobalInfo,
) -> None:
    """
    Modify the `outputs` list of function call Instructions and the function's
    `arguments` list using the context file.
    """
    fn_ref = global_info.address_of_gsym(function.name)
    fn_sig = fn_ref.type.get_function_pointer_signature()
    if fn_sig is not None:
        abi = global_info.arch.function_abi(
            fn_sig,
            likely_regs={reg: True for reg in global_info.arch.argument_regs},
            for_call=False,
        )

        possible_regs = {slot.reg for slot in abi.arg_slots + abi.possible_slots}
        function.arguments = [arg for arg in function.arguments if arg in possible_regs]

    # For now, this only handles known-void functions, but in the future it could
    # be extended to select a specific register subset based on type or use the
    # inferred signature above.
    for instr in function.body:
        if (
            isinstance(instr, Instruction)
            and isinstance(instr.function_target, AsmGlobalSymbol)
            and global_info.is_function_known_void(instr.function_target.symbol_name)
        ):
            instr.outputs.clear()


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
    start_regs: RegInfo = RegInfo(stack_info=stack_info)

    arch = global_info.arch
    start_regs[arch.stack_pointer_reg] = GlobalSymbol("sp", type=Type.ptr())
    for reg in arch.saved_regs:
        start_regs[reg] = stack_info.saved_reg_symbol(reg.register_name)

    fn_sym = global_info.address_of_gsym(function.name).expr
    assert isinstance(fn_sym, GlobalSymbol)

    fn_type = fn_sym.type
    fn_type.unify(Type.function())
    fn_sig = Type.ptr(fn_type).get_function_pointer_signature()
    assert fn_sig is not None, "fn_type is known to be a function"
    return_type = fn_sig.return_type
    stack_info.is_variadic = fn_sig.is_variadic

    def make_arg(offset: int, type: Type) -> PassedInArg:
        assert offset % 4 == 0
        return PassedInArg(offset, copied=False, stack_info=stack_info, type=type)

    abi = arch.function_abi(
        fn_sig,
        likely_regs={reg: True for reg in arch.argument_regs},
        for_call=False,
    )
    for slot in abi.arg_slots:
        stack_info.add_known_param(slot.offset, slot.name, slot.type)
        if slot.reg is not None:
            start_regs.set_with_meta(
                slot.reg, make_arg(slot.offset, slot.type), RegMeta(uninteresting=True)
            )
    for slot in abi.possible_slots:
        if slot.reg is not None:
            start_regs.set_with_meta(
                slot.reg, make_arg(slot.offset, slot.type), RegMeta(uninteresting=True)
            )

    if options.reg_vars == ["saved"]:
        reg_vars = arch.saved_regs
    elif options.reg_vars == ["most"]:
        reg_vars = arch.saved_regs + arch.simple_temp_regs
    elif options.reg_vars == ["all"]:
        reg_vars = arch.saved_regs + arch.simple_temp_regs + arch.argument_regs
    else:
        reg_vars = list(map(Register, options.reg_vars))
    for reg in reg_vars:
        stack_info.add_register_var(reg)

    if options.debug:
        print(stack_info)
        print("\nNow, we attempt to translate:")

    used_phis: List[PhiExpr] = []
    return_blocks: List[BlockInfo] = []
    translate_graph_from_block(
        flow_graph.entry_node(),
        start_regs,
        stack_info,
        used_phis,
        return_blocks,
        options,
    )

    for reg in arch.base_return_regs:
        propagate_register_meta(flow_graph.nodes, reg)

    return_reg: Optional[Register] = None

    if not options.void and not return_type.is_void():
        return_reg = determine_return_register(
            return_blocks, fn_sym.type_provided, arch
        )

    if return_reg is not None:
        for b in return_blocks:
            if return_reg in b.final_register_states:
                ret_val = b.final_register_states[return_reg]
                ret_val = as_type(ret_val, return_type, True)
                ret_val.use()
                b.return_value = ret_val
    else:
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

    return FunctionInfo(stack_info, flow_graph, return_type, fn_sym)
