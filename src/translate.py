import abc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import sys
import traceback
import typing
from typing import (
    AbstractSet,
    Callable,
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

from .asm_file import AsmData, AsmDataEntry
from .asm_instruction import (
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmLiteral,
    BinOp,
    Macro,
    Register,
)
from .c_types import CType, TypeMap
from .demangle_codewarrior import parse as demangle_codewarrior_parse, CxxSymbol
from .error import DecompFailure
from .expression import (
    AddressOf,
    ArrayAccess,
    BinaryOp,
    Cast,
    CommentExpr,
    CommentStmt,
    Condition,
    ErrorExpr,
    EvalOnceExpr,
    EvalOnceStmt,
    ExprStmt,
    Expression,
    FuncCall,
    GlobalSymbol,
    Literal,
    LocalVar,
    PassedInArg,
    PhiExpr,
    RegisterVar,
    SecondF64Half,
    SetPhiStmt,
    StackInfoBase,
    Statement,
    StoreStmt,
    StructAccess,
    SubroutineArg,
    SwitchControl,
    Var,
    as_type,
    as_function_ptr,
    early_unwrap,
    unwrap_deep,
    is_trivial_expression,
    elide_literal_casts,
)
from .flow_graph import (
    ArchFlowGraph,
    ConditionalNode,
    FlowGraph,
    Function,
    Node,
    ReturnNode,
    SwitchNode,
    TerminalNode,
    locs_clobbered_until_dominator,
)
from .instruction import (
    Instruction,
    InstrProcessingFailure,
    StackLocation,
    Location,
    current_instr,
)
from .ir_pattern import IrPattern, simplify_ir_patterns
from .options import Formatter, Options, Target
from .types import (
    FunctionParam,
    FunctionSignature,
    StructDeclaration,
    Type,
    TypePool,
)

InstrMap = Mapping[str, Callable[["InstrArgs"], Expression]]
StmtInstrMap = Mapping[str, Callable[["InstrArgs"], Statement]]
CmpInstrMap = Mapping[str, Callable[["InstrArgs"], Condition]]
StoreInstrMap = Mapping[str, Callable[["InstrArgs"], Optional[StoreStmt]]]


class Arch(ArchFlowGraph):
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
    def function_return(self, expr: Expression) -> Dict[Register, Expression]:
        """
        Compute register location(s) & values that will hold the return value
        of the function call `expr`.
        This must have a value for each register in `all_return_regs` in order to stay
        consistent with `Instruction.outputs`. This is why we can't use the
        function's return type, even though it may be more accurate.
        """
        ...

    # These are defined here to avoid a circular import in flow_graph.py
    ir_patterns: List[IrPattern] = []

    def simplify_ir(self, flow_graph: FlowGraph) -> None:
        simplify_ir_patterns(self, flow_graph, self.ir_patterns)


@dataclass
class StackInfo(StackInfoBase):
    function: Function
    global_info: "GlobalInfo"
    flow_graph: FlowGraph
    allocated_stack_size: int = 0
    is_leaf: bool = True
    is_variadic: bool = False
    uses_framepointer: bool = False
    subroutine_arg_top: int = 0
    callee_save_regs: Set[Register] = field(default_factory=set)
    callee_save_reg_region: Tuple[int, int] = (0, 0)
    unique_type_map: Dict[Tuple[str, object], "Type"] = field(default_factory=dict)
    local_vars: List[LocalVar] = field(default_factory=list)
    temp_vars: List[EvalOnceStmt] = field(default_factory=list)
    phi_vars: List[PhiExpr] = field(default_factory=list)
    reg_vars: Dict[Register, RegisterVar] = field(default_factory=dict)
    used_reg_vars: Set[Register] = field(default_factory=set)
    arguments: List[PassedInArg] = field(default_factory=list)
    temp_name_counter: Dict[str, int] = field(default_factory=dict)
    nonzero_accesses: Set[Expression] = field(default_factory=set)
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

    def add_local_var(self, var: LocalVar) -> None:
        if any(v.value == var.value for v in self.local_vars):
            return
        self.local_vars.append(var)
        # Make sure the local vars stay sorted in order on the stack.
        self.local_vars.sort(key=lambda v: v.value)

    def add_argument(self, arg: PassedInArg) -> None:
        if any(a.value == arg.value for a in self.arguments):
            return
        self.arguments.append(arg)
        self.arguments.sort(key=lambda a: a.value)

    def get_argument(self, location: int) -> Tuple[Expression, PassedInArg]:
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

    def record_struct_access(self, ptr: Expression, location: int) -> None:
        if location:
            self.nonzero_accesses.add(unwrap_deep(ptr))

    def has_nonzero_access(self, ptr: Expression) -> bool:
        return unwrap_deep(ptr) in self.nonzero_accesses

    def unique_type_for(self, category: str, key: object, default: Type) -> Type:
        key = (category, key)
        if key not in self.unique_type_map:
            self.unique_type_map[key] = default
        return self.unique_type_map[key]

    def saved_reg_symbol(self, reg_name: str) -> GlobalSymbol:
        sym_name = "saved_reg_" + reg_name
        type = self.unique_type_for("saved_reg", sym_name, Type.any_reg())
        return GlobalSymbol(symbol_name=sym_name, type=type)

    def should_save(self, expr: Expression, offset: Optional[int]) -> bool:
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

    def get_stack_var(self, location: int, *, store: bool) -> Expression:
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

    def maybe_get_register_var(self, reg: Register) -> Optional[RegisterVar]:
        return self.reg_vars.get(reg)

    def add_register_var(self, reg: Register, name: str) -> None:
        type = Type.floatish() if reg.is_float() else Type.intptr()
        self.reg_vars[reg] = RegisterVar(reg=reg, type=type, name=name)

    def use_register_var(self, var: RegisterVar) -> None:
        self.used_reg_vars.add(var.reg)

    def is_stack_reg(self, reg: Register) -> bool:
        if reg == self.global_info.arch.stack_pointer_reg:
            return True
        if reg == self.global_info.arch.frame_pointer_reg:
            return self.uses_framepointer
        return False

    def get_struct_type_map(self) -> Dict[Expression, Dict[int, Type]]:
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
                f"Callee save registers: {self.callee_save_regs}",
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
    callee_saved_offsets: List[int] = []
    # Track simple literal values stored into registers: MIPS compilers need a temp
    # reg to move the stack pointer more than 0x7FFF bytes.
    temp_reg_values: Dict[Register, int] = {}
    for inst in flow_graph.entry_node().block.instructions:
        arch_mnemonic = inst.arch_mnemonic(arch)
        if inst.function_target:
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
            in [
                "mips:sw",
                "mips:swc1",
                "mips:sdc1",
                "ppc:stw",
                "ppc:stmw",
                "ppc:stfd",
                "ppc:psq_st",
            ]
            and isinstance(inst.args[0], Register)
            and inst.args[0] in arch.saved_regs
            and isinstance(inst.args[1], AsmAddressMode)
            and inst.args[1].rhs == arch.stack_pointer_reg
            and (
                inst.args[0] not in info.callee_save_regs
                or arch_mnemonic == "ppc:psq_st"
            )
        ):
            # Initial saving of callee-save register onto the stack.
            if inst.args[0] in (arch.return_address_reg, Register("r0")):
                # Saving the return address on the stack.
                info.is_leaf = False
            # The registers & their stack accesses must be matched up in ArchAsm.parse
            for reg, mem in zip(inst.inputs, inst.outputs):
                if isinstance(reg, Register) and isinstance(mem, StackLocation):
                    assert mem.symbolic_offset is None
                    stack_offset = mem.offset
                    if arch_mnemonic != "ppc:psq_st":
                        # psq_st instructions store the same register as stfd, just
                        # as packed singles instead. Prioritize the stfd.
                        info.callee_save_regs.add(reg)
                    callee_saved_offsets.append(stack_offset)
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
        if callee_saved_offsets:
            callee_saved_offsets.sort()
            bottom = callee_saved_offsets[0]

            # Both IDO & GCC save registers in two subregions:
            # (a) One for double-sized registers
            # (b) One for word-sized registers, padded to a multiple of 8 bytes
            # IDO has (a) lower than (b); GCC has (b) lower than (a)
            # Check that there are no gaps in this region, other than a single
            # 4-byte word between subregions.
            top = bottom
            internal_padding_added = False
            for offset in callee_saved_offsets:
                if offset != top:
                    if not internal_padding_added and offset == top + 4:
                        internal_padding_added = True
                    else:
                        raise DecompFailure(
                            f"Gap in callee-saved word stack region. "
                            f"Saved: {callee_saved_offsets}, "
                            f"gap at: {offset} != {top}."
                        )
                top = offset + 4
            info.callee_save_reg_region = (bottom, top)

            # Subroutine arguments must be at the very bottom of the stack, so they
            # must come after the callee-saved region
            info.subroutine_arg_top = min(info.subroutine_arg_top, bottom)

    # Use a struct to represent the stack layout. If the struct is provided in the context,
    # its fields will be used for variable types & names.
    stack_struct_name = f"_m2c_stack_{function.name}"
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
        stack_struct = StructDeclaration.unknown(
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

    # True if the regdata was assigned by an Instruction marked as in_pattern;
    # it was part of a matched IR pattern but couldn't be elided at the time
    in_pattern: bool = False


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

    def set_active_instruction(self, instr: Optional[Instruction]) -> None:
        self._active_instr = instr

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
                "whose second half is non-static. This is a m2c restriction "
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

    def sym_imm(self, index: int) -> AddressOf:
        arg = self.raw_arg(index)
        assert isinstance(arg, AsmGlobalSymbol)
        return self.stack_info.global_info.address_of_gsym(arg.symbol_name)

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


def uses_expr(expr: Expression, expr_filter: Callable[[Expression], bool]) -> bool:
    if expr_filter(expr):
        return True
    for e in expr.dependencies():
        if uses_expr(e, expr_filter):
            return True
    return False


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


def reg_always_set(node: Node, reg: Register, *, dom_set: bool) -> bool:
    if node.immediate_dominator is None:
        return False
    seen = {node.immediate_dominator}
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
                if reg in instr.outputs:
                    clobbered = False
                elif reg in instr.clobbers:
                    clobbered = True
        if clobbered == True:
            return False
        if clobbered is None:
            stack.extend(n.parents)
    return True


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
            output_reg_name = stack_info.function.reg_formatter.format(phi.reg)
            prefix = f"phi_{output_reg_name}"
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

    # Set `uninteresting` and propagate it, `function_return`, and `in_pattern` forwards.
    # Start by assuming inherited values are all set; they will get unset iteratively,
    # but for cyclic dependency purposes we want to assume them set.
    for n in non_terminal:
        meta = get_block_info(n).final_register_states.get_meta(reg)
        if meta:
            if meta.inherited:
                meta.uninteresting = True
                meta.function_return = True
                meta.in_pattern = True
            else:
                meta.uninteresting |= (
                    meta.is_read or meta.function_return or meta.in_pattern
                )

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
        all_in_pattern = True
        for p in n.parents:
            par_meta = get_block_info(p).final_register_states.get_meta(reg)
            if par_meta:
                all_uninteresting &= par_meta.uninteresting
                all_function_return &= par_meta.function_return
                all_in_pattern &= par_meta.in_pattern
        if meta.uninteresting and not all_uninteresting and not meta.is_read:
            meta.uninteresting = False
            todo.extend(n.children())
        if meta.function_return and not all_function_return:
            meta.function_return = False
            todo.extend(n.children())
        if meta.in_pattern and not all_in_pattern:
            meta.in_pattern = False
            todo.extend(n.children())


def determine_return_register(
    return_blocks: List[BlockInfo], fn_decl_provided: bool, arch: Arch
) -> Optional[Register]:
    """Determine which of the arch's base_return_regs (i.e. v0, f0) is the most
    likely to contain the return value, or if the function is likely void."""

    def priority(block_info: BlockInfo, reg: Register) -> int:
        meta = block_info.final_register_states.get_meta(reg)
        if not meta:
            return 4
        if meta.uninteresting:
            return 2
        if meta.in_pattern:
            return 1
        if meta.function_return:
            return 0
        return 3

    if not return_blocks:
        return None

    best_reg: Optional[Register] = None
    best_prio = -1
    for reg in arch.base_return_regs:
        prios = [priority(b, reg) for b in return_blocks]
        max_prio = max(prios)
        if max_prio == 4:
            # Register is not always set, skip it
            continue
        if max_prio <= 2 and not fn_decl_provided:
            # Register is always read after being written, or comes from a
            # function call; seems unlikely to be an intentional return.
            # Skip it, unless we have a known non-void return type.
            continue
        if max_prio > best_prio:
            best_prio = max_prio
            best_reg = reg
    return best_reg


@dataclass
class NodeState:
    node: Node
    stack_info: StackInfo = field(repr=False)
    regs: RegInfo = field(repr=False)

    local_var_writes: Dict[LocalVar, Tuple[Register, Expression]] = field(
        default_factory=dict
    )
    subroutine_args: Dict[int, Expression] = field(default_factory=dict)
    in_pattern: bool = False

    to_write: List[Union[Statement]] = field(default_factory=list)
    branch_condition: Optional[Condition] = None
    switch_control: Optional[SwitchControl] = None
    has_function_call: bool = False

    def _eval_once(
        self,
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
        elif "_fictive_" in prefix and isinstance(expr, EvalOnceExpr):
            # Avoid creating additional EvalOnceExprs for fictive Registers
            # so they're less likely to appear in the output
            return expr

        assert reuse_var or prefix
        if prefix == "condition_bit":
            prefix = "cond"

        var = reuse_var or Var(self.stack_info, "temp_" + prefix)
        expr = EvalOnceExpr(
            wrapped_expr=expr,
            var=var,
            type=expr.type,
            emit_exactly_once=emit_exactly_once,
            trivial=trivial,
        )
        var.num_usages += 1
        stmt = EvalOnceStmt(expr)
        self.write_statement(stmt)
        self.stack_info.temp_vars.append(stmt)
        return expr

    def _prevent_later_uses(self, expr_filter: Callable[[Expression], bool]) -> None:
        """Prevent later uses of registers whose contents match a callback filter."""
        for r in self.regs.contents.keys():
            data = self.regs.contents.get(r)
            assert data is not None
            expr = data.value
            if not data.meta.force and expr_filter(expr):
                # Mark the register as "if used, emit the expression's once
                # var". We usually always have a once var at this point,
                # but if we don't, create one.
                if not isinstance(expr, EvalOnceExpr):
                    expr = self._eval_once(
                        expr,
                        emit_exactly_once=False,
                        trivial=False,
                        prefix=self.stack_info.function.reg_formatter.format(r),
                    )

                # This write isn't changing the value of the register; it didn't need
                # to be declared as part of the current instruction's inputs/outputs.
                self.regs.unchecked_set_with_meta(
                    r, expr, replace(data.meta, force=True)
                )

    def prevent_later_value_uses(self, sub_expr: Expression) -> None:
        """Prevent later uses of registers that recursively contain a given
        subexpression."""
        # Unused PassedInArg are fine; they can pass the uses_expr test simply based
        # on having the same variable name. If we didn't filter them out here it could
        # cause them to be incorrectly passed as function arguments -- the function
        # call logic sees an opaque wrapper and doesn't realize that they are unused
        # arguments that should not be passed on.
        self._prevent_later_uses(
            lambda e: uses_expr(e, lambda e2: e2 == sub_expr)
            and not (isinstance(e, PassedInArg) and not e.copied)
        )

    def prevent_later_function_calls(self) -> None:
        """Prevent later uses of registers that recursively contain a function call."""
        self._prevent_later_uses(
            lambda e: uses_expr(e, lambda e2: isinstance(e2, FuncCall))
        )

    def prevent_later_reads(self) -> None:
        """Prevent later uses of registers that recursively contain a read."""
        contains_read = lambda e: isinstance(e, (StructAccess, ArrayAccess))
        self._prevent_later_uses(lambda e: uses_expr(e, contains_read))

    def set_reg_without_eval(
        self, reg: Register, expr: Expression, *, function_return: bool = False
    ) -> None:
        self.regs.set_with_meta(
            reg,
            expr,
            RegMeta(in_pattern=self.in_pattern, function_return=function_return),
        )

    def set_reg_with_error(self, reg: Register, error: ErrorExpr) -> None:
        expr = self._eval_once(
            error,
            emit_exactly_once=True,
            trivial=False,
            prefix=self.stack_info.function.reg_formatter.format(reg),
        )
        if reg != Register("zero"):
            self.set_reg_without_eval(reg, expr)

    def set_reg(
        self, reg: Register, expr: Optional[Expression]
    ) -> Optional[Expression]:
        if expr is None:
            if reg in self.regs:
                del self.regs[reg]
            return None

        if isinstance(expr, LocalVar):
            if (
                isinstance(self.node, ReturnNode)
                and self.stack_info.maybe_get_register_var(reg)
                and self.stack_info.in_callee_save_reg_region(expr.value)
                and reg in self.stack_info.callee_save_regs
            ):
                # Elide saved register restores with --reg-vars (it doesn't
                # matter in other cases).
                return None
            if expr in self.local_var_writes:
                # Elide register restores (only for the same register for now,
                # to be conversative).
                orig_reg, orig_expr = self.local_var_writes[expr]
                if orig_reg == reg:
                    expr = orig_expr

        uw_expr = expr
        if not isinstance(expr, Literal):
            expr = self._eval_once(
                expr,
                emit_exactly_once=False,
                trivial=is_trivial_expression(expr),
                prefix=self.stack_info.function.reg_formatter.format(reg),
            )

        if reg == Register("zero"):
            # Emit the expression as is. It's probably a volatile load.
            expr.use()
            self.write_statement(ExprStmt(expr))
        else:
            dest = self.stack_info.maybe_get_register_var(reg)
            if dest is not None:
                self.stack_info.use_register_var(dest)
                # Avoid emitting x = x, but still refresh EvalOnceExpr's etc.
                if not (isinstance(uw_expr, RegisterVar) and uw_expr.reg == reg):
                    source = as_type(expr, dest.type, True)
                    source.use()
                    self.write_statement(StoreStmt(source=source, dest=dest))
                expr = dest
            self.set_reg_without_eval(reg, expr)
        return expr

    def clear_caller_save_regs(self) -> None:
        for reg in self.stack_info.global_info.arch.temp_regs:
            if reg in self.regs:
                del self.regs[reg]

    def maybe_clear_local_var_writes(self, func_args: List[Expression]) -> None:
        # Clear the `local_var_writes` dict if any of the `func_args` contain
        # a reference to a stack var. (The called function may modify the stack,
        # replacing the value we have in `local_var_writes`.)
        for arg in func_args:
            if uses_expr(
                arg,
                lambda expr: isinstance(expr, AddressOf)
                and isinstance(expr.expr, LocalVar),
            ):
                self.local_var_writes.clear()
                return

    def set_branch_condition(self, cond: Condition) -> None:
        assert isinstance(self.node, ConditionalNode)
        assert self.branch_condition is None
        self.branch_condition = cond

    def set_switch_expr(self, expr: Expression) -> None:
        assert isinstance(self.node, SwitchNode)
        assert self.switch_control is None
        self.switch_control = SwitchControl.from_expr(expr)

    def write_statement(self, stmt: Statement) -> None:
        self.to_write.append(stmt)

    def store_memory(
        self, *, source: Expression, dest: Expression, reg: Register
    ) -> None:
        if isinstance(dest, SubroutineArg):
            # About to call a subroutine with this argument. Skip arguments for the
            # first four stack slots; they are also passed in registers.
            if dest.value >= 0x10:
                self.subroutine_args[dest.value] = source
            return

        if isinstance(dest, LocalVar):
            self.stack_info.add_local_var(dest)
            raw_value = source
            if isinstance(raw_value, Cast) and raw_value.reinterpret:
                # When preserving values on the stack across function calls,
                # ignore the type of the stack variable. The same stack slot
                # might be used to preserve values of different types.
                raw_value = raw_value.expr
            self.local_var_writes[dest] = (reg, raw_value)

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
        source.use()
        dest.use()
        self.prevent_later_value_uses(dest)
        self.prevent_later_function_calls()
        self.write_statement(StoreStmt(source=source, dest=dest))

    def make_function_call(
        self, fn_target: Expression, outputs: List[Location]
    ) -> None:
        arch = self.stack_info.global_info.arch
        fn_target = as_function_ptr(fn_target)
        fn_sig = fn_target.type.get_function_pointer_signature()
        assert fn_sig is not None, "known function pointers must have a signature"

        likely_regs: Dict[Register, bool] = {}
        for reg, data in self.regs.contents.items():
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
            elif data.meta.in_pattern:
                # Like `meta.function_return` mentioned above, `meta.in_pattern` will only be
                # accurate for registers set within this basic block.
                likely_regs[reg] = False
            elif isinstance(data.value, PassedInArg) and not data.value.copied:
                likely_regs[reg] = False
            else:
                likely_regs[reg] = True

        abi = arch.function_abi(fn_sig, likely_regs, for_call=True)

        func_args: List[Expression] = []
        for slot in abi.arg_slots:
            if slot.reg:
                expr = self.regs[slot.reg]
            elif slot.offset in self.subroutine_args:
                expr = self.subroutine_args.pop(slot.offset)
            else:
                expr = ErrorExpr(f"Unable to find stack arg {slot.offset:#x} in block")
            func_args.append(
                CommentExpr.wrap(as_type(expr, slot.type, True), prefix=slot.comment)
            )

        for slot in abi.possible_slots:
            assert slot.reg is not None
            func_args.append(self.regs[slot.reg])

        # Add the arguments after a3.
        # TODO: limit this based on abi.arg_slots. If the function type is known
        # and not variadic, this list should be empty.
        for _, arg in sorted(self.subroutine_args.items()):
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
        self.subroutine_args.clear()

        call: Expression = FuncCall(
            fn_target, func_args, fn_sig.return_type.weaken_void_ptr()
        )
        call = self._eval_once(
            call, emit_exactly_once=True, trivial=False, prefix="ret"
        )

        # Clear out caller-save registers, for clarity and to ensure that
        # argument regs don't get passed into the next function.
        self.clear_caller_save_regs()

        # Clear out local var write tracking if any argument contains a stack
        # reference. That dict is used to track register saves/restores, which
        # are unreliable if we call a function with a stack reference.
        self.maybe_clear_local_var_writes(func_args)

        # Prevent reads and function calls from moving across this call.
        # This isn't really right, because this call might be moved later,
        # and then this prevention should also be... but it's the best we
        # can do with the current code architecture.
        self.prevent_later_function_calls()
        self.prevent_later_reads()

        return_reg_vals = arch.function_return(call)
        for out in outputs:
            if not isinstance(out, Register):
                continue
            val = return_reg_vals[out]
            if not isinstance(val, SecondF64Half):
                val = self._eval_once(
                    val,
                    emit_exactly_once=False,
                    trivial=False,
                    prefix=self.stack_info.function.reg_formatter.format(out),
                )
            self.set_reg_without_eval(out, val, function_return=True)

        self.has_function_call = True

    @contextmanager
    def current_instr(self, instr: Instruction) -> Iterator[None]:
        self.regs.set_active_instruction(instr)
        self.in_pattern = instr.in_pattern
        try:
            with current_instr(instr):
                yield
        finally:
            self.regs.set_active_instruction(None)
            self.in_pattern = False


def evaluate_instruction(instr: Instruction, state: NodeState) -> None:
    # Check that instr's attributes are consistent
    if instr.is_return:
        assert isinstance(state.node, ReturnNode)
    if instr.is_conditional:
        assert state.branch_condition is None and state.switch_control is None

    if instr.eval_fn is not None:
        args = InstrArgs(instr.args, state.regs, state.stack_info)
        eval_fn = typing.cast(Callable[[NodeState, InstrArgs], object], instr.eval_fn)
        eval_fn(state, args)

    # Check that conditional instructions set at least one of branch_condition or switch_control
    if instr.is_conditional:
        assert state.branch_condition is not None or state.switch_control is not None


def translate_node_body(node: Node, regs: RegInfo, stack_info: StackInfo) -> BlockInfo:
    """
    Given a node and current register contents, return a BlockInfo containing
    the translated AST for that node.
    """
    state = NodeState(node=node, regs=regs, stack_info=stack_info)

    for instr in node.block.instructions:
        with state.current_instr(instr):
            evaluate_instruction(instr, state)

    if state.branch_condition is not None:
        state.branch_condition.use()
    if state.switch_control is not None:
        state.switch_control.control_expr.use()

    return BlockInfo(
        to_write=state.to_write,
        return_value=None,
        switch_control=state.switch_control,
        branch_condition=state.branch_condition,
        final_register_states=state.regs,
        has_function_call=state.has_function_call,
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

        phi_regs = (
            r for r in locs_clobbered_until_dominator(child) if isinstance(r, Register)
        )
        for reg in phi_regs:
            if reg_always_set(child, reg, dom_set=(reg in regs)):
                expr: Optional[Expression] = stack_info.maybe_get_register_var(reg)
                if expr is None:
                    expr = PhiExpr(
                        reg=reg, node=child, used_phis=used_phis, type=Type.any_reg()
                    )
                new_regs.set_with_meta(reg, expr, RegMeta(inherited=True))
            elif reg in new_regs:
                del new_regs[reg]
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

            # If the symbol is a C++ vtable, try to build a custom type for it by parsing it
            if (
                self.target.language == Target.LanguageEnum.CXX
                and sym_name.startswith("__vt__")
                and sym.asm_data_entry is not None
            ):
                sym.type.unify(self.vtable_type(sym_name, sym.asm_data_entry))

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
                sym.type.unify(
                    Type.demangled_symbol(self.typemap, self.typepool, demangled_symbol)
                )

        return AddressOf(sym, type=sym.type.reference())

    def vtable_type(self, sym_name: str, asm_data_entry: AsmDataEntry) -> Type:
        """
        Parse MWCC vtable data to create a custom struct to represent it.
        This format is not well documented, but is briefly explored in this series of posts:
        https://web.archive.org/web/20220413174849/http://hacksoflife.blogspot.com/2007/02/c-objects-part-2-single-inheritance.html
        """
        size = asm_data_entry.size_range_bytes()[1]
        struct = StructDeclaration.unknown(
            self.typepool, size=size, align=4, tag_name=sym_name
        )
        offset = 0
        for entry in asm_data_entry.data:
            if isinstance(entry, bytes):
                # MWCC vtables start with a pointer to a typeid struct (or NULL) and an offset
                if len(entry) % 4 != 0:
                    raise DecompFailure(
                        f"Unable to parse misaligned vtable data in {sym_name}"
                    )
                for i in range(len(entry) // 4):
                    field_name = f"{struct.new_field_prefix}{offset:X}"
                    struct.try_add_field(
                        Type.reg32(likely_float=False), offset, field_name, size=4
                    )
                    offset += 4
            else:
                entry_name = entry
                try:
                    demangled_field_sym = demangle_codewarrior_parse(entry)
                    if demangled_field_sym.name.qualified_name is not None:
                        entry_name = str(demangled_field_sym.name.qualified_name[-1])
                except ValueError:
                    pass

                field = struct.try_add_field(
                    self.address_of_gsym(entry).type,
                    offset,
                    name=entry_name,
                    size=4,
                )
                assert field is not None
                field.known = True
                offset += 4
        return Type.struct(struct)

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
                    enum_name = type.get_enum_name(value)
                    if enum_name is not None:
                        return enum_name
                    expr = as_type(Literal(value), type, True)
                    return elide_literal_casts(expr).format(fmt)

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

                # TODO: Use original AsmFile ordering for variables
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
                # In modes except "all", skip vtable decls when compiling C++
                if (
                    decls != Options.GlobalDeclsEnum.ALL
                    and self.target.language == Target.LanguageEnum.CXX
                    and name.startswith("__vt__")
                ):
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


def narrow_func_call_outputs(
    function: Function,
    global_info: GlobalInfo,
) -> None:
    """
    Modify the `outputs` list of function call Instructions using the context file.
    For now, this only handles known-void functions, but in the future it could
    be extended to select a specific register subset based on type.
    """
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
        reg_vars = [
            stack_info.function.reg_formatter.parse(x, arch) for x in options.reg_vars
        ]
    for reg in reg_vars:
        reg_name = stack_info.function.reg_formatter.format(reg)
        stack_info.add_register_var(reg, reg_name)

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
