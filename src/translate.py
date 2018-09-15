import attr
import traceback

import typing
from typing import List, Union, Iterator, Optional, Dict, Callable, Tuple, Any

from options import Options
from parse_instruction import *
from flow_graph import *

# TODO: include temporary floating-point registers
CALLER_SAVE_REGS = [
    'a0', 'a1', 'a2', 'a3',
    'f12', 'f14',
    'at',
    't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9',
    'hi', 'lo', 'condition_bit', 'return_reg'
]

SPECIAL_REGS = [
    'a0', 'a1', 'a2', 'a3',
    'f12', 'f14',
    's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7',
    'ra',
    '31'
]


@attr.s
class StackInfo:
    function: Function = attr.ib()
    allocated_stack_size: int = attr.ib(default=0)
    is_leaf: bool = attr.ib(default=True)
    local_vars_region_bottom: int = attr.ib(default=0)
    return_addr_location: int = attr.ib(default=0)
    callee_save_reg_locations: Dict[Register, int] = attr.ib(factory=dict)
    local_vars: List['LocalVar'] = attr.ib(factory=list)

    def in_subroutine_arg_region(self, location: int) -> bool:
        assert not self.is_leaf
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

    def add_local_var(self, var: 'LocalVar') -> None:
        self.local_vars.append(var)
        # Make sure the local vars stay sorted in order on the stack.
        self.local_vars.sort(key=lambda v: v.value)

    def __str__(self) -> str:
        return '\n'.join([
            f'Stack info for function {self.function.name}:',
            f'Allocated stack size: {self.allocated_stack_size}',
            f'Leaf? {self.is_leaf}',
            f'Bottom of local vars region: {self.local_vars_region_bottom}',
            f'Location of return addr: {self.return_addr_location}',
            f'Locations of callee save registers: {self.callee_save_reg_locations}'
        ])

def get_stack_info(function: Function, start_node: Node) -> StackInfo:
    info = StackInfo(function)

    # The goal here is to pick out special instructions that provide information
    # about this function's stack setup.
    for inst in start_node.block.instructions:
        if not inst.args:
            continue

        destination = typing.cast(Register, inst.args[0])

        if inst.mnemonic == 'addiu' and destination.register_name == 'sp':
            # Moving the stack pointer.
            assert isinstance(inst.args[2], AsmLiteral)
            info.allocated_stack_size = -inst.args[2].value
        elif inst.mnemonic == 'sw' and destination.register_name == 'ra':
            # Saving the return address on the stack.
            assert isinstance(inst.args[1], AsmAddressMode)
            assert inst.args[1].rhs.register_name == 'sp'
            info.is_leaf = False
            if inst.args[1].lhs:
                assert isinstance(inst.args[1].lhs, AsmLiteral)
                info.return_addr_location = inst.args[1].lhs.value
            else:
                # Note that this should only happen in the rare case that
                # this function only calls subroutines with no arguments.
                info.return_addr_location = 0
        elif (inst.mnemonic == 'sw' and
              destination.is_callee_save() and
              isinstance(inst.args[1], AsmAddressMode) and
              inst.args[1].rhs.register_name == 'sp'):
            # Initial saving of callee-save register onto the stack.
            assert isinstance(inst.args[1].rhs, Register)
            if inst.args[1].lhs:
                assert isinstance(inst.args[1].lhs, AsmLiteral)
                info.callee_save_reg_locations[destination] = inst.args[1].lhs.value
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
    return format(val, 'x').upper()


@attr.s
class TypeHint:
    type: str = attr.ib()
    value: 'Expression' = attr.ib()

    def __str__(self) -> str:
        return f'{self.type}({self.value})'

@attr.s
class BinaryOp:
    left: 'Expression' = attr.ib()
    op: str = attr.ib()
    right: 'Expression' = attr.ib()

    def is_boolean(self) -> bool:
        return self.op in ['==', '!=', '>', '<', '>=', '<=']

    def negated(self) -> 'BinaryOp':
        assert self.is_boolean()
        return BinaryOp(
            left=self.left,
            op={
                '==': '!=',
                '!=': '==',
                '>' : '<=',
                '<' : '>=',
                '>=':  '<',
                '<=':  '>',
            }[self.op],
            right=self.right
        )

    def simplify(self) -> 'BinaryOp':
        if (isinstance(self.left, BinaryOp) and
            self.left.is_boolean()          and
            self.right == IntLiteral(0)):
            if self.op == '==':
                return self.left.negated().simplify()
            elif self.op == '!=':
                return self.left.simplify()
        return self

    def __str__(self) -> str:
        return f'({self.left} {self.op} {self.right})'

@attr.s
class UnaryOp:
    op: str = attr.ib()
    expr = attr.ib()

    def __str__(self) -> str:
        return f'{self.op}{self.expr}'

@attr.s
class Cast:
    to_type: str = attr.ib()
    expr = attr.ib()

    def __str__(self) -> str:
        return f'({self.to_type}) {self.expr}'

@attr.s
class FuncCall:
    func_name: str = attr.ib()
    args: List['Expression'] = attr.ib()

    def __str__(self) -> str:
        return f'{self.func_name}({", ".join(str(arg) for arg in self.args)})'

@attr.s
class LocalVar:
    value: int = attr.ib()
    # TODO: Definitely need type

    def __str__(self) -> str:
        return f'sp{format_hex(self.value)}'

@attr.s
class PassedInArg:
    value: int = attr.ib()
    # type?

    def __str__(self) -> str:
        return f'arg{format_hex(self.value)}'

@attr.s
class StructAccess:
    struct_var: 'Expression' = attr.ib()
    offset: int = attr.ib()

    def __str__(self) -> str:
        # TODO: don't treat offset == 0 specially if there have been other
        # non-zero-offset accesses for the same struct_var
        if isinstance(self.struct_var, AddressOf):
            if self.offset == 0:
                return f'{self.struct_var.expr}'
            else:
                return f'{self.struct_var.expr}.unk{format_hex(self.offset)}'
        else:
            if self.offset == 0:
                return f'*{self.struct_var}'
            else:
                return f'{self.struct_var}->unk{format_hex(self.offset)}'

@attr.s
class SubroutineArg:
    value: int = attr.ib()
    # type?

    def __str__(self) -> str:
        return f'subroutine_arg{format_hex(self.value)}'

@attr.s(frozen=True)
class GlobalSymbol:
    symbol_name: str = attr.ib()

    def __str__(self):
        return self.symbol_name

@attr.s(frozen=True)
class FloatLiteral:
    value: float = attr.ib()

    def __str__(self) -> str:
        return f'{self.value}f'

@attr.s(frozen=True)
class IntLiteral:
    value: int = attr.ib()

    def __str__(self) -> str:
        if abs(self.value) < 10:
            return str(self.value)
        return hex(self.value)

@attr.s(frozen=True)
class AddressOf:
    expr: 'Expression' = attr.ib()

    def __str__(self):
        return f'&{self.expr}'

@attr.s(frozen=True)
class AddressMode:
    offset: int = attr.ib()
    rhs: Register = attr.ib()

    def __str__(self):
        if self.offset:
            return f'{self.offset}({self.rhs})'
        else:
            return f'({self.rhs})'

@attr.s
class StoreStmt:
    size: int = attr.ib()
    source: 'Expression' = attr.ib()
    dest: 'Expression' = attr.ib()
    float: bool = attr.ib(default=False)

    def __str__(self):
        type = f'(f{self.size})' if self.float else f'(s{self.size})'
        return f'{type} {self.dest} = {self.source};'

@attr.s
class FuncCallStmt:
    expr: FuncCall = attr.ib()

    def __str__(self) -> str:
        return f'{self.expr};'

@attr.s
class CommentStmt:
    contents: str = attr.ib()

    def __str__(self) -> str:
        return f'// {self.contents}'

Expression = Union[
    BinaryOp,
    UnaryOp,
    Cast,
    FuncCall,
    GlobalSymbol,
    IntLiteral,
    AddressOf,
    FloatLiteral,
    LocalVar,
    PassedInArg,
    StructAccess,
    SubroutineArg,
]

Statement = Union[
    StoreStmt,
    FuncCallStmt,
    CommentStmt,
]

@attr.s
class RegInfo:
    contents: Dict[Register, Expression] = attr.ib(factory=dict)
    wrote_return_register = attr.ib(default=False)

    def __getitem__(self, key: Register) -> Expression:
        return self.contents[key]

    def __contains__(self, key: Register) -> bool:
        return key in self.contents

    def __setitem__(self, key: Register, value: Optional[Expression]) -> None:
        assert key != Register('zero')
        if value is not None:
            self.contents[key] = value
        elif key in self.contents:
            del self.contents[key]
        if key.register_name in ['f0', 'v0']:
            self[Register('return_reg')] = value
            self.wrote_return_register = True

    def __delitem__(self, key: Register) -> None:
        assert key != Register('zero')
        del self.contents[key]

    def clear_caller_save_regs(self) -> None:
        for reg in map(Register, CALLER_SAVE_REGS):
            assert reg != Register('zero')
            if reg in self.contents:
                del self.contents[reg]

    def copy(self) -> 'RegInfo':
        return RegInfo(contents=self.contents.copy())

    def __str__(self) -> str:
        return ', '.join(f"{k}: {v}" for k,v in sorted(self.contents.items()))


@attr.s
class BlockInfo:
    """
    Contains translated assembly code (to_write), the block's branch condition,
    and block's final register states.
    """
    to_write: List[Statement] = attr.ib()
    branch_condition: Optional[BinaryOp] = attr.ib()
    final_register_states: RegInfo = attr.ib()

    def __str__(self) -> str:
        newline = '\n\t'
        return '\n'.join([
            f'To write: {newline.join(str(write) for write in self.to_write)}',
            f'Branch condition: {self.branch_condition}',
            f'Final register states: {self.final_register_states}'])


@attr.s
class InstrArgs:
    raw_args: List[Argument] = attr.ib()
    regs: RegInfo = attr.ib(repr=False)

    def reg_ref(self, index: int) -> Register:
        ret = self.raw_args[index]
        assert isinstance(ret, Register)
        return ret

    def reg(self, index: int) -> Expression:
        return self.regs[self.reg_ref(index)]

    def imm(self, index: int) -> Expression:
        ret = literal_expr(self.raw_args[index])
        if isinstance(ret, GlobalSymbol):
            return AddressOf(ret)
        return ret

    def memory_ref(self, index: int) -> Union[AddressMode, GlobalSymbol]:
        ret = self.raw_args[index]
        if isinstance(ret, AsmAddressMode):
            if ret.lhs is None:
                return AddressMode(offset=0, rhs=ret.rhs)
            assert isinstance(ret.lhs, AsmLiteral)  # macros were removed
            return AddressMode(offset=ret.lhs.value, rhs=ret.rhs)
        assert isinstance(ret, AsmGlobalSymbol)
        return GlobalSymbol(symbol_name=ret.symbol_name)

    def count(self) -> int:
        return len(self.raw_args)


def deref(
    arg: Union[AddressMode, GlobalSymbol],
    regs: RegInfo,
    stack_info: StackInfo
) -> Expression:
    if isinstance(arg, AddressMode):
        location=arg.offset
        if arg.rhs.register_name == 'sp':
            # This is either a local variable or an argument.
            if stack_info.in_local_var_region(location):
                return LocalVar(location)
            elif stack_info.location_above_stack(location):
                return PassedInArg(location)
            elif stack_info.in_subroutine_arg_region(location):
                return SubroutineArg(location)
            else:
                # Some annoying bookkeeping instruction. To avoid
                # further special-casing, just return whatever - it won't matter.
                return LocalVar(location)
        else:
            # Struct member is being dereferenced.
            return StructAccess(struct_var=regs[arg.rhs], offset=location)
    else:
        # Keep GlobalSymbols as-is.
        assert isinstance(arg, GlobalSymbol)
        return arg

def literal_expr(arg: Argument) -> Expression:
    if isinstance(arg, AsmGlobalSymbol):
        return GlobalSymbol(symbol_name=arg.symbol_name)
    if isinstance(arg, AsmLiteral):
        return IntLiteral(arg.value)
    assert isinstance(arg, BinOp), f'argument {arg} must be a literal'
    return BinaryOp(left=literal_expr(arg.lhs), op=arg.op,
            right=literal_expr(arg.rhs))


def load_upper(args: InstrArgs, regs: RegInfo) -> Expression:
    expr = args.imm(1)
    if isinstance(expr, BinaryOp) and expr.op == '>>':
        # Something like "lui REG (lhs >> 16)". Just take "lhs".
        assert expr.right == IntLiteral(16)
        return expr.left
    elif isinstance(expr, IntLiteral):
        # Something like "lui 0x1", meaning 0x10000.
        return IntLiteral(expr.value << 16)
    else:
        # Something like "lui REG %hi(arg)", but we got rid of the macro.
        return expr

def handle_ori(args: InstrArgs, regs: RegInfo) -> Expression:
    if args.count() == 3:
        return BinaryOp(left=args.reg(1), op='|', right=args.imm(2))

    # Special 2-argument form.
    expr = args.imm(1)
    if isinstance(expr, BinaryOp):
        # Something like "ori REG (lhs & 0xFFFF)". We (hopefully) already
        # handled this above, but let's put lhs into this register too.
        assert expr.op == '&'
        assert expr.right == IntLiteral(0xFFFF)
        return expr.left
    else:
        # Regular bitwise OR.
        return BinaryOp(left=args.reg(0), op='|', right=expr)

def handle_addi(args: InstrArgs, regs: RegInfo) -> Expression:
    if args.count() == 2:
        # Used to be "addi reg1 reg2 %lo(...)", but we got rid of the macro.
        # Return the former argument of the macro.
        return args.imm(1)
    elif args.reg_ref(1).register_name == 'zero':
        # addiu $reg $zero <imm> is one way of writing 'li'
        return args.imm(2)
    elif args.reg_ref(1).register_name == 'sp':
        # Adding to sp, i.e. passing an address.
        lit = args.imm(2)
        assert isinstance(lit, IntLiteral)
        return AddressOf(LocalVar(lit.value))
    else:
        # Regular binary addition.
        return BinaryOp(left=args.reg(1), op='+', right=args.imm(2))

def make_store(
    args: InstrArgs, stack_info: StackInfo, size: int, float=False
) -> Optional[StoreStmt]:
    source_reg = args.reg_ref(0)
    source_val = args.reg(0)
    target = args.memory_ref(1)
    if (source_reg.register_name in SPECIAL_REGS and
            isinstance(target, AddressMode) and
            target.rhs.register_name == 'sp'):
        # TODO: This isn't really right, but it helps get rid of some pointless stores.
        return None
    return StoreStmt(
        size, source=source_val, dest=deref(target, args.regs, stack_info), float=float
    )

def convert_to_float(num: int):
    if num == 0:
        return 0.0
    rep =  f'{num:032b}'  # zero-padded binary representation of num
    dec = lambda x: int(x, 2)  # integer value for num
    sign = dec(rep[0])
    expo = dec(rep[1:9])
    frac = dec(rep[9:])
    return ((-1) ** sign) * (2 ** (expo - 127)) * (frac / (2 ** 23) + 1)

def handle_mtc1(source: Expression) -> Expression:
    if isinstance(source, IntLiteral):
        return FloatLiteral(convert_to_float(source.value))
    else:
        return source

def strip_macros(arg: Argument) -> Argument:
    if isinstance(arg, Macro):
        return arg.argument
    elif isinstance(arg, AsmAddressMode) and isinstance(arg.lhs, Macro):
        assert arg.lhs.macro_name == 'lo'  # %hi(...)(REG) doesn't make sense.
        return arg.lhs.argument
    else:
        return arg


def translate_block_body(
    block: Block, regs: RegInfo, stack_info: StackInfo
) -> BlockInfo:
    """
    Given a block and current register contents, return a BlockInfo containing
    the translated AST for that block.
    """

    InstrMap = Dict[str, Callable[[InstrArgs], Expression]]
    CmpInstrMap = Dict[str, Callable[[InstrArgs], Optional[BinaryOp]]]
    StoreInstrMap = Dict[str, Callable[[InstrArgs], Optional[StoreStmt]]]
    MaybeInstrMap = Dict[str, Callable[[InstrArgs], Optional[Expression]]]
    PairInstrMap = Dict[str, Callable[[InstrArgs], Tuple[Optional[Expression], Optional[Expression]]]]

    cases_source_first_expression: StoreInstrMap = {
        # Storage instructions
        'sb': lambda a: make_store(a, stack_info, size=8),
        'sh': lambda a: make_store(a, stack_info, size=16),
        'sw': lambda a: make_store(a, stack_info, size=32),
        # Floating point storage/conversion
        'swc1': lambda a: make_store(a, stack_info, size=32, float=True),
        'sdc1': lambda a: make_store(a, stack_info, size=64, float=True),
    }
    cases_source_first_register: InstrMap = {
        # Floating point moving instruction
        #'mtc1': lambda a: TypeHint(type='f32', value=a.reg(0)),
        'mtc1': lambda a: handle_mtc1(a.reg(0)),
    }
    cases_branches: CmpInstrMap = {
        # Branch instructions/pseudoinstructions
        # TODO! These are wrong. (Are they??)
        'b': lambda a: None,
        'beq': lambda a:  BinaryOp(left=a.reg(0), op='==', right=a.reg(1)),
        'bne': lambda a:  BinaryOp(left=a.reg(0), op='!=', right=a.reg(1)),
        'beqz': lambda a: BinaryOp(left=a.reg(0), op='==', right=IntLiteral(0)),
        'bnez': lambda a: BinaryOp(left=a.reg(0), op='!=', right=IntLiteral(0)),
        'blez': lambda a: BinaryOp(left=a.reg(0), op='<=', right=IntLiteral(0)),
        'bgtz': lambda a: BinaryOp(left=a.reg(0), op='>',  right=IntLiteral(0)),
        'bltz': lambda a: BinaryOp(left=a.reg(0), op='<',  right=IntLiteral(0)),
        'bgez': lambda a: BinaryOp(left=a.reg(0), op='>=', right=IntLiteral(0)),
    }
    cases_float_branches: CmpInstrMap = {
        # Floating-point branch instructions
        # We don't have to do any work here, since the condition bit was already set.
        'bc1t': lambda a: None,
        'bc1f': lambda a: None,
    }
    cases_jumps: MaybeInstrMap = {
        # Unconditional jumps
        'jal': lambda a: a.imm(0),  # not sure what arguments!
        'jr':  lambda a: None       # not sure what to return!
    }
    cases_float_comp: CmpInstrMap = {
        # Floating point comparisons
        'c.eq.s': lambda a: BinaryOp(left=a.reg(0), op='==', right=a.reg(1)),
        'c.le.s': lambda a: BinaryOp(left=a.reg(0), op='<=', right=a.reg(1)),
        'c.lt.s': lambda a: BinaryOp(left=a.reg(0), op='<',  right=a.reg(1)),
    }
    cases_special: InstrMap = {
        # Handle these specially to get better debug output.
        # These should be unspecial'd at some point by way of an initial
        # pass-through, similar to the stack-info acquisition step.
        'lui':  lambda a: load_upper(a, regs),
        'ori':  lambda a: handle_ori(a, regs),
        'addi': lambda a: handle_addi(a, regs),
    }
    cases_hi_lo: PairInstrMap = {
        # Div and mul output results to LO/HI registers.
        'div': lambda a: (BinaryOp(left=a.reg(1), op='%', right=a.reg(2)),    # hi
                          BinaryOp(left=a.reg(1), op='/', right=a.reg(2))),   # lo
        'multu': lambda a: (None,                                             # hi
                            BinaryOp(left=a.reg(0), op='*', right=a.reg(1))), # lo
    }
    cases_destination_first: InstrMap = {
        # Flag-setting instructions
        'slt': lambda a:  BinaryOp(left=a.reg(1), op='<', right=a.reg(2)),
        'slti': lambda a: BinaryOp(left=a.reg(1), op='<', right=a.imm(2)),
        # LRU (non-floating)
        'addu': lambda a: BinaryOp(left=a.reg(1), op='+', right=a.reg(2)),
        'subu': lambda a: BinaryOp(left=a.reg(1), op='-', right=a.reg(2)),
        'negu': lambda a: UnaryOp(op='-', expr=a.reg(1)),
        # Hi/lo register uses (used after division/multiplication)
        'mfhi': lambda a: regs[Register('hi')],
        'mflo': lambda a: regs[Register('lo')],
        # Floating point arithmetic
        'div.s': lambda a: BinaryOp(left=a.reg(1), op='/', right=a.reg(2)),
        'mul.s': lambda a: BinaryOp(left=a.reg(1), op='*', right=a.reg(2)),
        # Floating point conversions
        'cvt.d.s': lambda a: Cast(to_type='f64', expr=a.reg(1)),
        'cvt.s.d': lambda a: Cast(to_type='f32', expr=a.reg(1)),
        'cvt.w.d': lambda a: Cast(to_type='s32', expr=a.reg(1)),
        'cvt.s.u': lambda a: Cast(to_type='f32',
            expr=Cast(to_type='u32', expr=a.reg(1))),
        'trunc.w.s': lambda a: Cast(to_type='s32', expr=a.reg(1)),
        'trunc.w.d': lambda a: Cast(to_type='s32', expr=a.reg(1)),
        # Bit arithmetic
        'and': lambda a: BinaryOp(left=a.reg(1), op='&', right=a.reg(2)),
        'or': lambda a:  BinaryOp(left=a.reg(1), op='^', right=a.reg(2)),
        'xor': lambda a: BinaryOp(left=a.reg(1), op='^', right=a.reg(2)),

        'andi': lambda a: BinaryOp(left=a.reg(1), op='&',  right=a.imm(2)),
        'xori': lambda a: BinaryOp(left=a.reg(1), op='^',  right=a.imm(2)),
        'sll': lambda a:  BinaryOp(left=a.reg(1), op='<<', right=a.imm(2)),
        'sllv': lambda a: BinaryOp(left=a.reg(1), op='<<', right=a.reg(2)),
        'srl': lambda a:  BinaryOp(left=a.reg(1), op='>>', right=a.imm(2)),
        'srlv': lambda a:  BinaryOp(left=a.reg(1), op='>>', right=a.reg(2)),
        # Move pseudoinstruction
        'move': lambda a: a.reg(1),
        # Floating point moving instructions
        'mfc1': lambda a: a.reg(1),
        'mov.s': lambda a: a.reg(1),
        'mov.d': lambda a: a.reg(1),
        # Loading instructions
        'li': lambda a: a.imm(1),
        'lb': lambda a: deref(a.memory_ref(1), regs, stack_info),
        'lh': lambda a: deref(a.memory_ref(1), regs, stack_info),
        'lw': lambda a: deref(a.memory_ref(1), regs, stack_info),
        'lbu': lambda a: deref(a.memory_ref(1), regs, stack_info),
        'lhu': lambda a: deref(a.memory_ref(1), regs, stack_info),
        'lwu': lambda a: deref(a.memory_ref(1), regs, stack_info),
        # Floating point loading instructions
        'lwc1': lambda a: deref(a.memory_ref(1), regs, stack_info),
        'ldc1': lambda a: deref(a.memory_ref(1), regs, stack_info),
    }
    cases_repeats = {
        # Addition and division, unsigned vs. signed, doesn't matter (?)
        'addiu': 'addi',
        'divu': 'div',
        # Single-precision float addition is the same as regular addition.
        'add.s': 'addu',
        'sub.s': 'subu',
        'neg.s': 'negu',
        # TODO: Deal with doubles differently.
        'add.d': 'addu',
        'sub.d': 'subu',
        'neg.d': 'negu',
        'div.d': 'div.s',
        'mul.d': 'mul.s',
        # Casting (the above applies here too)
        'cvt.d.w': 'cvt.d.s',
        'cvt.s.w': 'cvt.s.d',
        'cvt.w.s': 'cvt.w.d',
        # Floating point comparisons (the above also applies)
        'c.lt.d': 'c.lt.s',
        'c.eq.d': 'c.eq.s',
        'c.le.d': 'c.le.s',
        # Arithmetic right-shifting (TODO: type cast correctly)
        'sra': 'srl',
        'srav': 'srlv',
        # Flag setting.
        'sltiu': 'slti',
        'sltu': 'slt',
        # FCSR-using instructions
        'ctc1': 'mtc1',
        'cfc1': 'mfc1',
    }

    to_write: List[Union[Statement]] = []
    subroutine_args: List[Tuple[Expression, int]] = []
    branch_condition: Optional[BinaryOp] = None
    for instr in block.instructions:
        # Save the current mnemonic.
        mnemonic = instr.mnemonic
        if mnemonic == 'nop':
            continue
        if mnemonic in cases_repeats:
            # Determine "true" mnemonic.
            mnemonic = cases_repeats[mnemonic]

        raw_args = instr.args

        # HACK: Remove any %hi(...) or %lo(...) macros; we will just put the
        # full value into each intermediate register, because this really
        # doesn't affect program behavior almost ever.
        if (mnemonic in ['addi', 'addiu'] and len(raw_args) == 3 and
                isinstance(raw_args[2], Macro)):
            del raw_args[1]
        raw_args = list(map(strip_macros, raw_args))

        args = InstrArgs(raw_args, regs)

        # Figure out what code to generate!
        if mnemonic in cases_source_first_expression:
            # Store a value in a permanent place.
            to_store = cases_source_first_expression[mnemonic](args)
            if to_store is not None and isinstance(to_store.dest, SubroutineArg):
                # About to call a subroutine with this argument.
                subroutine_args.append((to_store.source, to_store.dest.value))
            elif to_store is not None:
                if (isinstance(to_store.dest, LocalVar) and
                    to_store.dest not in stack_info.local_vars):
                    # Keep track of all local variables.
                    stack_info.add_local_var(to_store.dest)
                # This needs to be written out.
                to_write.append(to_store)

        elif mnemonic in cases_source_first_register:
            # Just 'mtc1'. It's reversed, so we have to specially handle it.
            regs[args.reg_ref(1)] = cases_source_first_register[mnemonic](args)

        elif mnemonic in cases_branches:
            assert branch_condition is None
            branch_condition = cases_branches[mnemonic](args)

        elif mnemonic in cases_float_branches:
            assert branch_condition is None
            cond_bit = regs[Register('condition_bit')]
            assert isinstance(cond_bit, BinaryOp)
            if mnemonic == 'bc1t':
                branch_condition = cond_bit
            elif mnemonic == 'bc1f':
                branch_condition = cond_bit.negated()

        elif mnemonic in cases_jumps:
            result = cases_jumps[mnemonic](args)
            if result is None:
                # Return from the function.
                assert mnemonic == 'jr'
                # TODO: Maybe assert ReturnNode?
                # TODO: Figure out what to return. (Look through $v0 and $f0)
            else:
                # Function call. Well, let's double-check:
                assert mnemonic == 'jal'
                target = args.imm(0)
                assert isinstance(target, AddressOf)
                target = target.expr
                assert isinstance(target, GlobalSymbol)
                # At most one of $f12 and $a0 may be passed, and at most one of
                # $f14 and $a1. We could try to figure out which ones, and cap
                # the function call at the point where a register is empty, but
                # for now we'll leave that for manual fixup.
                func_args: List[Expression] = []
                for register in map(Register, ['f12', 'f14', 'a0', 'a1', 'a2', 'a3']):
                    # The latter check verifies that the register is not a
                    # placeholder.
                    if register in regs and regs[register] != GlobalSymbol(register.register_name):
                        func_args.append(regs[register])
                # Add the arguments after a3.
                subroutine_args.sort(key=lambda a: a[1])
                for arg in subroutine_args:
                    func_args.append(arg[0])
                # Reset subroutine_args, for the next potential function call.
                subroutine_args = []

                call = FuncCall(target.symbol_name, func_args)
                # TODO: It doesn't make sense to put this function call in
                # to_write in all cases, since sometimes it's just the
                # return value which matters.
                to_write.append(FuncCallStmt(call))
                # Clear out caller-save registers, for clarity and to ensure
                # that argument regs don't get passed into the next function.
                regs.clear_caller_save_regs()
                # We don't know what this function's return register is,
                # be it $v0, $f0, or something else, so this hack will have
                # to do. (TODO: handle it...)
                regs[Register('f0')] = call
                regs[Register('v0')] = call
                regs[Register('return_reg')] = call

        elif mnemonic in cases_float_comp:
            regs[Register('condition_bit')] = cases_float_comp[mnemonic](args)

        elif mnemonic in cases_special:
            output = args.reg_ref(0)
            res = cases_special[mnemonic](args)

            # Keep track of all local variables that we take addresses of.
            if (output.register_name != 'sp' and
                    isinstance(res, AddressOf) and
                    isinstance(res.expr, LocalVar) and
                    res.expr not in stack_info.local_vars):
                stack_info.add_local_var(res.expr)

            regs[output] = res

        elif mnemonic in cases_hi_lo:
            regs[Register('hi')], regs[Register('lo')] = cases_hi_lo[mnemonic](args)

        elif mnemonic in cases_destination_first:
            regs[args.reg_ref(0)] = cases_destination_first[mnemonic](args)

        else:
            assert False, f"I don't know how to handle {mnemonic}!"

    return BlockInfo(to_write, branch_condition, regs)


def translate_graph_from_block(
    node: Node, regs: RegInfo, stack_info: StackInfo, options: Options
) -> None:
    """
    Given a FlowGraph node and a dictionary of register contents, give that node
    its appropriate BlockInfo (which contains the AST of its code).
    """
    # Do not recalculate block info.
    if node.block.block_info is not None:
        return

    if options.debug:
        print(f'\nNode in question: {node.block}')

    # Translate the given node and discover final register states.
    try:
        block_info = translate_block_body(node.block, regs, stack_info)
        if options.debug:
            print(block_info)
    except Exception as e:  # TODO: handle issues better
        if options.stop_on_error:
            raise e
        traceback.print_exc()
        error_stmt = CommentStmt('Error: ' + str(e).replace('\n', ''))
        block_info = BlockInfo([error_stmt], None, RegInfo(contents={}))

    node.block.add_block_info(block_info)

    # Translate descendants recursively. Pass a copy of the dictionary since
    # it will be modified.
    if isinstance(node, BasicNode):
        translate_graph_from_block(node.successor, regs.copy(),
                stack_info, options)
    elif isinstance(node, ConditionalNode):
        translate_graph_from_block(node.conditional_edge, regs.copy(),
                stack_info, options)
        translate_graph_from_block(node.fallthrough_edge, regs.copy(),
                stack_info, options)
    else:
        assert isinstance(node, ReturnNode)

@attr.s
class FunctionInfo:
    stack_info: StackInfo = attr.ib()
    flow_graph: FlowGraph = attr.ib()

def translate_to_ast(function: Function, options: Options) -> FunctionInfo:
    """
    Given a function, produce a FlowGraph that both contains control-flow
    information and has AST transformations for each block of code and
    branch condition.
    """
    # Initialize info about the function.
    flow_graph: FlowGraph = build_callgraph(function)
    stack_info = get_stack_info(function, flow_graph.nodes[0])

    if options.debug:
        print(stack_info)
        print('\nNow, we attempt to translate:')

    start_node = flow_graph.nodes[0]
    start_reg: RegInfo = RegInfo(contents={
        Register('zero'): IntLiteral(0),
        # Add dummy values for callee-save registers and args.
        # TODO: There's a better way to do this; this is screwing up the
        # arguments to function calls.
        **{Register(name): GlobalSymbol(name) for name in SPECIAL_REGS}
    })
    translate_graph_from_block(start_node, start_reg, stack_info, options)
    return FunctionInfo(stack_info, flow_graph)
