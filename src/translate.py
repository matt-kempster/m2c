import attr
import traceback

import typing
from typing import List, Union, Iterator, Optional, Dict, Callable, Any

from parse_instruction import *
from flow_graph import *
from parse_file import *

SPECIAL_REGS = [
    'a0', 'a1', 'a2', 'a3',
    's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7',
    'ra'
]

@attr.s
class StackInfo:
    function: Function = attr.ib()
    allocated_stack_size: int = attr.ib(default=0)
    is_leaf: bool = attr.ib(default=True)
    local_vars_region_bottom: int = attr.ib(default=0)
    return_addr_location: int = attr.ib(default=0)
    callee_save_reg_locations: Dict[Register, int] = attr.ib(default={})
    local_vars: List['LocalVar'] = attr.ib(default=[])

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

    def add_local_var(self, var: 'LocalVar'):
        self.local_vars.append(var)

    def __str__(self):
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
            assert isinstance(inst.args[2], NumberLiteral)
            info.allocated_stack_size = -inst.args[2].value
        elif inst.mnemonic == 'sw' and destination.register_name == 'ra':
            # Saving the return address on the stack.
            assert isinstance(inst.args[1], AddressMode)
            assert isinstance(inst.args[1].rhs, Register)
            assert inst.args[1].rhs.register_name == 'sp'
            info.is_leaf = False
            if inst.args[1].lhs:
                assert isinstance(inst.args[1].lhs, NumberLiteral)
                info.return_addr_location = inst.args[1].lhs.value
            else:
                # Note that this should only happen in the rare case that
                # this function only calls subroutines with no arguments.
                info.return_addr_location = 0
        elif (inst.mnemonic == 'sw' and
              destination.is_callee_save() and
              isinstance(inst.args[1], AddressMode) and
              isinstance(inst.args[1].rhs, Register) and
              inst.args[1].rhs.register_name == 'sp'):
            # Initial saving of callee-save register onto the stack.
            assert isinstance(inst.args[1].rhs, Register)
            if inst.args[1].lhs:
                assert isinstance(inst.args[1].lhs, NumberLiteral)
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


@attr.s
class Store:
    size: int = attr.ib()
    source: Any = attr.ib()
    dest: Any = attr.ib()
    float: bool = attr.ib(default=False)

    def __str__(self):
        type = f'(f{self.size})' if self.float else f'(s{self.size})'
        return f'{type} {self.dest} = {self.source}'

@attr.s
class TypeHint:
    type: str = attr.ib()
    value: Any = attr.ib()

    def __str__(self):
        return f'{self.type}({self.value})'

@attr.s
class BinaryOp:
    left = attr.ib()
    op: str = attr.ib()
    right = attr.ib()

    def is_boolean(self):
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
            self.right == NumberLiteral(0)):
            if self.op == '==':
                return self.left.negated().simplify()
            elif self.op == '!=':
                return self.left.simplify()
        return self

    def __str__(self):
        return f'({self.left} {self.op} {self.right})'

@attr.s
class UnaryOp:
    op: str = attr.ib()
    expr = attr.ib()

    def __str__(self):
        return f'{self.op}{self.expr}'

@attr.s
class Cast:
    to_type: str = attr.ib()
    expr = attr.ib()

    def __str__(self):
        return f'({self.to_type}) {self.expr}'

@attr.s
class Return:
    def __str__(self):
        return 'return'

@attr.s
class FuncCall:
    func_name: str = attr.ib()
    args: List[Any] = attr.ib()

    def __str__(self):
        return f'{self.func_name}({", ".join(str(arg) for arg in self.args)})'


def strip_macros(arg):
    if isinstance(arg, Macro):
        return arg.argument
    elif isinstance(arg, AddressMode) and isinstance(arg.lhs, Macro):
        assert arg.lhs.macro_name == 'lo'  # %hi(...)(REG) doesn't make sense.
        return arg.lhs.argument
    else:
        return arg

def format_hex(val: int):
    return format(val, 'x').upper()

@attr.s
class LocalVar:
    value: int = attr.ib()
    # TODO: Definitely need type

    def __str__(self):
        return f'sp{format_hex(self.value)}'

@attr.s
class PassedInArg:
    value: int = attr.ib()
    # type?

    def __str__(self):
        return f'arg{format_hex(self.value)}'

@attr.s
class StructAccess:
    struct_var = attr.ib()
    offset: int = attr.ib()

    def __str__(self):
        return f'{self.struct_var}->unk{format_hex(self.offset)}'

@attr.s
class SubroutineArg:
    value: int = attr.ib()
    # type?

    def __str__(self):
        return f'subroutine_arg{format_hex(self.value)}'


def deref(arg, reg, stack_info: StackInfo):
    if isinstance(arg, AddressMode):
        assert isinstance(arg.rhs, Register)
        if arg.lhs is None:
            location = 0
        else:
            assert isinstance(arg.lhs, NumberLiteral)  # macros were removed
            location = arg.lhs.value
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
                return arg
        else:
            # Struct member is being dereferenced.
            return StructAccess(struct_var=reg[arg.rhs], offset=location)
    elif isinstance(arg, Register):
        # Look up the register's contents.
        return reg[arg]
    else:
        # Keep GlobalSymbols as-is.
        assert isinstance(arg, GlobalSymbol)
        return arg


def load_upper(args, reg):
    if isinstance(args[1], BinOp):
        # Something like "lui REG (lhs >> 16)". Just take "lhs".
        assert args[1].op == '>>'
        assert args[1].rhs == NumberLiteral(16)
        return args[1].lhs
    elif isinstance(args[1], NumberLiteral):
        # Something like "lui 0x1", meaning 0x10000. Shift left and return.
        return BinaryOp(left=args[1], op='<<', right=NumberLiteral(16))
    else:
        # Something like "lui REG %hi(arg)", but we got rid of the macro.
        return args[1]

def handle_ori(args, reg):
    if isinstance(args[1], BinOp):
        # Something like "ori REG (lhs & 0xFFFF)". We (hopefully) already
        # handled this above, but let's put lhs into this register too.
        assert args[1].op == '&'
        assert args[1].rhs == NumberLiteral(0xFFFF)
        return args[1].lhs
    else:
        # Regular bitwise OR.
        return BinaryOp(left=reg[args[0]], op='|', right=args[1])

def handle_addi(args, reg):
    if len(args) == 2:
        # Used to be "addi REG %lo(...)", but we got rid of the macro.
        # Return the former argument of the macro.
        return args[1]
    elif args[1].register_name == 'sp':
        # Adding to sp, i.e. passing an address.
        assert isinstance(args[2], NumberLiteral)
        #return UnaryOp(op='&', expr=LocalVar(args[2]))
        return UnaryOp(op='&', expr=AddressMode(lhs=args[2], rhs=Register('sp')))
    else:
        # Regular binary addition.
        return BinaryOp(left=reg[args[1]], op='+', right=args[2])

@attr.s
class BlockInfo:
    """
    Contains translated assembly code (to_write), the block's branch condition,
    and block's final register states.
    """
    to_write: List[Union[Store, FuncCall]] = attr.ib()
    branch_condition: Optional[Any] = attr.ib()
    final_register_states: Dict[Register, Any] = attr.ib()

    def __str__(self):
        newline = '\n\t'
        return '\n'.join([
            f'To write: {newline.join(str(write) for write in self.to_write)}',
            f'Branch condition: {self.branch_condition}',
            f'Final register states: ' +
            f'{[f"{k}: {v}" for k,v in self.final_register_states.items()]}'])


def make_store(args, reg, stack_info: StackInfo, size: int, float=False):
    assert isinstance(args[0], Register)
    if args[0].register_name in SPECIAL_REGS:
        return None
    else:
        return Store(
            size, source=reg[args[0]], dest=deref(args[1], reg, stack_info), float=float
        )

def translate_block_body(
    block: Block, reg: Dict[Register, Any], stack_info: StackInfo
) -> BlockInfo:
    """
    Given a block and current register contents, return a BlockInfo containing
    the translated AST for that block.
    """
    cases_source_first_expression = {
        # Storage instructions
        'sb': lambda a: make_store(a, reg, stack_info, size=8),
        'sh': lambda a: make_store(a, reg, stack_info, size=16),
        'sw': lambda a: make_store(a, reg, stack_info, size=32),
        # Floating point storage/conversion
        'swc1': lambda a: make_store(a, reg, stack_info, size=32, float=True),
        'sdc1': lambda a: make_store(a, reg, stack_info, size=64, float=True),
    }
    cases_source_first_register = {
        # Floating point moving instruction
        'mtc1': lambda a: TypeHint(type='f32', value=reg[a[0]]),
    }
    cases_branches = {  # TODO! These are wrong.
        # Branch instructions/pseudoinstructions
        'b': lambda a: None,
        'beq': lambda a:  BinaryOp(left=reg[a[0]], op='==', right=reg[a[1]]),
        'bne': lambda a:  BinaryOp(left=reg[a[0]], op='!=', right=reg[a[1]]),
        'beqz': lambda a: BinaryOp(left=reg[a[0]], op='==', right=NumberLiteral(0)),
        'bnez': lambda a: BinaryOp(left=reg[a[0]], op='!=', right=NumberLiteral(0)),
        'blez': lambda a: BinaryOp(left=reg[a[0]], op='<=', right=NumberLiteral(0)),
        'bgtz': lambda a: BinaryOp(left=reg[a[0]], op='>',  right=NumberLiteral(0)),
        'bltz': lambda a: BinaryOp(left=reg[a[0]], op='<',  right=NumberLiteral(0)),
        'bgez': lambda a: BinaryOp(left=reg[a[0]], op='>=', right=NumberLiteral(0)),
    }
    cases_float_branches = {
        # Floating-point branch instructions
        # We don't have to do any work here, since the condition bit was already set.
        'bc1t': lambda a: None,
        'bc1f': lambda a: None,
    }
    cases_jumps = {
        # Unconditional jumps
        'jal': lambda a: a[0],  # not sure what arguments!
        'jr':  lambda a: Return()  # not sure what to return!
    }
    cases_float_comp = {
        # Floating point comparisons
        'c.eq.s': lambda a: BinaryOp(left=reg[a[0]], op='==', right=reg[a[1]]),
        'c.le.s': lambda a: BinaryOp(left=reg[a[0]], op='<=', right=reg[a[1]]),
        'c.lt.s': lambda a: BinaryOp(left=reg[a[0]], op='<',  right=reg[a[1]]),
    }
    cases_special = {
        # Handle these specially to get better debug output.
        # These should be unspecial'd at some point by way of an initial
        # pass-through, similar to the stack-info acquisition step.
        'lui':  lambda a: load_upper(a, reg),
        'ori':  lambda a: handle_ori(a, reg),
        'addi': lambda a: handle_addi(a, reg),
    }
    cases_div = {
        # Div is just weird.
        'div': lambda a: (BinaryOp(left=reg[a[1]], op='%', right=reg[a[2]]),  # hi
                          BinaryOp(left=reg[a[1]], op='/', right=reg[a[2]])), # lo
    }
    cases_destination_first = {
        # Flag-setting instructions
        'slt': lambda a:  BinaryOp(left=reg[a[1]], op='<', right=reg[a[2]]),
        'slti': lambda a: BinaryOp(left=reg[a[1]], op='<', right=a[2]),
        # LRU (non-floating)
        'addu': lambda a:  BinaryOp(left=reg[a[1]], op='+', right=reg[a[2]]),
        'multu': lambda a: BinaryOp(left=reg[a[1]], op='*', right=reg[a[2]]),
        'subu': lambda a:  BinaryOp(left=reg[a[1]], op='-', right=reg[a[2]]),
        'negu': lambda a:  UnaryOp(op='-', expr=reg[a[1]]),
        # Hi/lo register uses (used after division)
        'mfhi': lambda a: reg[Register('hi')],
        'mflo': lambda a: reg[Register('lo')],
        # Floating point arithmetic
        'div.s': lambda a: BinaryOp(left=reg[a[1]], op='/', right=reg[a[2]]),
        # Floating point conversions
        'cvt.d.s': lambda a: Cast(to_type='f64', expr=reg[a[1]]),
        'cvt.s.d': lambda a: Cast(to_type='f32', expr=reg[a[1]]),
        'cvt.w.d': lambda a: Cast(to_type='s32', expr=reg[a[1]]),
        'trunc.w.s': lambda a: Cast(to_type='s32', expr=reg[a[1]]),
        'trunc.w.d': lambda a: Cast(to_type='s32', expr=reg[a[1]]),
        # Bit arithmetic
        'and': lambda a: BinaryOp(left=reg[a[1]], op='&', right=reg[a[2]]),
        'or': lambda a:  BinaryOp(left=reg[a[1]], op='^', right=reg[a[2]]),
        'xor': lambda a: BinaryOp(left=reg[a[1]], op='^', right=reg[a[2]]),

        'andi': lambda a: BinaryOp(left=reg[a[1]], op='&',  right=a[2]),
        'xori': lambda a: BinaryOp(left=reg[a[1]], op='^',  right=a[2]),
        'sll': lambda a:  BinaryOp(left=reg[a[1]], op='<<', right=a[2]),
        'srl': lambda a:  BinaryOp(left=reg[a[1]], op='>>', right=a[2]),
        # Move pseudoinstruction
        'move': lambda a: reg[a[1]],
        # Floating point moving instructions
        'mfc1': lambda a: reg[a[1]],
        # Loading instructions
        'li': lambda a: a[1],
        'lb': lambda a:  deref(a[1], reg, stack_info),
        'lh': lambda a:  deref(a[1], reg, stack_info),
        'lw': lambda a:  deref(a[1], reg, stack_info),
        'lbu': lambda a: deref(a[1], reg, stack_info),
        'lhu': lambda a: deref(a[1], reg, stack_info),
        'lwu': lambda a: deref(a[1], reg, stack_info),
        # Floating point loading instructions
        'lwc1': lambda a: deref(a[1], reg, stack_info),
        'ldc1': lambda a: deref(a[1], reg, stack_info),

        # 'lb': lambda a:  TypeHint(type='s8',  value=deref(a[1], reg, stack_info)),
        # 'lh': lambda a:  TypeHint(type='s16', value=deref(a[1], reg, stack_info)),
        # 'lw': lambda a:  TypeHint(type='s32', value=deref(a[1], reg, stack_info)),
        # 'lbu': lambda a: TypeHint(type='u8',  value=deref(a[1], reg, stack_info)),
        # 'lhu': lambda a: TypeHint(type='u16', value=deref(a[1], reg, stack_info)),
        # 'lwu': lambda a: TypeHint(type='u32', value=deref(a[1], reg, stack_info)),
        # # Floating point loading instructions
        # 'lwc1': lambda a: TypeHint(type='f32', value=deref(a[1], reg, stack_info)),
        # 'ldc1': lambda a: TypeHint(type='f64', value=deref(a[1], reg, stack_info)),
    }
    cases_uniques: Dict[str, Callable[[List[Argument]], Any]] = {
        **cases_source_first_expression,
        **cases_source_first_register,
        **cases_branches,
        **cases_float_branches,
        **cases_jumps,
        **cases_float_comp,
        **cases_special,
        **cases_div,
        **cases_destination_first,
    }
    cases_repeats = {
        # Addition and division, unsigned vs. signed, doesn't matter (?)
        'addiu': 'addi',
        'divu': 'div',
        # Single-precision float addition is the same as regular addition.
        'add.s': 'addu',
        'mul.s': 'multu',
        'sub.s': 'subu',
        # TODO: Deal with doubles differently.
        'add.d': 'addu',
        'div.d': 'div.s',
        'mul.d': 'multu',
        'sub.d': 'subu',
        # Casting (the above applies here too)
        'cvt.d.w': 'cvt.d.s',
        'cvt.s.w': 'cvt.s.d',
        'cvt.w.s': 'cvt.w.d',
        # Floating point comparisons (the above also applies)
        'c.lt.d': 'c.lt.s',
        'c.eq.d': 'c.eq.s',
        'c.le.d': 'c.le.s',
        # Right-shifting.
        'sra': 'srl',
        # Flag setting.
        'sltiu': 'slti',
        'sltu': 'slt',
    }

    to_write: List[Union[Store, FuncCall]] = []
    subroutine_args: List[Store] = []
    branch_condition: Optional[Any] = None
    for instr in block.instructions:
        assert reg[Register('zero')] == NumberLiteral(0)  # sanity check

        # Save the current mnemonic.
        mnemonic = instr.mnemonic
        if mnemonic in ['nop', 'break']:
            continue
        if mnemonic in cases_repeats:
            # Determine "true" mnemonic.
            mnemonic = cases_repeats[mnemonic]

        # HACK: Remove any %hi(...) or %lo(...) macros; we will just put the
        # full value into each intermediate register, because this really
        # doesn't affect program behavior almost ever.
        instr = Instruction(mnemonic, list(map(strip_macros, instr.args)))

        # Figure out what code to generate!
        if mnemonic in cases_source_first_expression:
            # Store a value in a permanent place.
            to_store = cases_source_first_expression[mnemonic](instr.args)
            if to_store is not None and isinstance(to_store.dest, SubroutineArg):
                # About to call a subroutine with this argument.
                subroutine_args.append(to_store)
            elif to_store is not None:
                if (isinstance(to_store.dest, LocalVar) and
                    to_store.dest not in stack_info.local_vars):
                    # Keep track of all local variables.
                    stack_info.add_local_var(to_store.dest)
                # This needs to be written out.
                to_write.append(to_store)

        elif mnemonic in cases_source_first_register:
            # Just 'mtc1'. It's reversed, so we have to specially handle it.
            assert isinstance(instr.args[1], Register)  # could also assert float register
            reg[instr.args[1]] = cases_source_first_register[mnemonic](instr.args)

        elif mnemonic in cases_branches:
            assert branch_condition is None
            branch_condition = cases_branches[mnemonic](instr.args)

        elif mnemonic in cases_float_branches:
            assert branch_condition is None
            assert Register('condition_bit') in reg
            if mnemonic == 'bc1t':
                branch_condition = reg[Register('condition_bit')]
            elif mnemonic == 'bc1f':
                branch_condition = reg[Register('condition_bit')].negated()

        elif mnemonic in cases_jumps:
            result = cases_jumps[mnemonic](instr.args)
            if isinstance(result, Return):
                # Return from the function.
                assert mnemonic == 'jr'
                # TODO: Maybe assert ReturnNode?
                # TODO: Figure out what to return. (Look through $v0 and $f0)
            else:
                # Function call. Well, let's double-check:
                assert mnemonic == 'jal'
                assert isinstance(instr.args[0], GlobalSymbol)
                func_args = []
                for register in map(Register, ['a0', 'a1', 'a2', 'a3']):
                    # The latter check verifies that the register is not a
                    # placeholder.
                    if register in reg and reg[register] != GlobalSymbol(register.register_name):
                        func_args.append(reg[register])
                # Add the arguments after a3.
                subroutine_args.sort(key=lambda a: a.dest.value)
                for arg in subroutine_args:
                    func_args.append(arg.source)
                # Reset subroutine_args, for the next potential function call.
                subroutine_args = []

                call = FuncCall(instr.args[0].symbol_name, func_args)
                # TODO: It doesn't make sense to put this function call in
                # to_write in all cases, since sometimes it's just the
                # return value which matters.
                to_write.append(call)
                # We don't know what this function's return register is,
                # be it $v0, $f0, or something else, so this hack will have
                # to do. (TODO: handle it...)
                reg[Register('f0')] = call
                reg[Register('v0')] = call

        elif mnemonic in cases_float_comp:
            reg[Register('condition_bit')] = cases_float_comp[mnemonic](instr.args)

        elif mnemonic in cases_special:
            assert isinstance(instr.args[0], Register)
            reg[instr.args[0]] = cases_special[mnemonic](instr.args)

        elif mnemonic in cases_div:
            reg[Register('hi')], reg[Register('lo')] = cases_div[mnemonic](instr.args)

        elif mnemonic in cases_destination_first:
            assert isinstance(instr.args[0], Register)
            reg[instr.args[0]] = cases_destination_first[mnemonic](instr.args)

        else:
            assert False, f"I don't know how to handle {mnemonic}!"

    return BlockInfo(to_write, branch_condition, reg)


def translate_graph_from_block(
    node: Node, reg: Dict[Register, Any], stack_info: StackInfo
) -> None:
    """
    Given a FlowGraph node and a dictionary of register contents, give that node
    its appropriate BlockInfo (which contains the AST of its code).
    """
    # Do not recalculate block info.
    if node.block.block_info is not None:
        return

    print(f'\nNode in question: {node.block}')

    # Translate the given node and discover final register states.
    try:
        block_info = translate_block_body(node.block, reg, stack_info)
        print(block_info)
    except Exception as _:
        traceback.print_exc()
        block_info = BlockInfo([], None, {})  # TODO: handle issues

    node.block.add_block_info(block_info)

    # Translate descendants recursively. Pass a copy of the dictionary since
    # it will be modified.
    if isinstance(node, BasicNode):
        translate_graph_from_block(node.successor, reg.copy(), stack_info)
    elif isinstance(node, ConditionalNode):
        translate_graph_from_block(node.conditional_edge, reg.copy(), stack_info)
        translate_graph_from_block(node.fallthrough_edge, reg.copy(), stack_info)
    else:
        assert isinstance(node, ReturnNode)

@attr.s
class FunctionInfo:
    stack_info: StackInfo = attr.ib()
    flow_graph: FlowGraph = attr.ib()

def translate_to_ast(function: Function) -> FunctionInfo:
    """
    Given a function, produce a FlowGraph that both contains control-flow
    information and has AST transformations for each block of code and
    branch condition.
    """
    # Initialize info about the function.
    flow_graph: FlowGraph = build_callgraph(function)
    stack_info = get_stack_info(function, flow_graph.nodes[0])

    print(stack_info)
    print('\nNow, we attempt to translate:')
    start_node = flow_graph.nodes[0]
    start_reg: Dict[Register, Any] = {
        Register('zero'): NumberLiteral(0),
        # Add dummy values for callee-save registers and args.
        # TODO: There's a better way to do this; this is screwing up the
        # arguments to function calls.
        **{Register(name): GlobalSymbol(name) for name in SPECIAL_REGS}
    }
    translate_graph_from_block(start_node, start_reg, stack_info)
    return FunctionInfo(stack_info, flow_graph)

