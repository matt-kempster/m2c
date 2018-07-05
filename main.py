import ast
import attr
import re
import string
import sys

import typing
from typing import List, Union, Iterator, Optional

@attr.s(frozen=True)
class Register(object):
    register_name: str = attr.ib()

    def __str__(self):
        return '$%s' % (self.register_name,)

@attr.s(frozen=True)
class GlobalSymbol(object):
    symbol_name: str = attr.ib()

    def __str__(self):
        return '%s' % (self.symbol_name,)

@attr.s(frozen=True)
class Macro(object):
    macro_name: str = attr.ib()
    argument = attr.ib()

    def __str__(self):
        return '%%%s(%s)' % (self.macro_name, self.argument)

@attr.s(frozen=True)
class AddressMode(object):
    lhs = attr.ib()
    rhs = attr.ib()

    def __str__(self):
        if self.lhs is not None:
            return '%s(%s)' % (self.lhs, self.rhs)
        else:
            return '(%s)' % (self.rhs,)

@attr.s(frozen=True)
class NumberLiteral(object):
    value: int = attr.ib()

    def __str__(self):
        return hex(self.value)

@attr.s(frozen=True)
class BinOp(object):
    op = attr.ib()
    lhs = attr.ib()
    rhs = attr.ib()

    def __str__(self):
        return '%s %s %s' % (self.lhs, self.op, self.rhs)

@attr.s(frozen=True)
class JumpTarget(object):
    target: str = attr.ib()

    def __str__(self):
        return '.%s' % (self.target,)

Argument = Union[
    Register,
    GlobalSymbol,
    Macro,
    AddressMode,
    NumberLiteral,
    BinOp,
    JumpTarget
]

valid_word = string.ascii_letters + string.digits + '_'
valid_number = '-x' + string.hexdigits

def parse_word(elems: List[str], valid: str=valid_word) -> str:
    S: str = ''
    while elems and elems[0] in valid:
        S += elems.pop(0)
    return S

def parse_number(elems: List[str]) -> int:
    number_str = parse_word(elems, valid_number)
    return ast.literal_eval(number_str)

# Hacky parser.
def parse_arg_elems(arg_elems: List[str]) -> Optional[Argument]:
    value: Optional[Argument] = None

    def expect(n):
        g = arg_elems.pop(0)
        if g not in n:
            print(f'Expected one of {list(n)}, got {g} (rest: {arg_elems})')
            sys.exit(1)
        return g

    while arg_elems:
        tok: str = arg_elems[0]
        if tok.isspace():
            # Ignore whitespace.
            arg_elems.pop(0)
        elif tok == '$':
            # Register.
            assert value is None
            arg_elems.pop(0)
            value = Register(parse_word(arg_elems))
        elif tok == '.':
            # A jump target (i.e. a label).
            assert value is None
            arg_elems.pop(0)
            value = JumpTarget(parse_word(arg_elems))
        elif tok == '%':
            # A macro (i.e. %hi(...) or %lo(...)).
            assert value is None
            arg_elems.pop(0)
            macro_name = parse_word(arg_elems)
            assert macro_name in ('hi', 'lo')
            expect('(')
            m = parse_arg_elems(arg_elems)
            expect(')')
            return Macro(macro_name, m)
        elif tok == ')':
            break
        elif tok in ('-' + string.digits):
            # Try a number.
            assert value is None
            value = NumberLiteral(parse_number(arg_elems))
        elif tok == '(':
            # Address mode.
            # There was possibly an offset, so value could be a NumberLiteral.
            assert value is None or isinstance(value, NumberLiteral)
            expect('(')
            rhs = parse_arg_elems(arg_elems)
            expect(')')
            value = AddressMode(value, rhs)
        elif tok in valid_word:
            # Global symbol.
            assert value is None
            value = GlobalSymbol(parse_word(arg_elems))
        elif tok in '>+&':
            # Binary operators, used e.g. to modify global symbols or constants.
            assert isinstance(value, NumberLiteral) or isinstance(value, GlobalSymbol)

            if tok == '>':
                expect('>')
                expect('>')
                op = '>>'
            else:
                op = expect('&+')

            rhs = parse_arg_elems(arg_elems)
            # These operators can only use constants as the right-hand-side.
            assert isinstance(rhs, NumberLiteral)
            return BinOp(op, value, rhs)
        else:
            # Unknown.
            print(tok, arg_elems)
            sys.exit(1)

    return value

def parse_arg(arg: str) -> Optional[Argument]:
    arg_elems: List[str] = list(arg)
    return parse_arg_elems(arg_elems)

@attr.s(frozen=True)
class Instruction(object):
    mnemonic: str = attr.ib()
    args: List[Argument] = attr.ib()

    def is_branch_instruction(self):
        return self.mnemonic in ['b', 'beq', 'bne', 'bgez', 'bgtz', 'blez', 'bltz']

    def __str__(self):
        return '    %s %s' % (self.mnemonic, ', '.join(str(arg) for arg in self.args))

def parse_instruction(line: str) -> Instruction:
    # First token is instruction name, rest is args.
    line = line.strip()
    mnemonic, _, args_str = line.partition(' ')
    # Parse arguments.
    args: List[Argument] = list(filter(
        None,
        [parse_arg(arg_str.strip()) for arg_str in args_str.split(',')]))
    return Instruction(mnemonic, args)

@attr.s(frozen=True)
class Label(object):
    name: str = attr.ib()

    def __str__(self):
        return '  .%s:' % (self.name, )

@attr.s
class Function(object):
    name: str = attr.ib()
    body: List[Union[Instruction, Label]] = attr.ib(factory=list)

    def new_label(self, name: str):
        self.body.append(Label(name))

    def new_instruction(self, instruction):
        self.body.append(instruction)

    def __str__(self):
        return 'glabel %s\n%s' % (self.name, '\n'.join(str(item) for item in self.body))

@attr.s
class Program(object):
    filename: str = attr.ib()
    functions: List[Function] = attr.ib(factory=list)
    current_function: Optional[Function] = attr.ib(default=None, repr=False)

    def new_function(self, name: str):
        self.current_function = Function(name=name)
        self.functions.append(self.current_function)

    def new_instruction(self, instruction: Instruction):
        assert self.current_function is not None
        self.current_function.new_instruction(instruction)

    def new_label(self, label_name):
        self.current_function.new_label(label_name)

    def __str__(self):
        return '# %s\n%s' % (self.filename, '\n\n'.join(str(function) for function in self.functions))

@attr.s(frozen=True)
class Block(object):
    index: int = attr.ib()
    label: Optional[Label] = attr.ib()
    instructions: List[Instruction] = attr.ib(factory=list)

    def __str__(self):
        if self.label:
            name = '%s (%s)' % (self.index, self.label.name)
        else:
            name = self.index
        return '# %s\n%s\n' % (name, '\n'.join(str(instruction) for instruction in self.instructions))

# Forward-declare the Node types since they're not defined yet.
def is_loop_edge(node: 'Node', edge: 'Node'):
    # Loops are represented by backwards jumps.
    return edge.block.index < node.block.index

@attr.s(frozen=True)
class BasicNode(object):
    block: Block = attr.ib()
    exit_edge: 'Node' = attr.ib()  # forward-declare type

    def is_loop(self):
        return is_loop_edge(self, self.exit_edge)

    def __str__(self):
        return '%s\n# %d -> %d%s' % (self.block, self.block.index, self.exit_edge.block.index, ' (loop)' if self.is_loop() else '')

@attr.s(frozen=True)
class ExitNode(object):
    block: Block = attr.ib()

    def __str__(self):
        return '%s\n# %d -> ret' % (self.block, self.block.index)

@attr.s(frozen=True)
class ConditionalNode(object):
    block: Block = attr.ib()
    conditional_edge: 'Node' = attr.ib()  # forward-declare types
    fallthrough_edge: 'Node' = attr.ib()

    def is_loop(self):
        return is_loop_edge(self, self.conditional_edge)

    def __str__(self):
        return '%s\n# %d -> cond: %d%s, def: %s' % (self.block, self.block.index, self.conditional_edge.block.index, ' (loop)' if self.is_loop() else '', self.fallthrough_edge.block.index)

Node = Union[
    BasicNode,
    ExitNode,
    ConditionalNode
]

@attr.s(frozen=True)
class FlowAnalysis(object):
    nodes: List[Node] = attr.ib()


def do_flow_analysis(function: Function):
    # Build blocks.
    blocks: List[Block] = []

    curr_index: int = 0
    curr_label: Optional[Label] = None
    curr_instructions: List[Instruction] = []

    def new_block():
        nonlocal curr_index, curr_label, curr_instructions

        if len(curr_instructions) == 0:
            return
        block = Block(curr_index, curr_label, curr_instructions)
        curr_label = None
        curr_index += 1
        blocks.append(block)
        curr_instructions = []

    def take_instruction(instruction: Instruction):
        curr_instructions.append(instruction)

    body_iter: Iterator[Union[Instruction, Label]] = iter(function.body)
    for item in body_iter:
        if isinstance(item, Label):
            # Split blocks at labels.
            new_block()
            curr_label = item
        elif isinstance(item, Instruction):
            take_instruction(item)
            if item.is_branch_instruction():
                # Handle delay slot. Take the next instruction before splitting body.
                # The cast is necessary because we know next() must be an Instruction,
                # but mypy does not.
                take_instruction(typing.cast(Instruction, next(body_iter)))
                new_block()
    new_block()

    # Now build edges.

    # TODO: Is this necessary?
    exit_block: Block = blocks[-1]
    exit_node: ExitNode = ExitNode(exit_block)
    nodes: List[Node] = [exit_node]

    def find_block_by_label(label: JumpTarget):
        for block in blocks:
            if block.label and block.label.name == label.target:
                return block

    def get_block_analysis(block: Block):
        # Don't reanalyze blocks.
        for node in nodes:
            if node.block == block:
                return node
        # Perform the analysis.
        node = do_block_analysis(block)
        nodes.append(node)
        return node

    def do_block_analysis(block: Block) -> Node:
        # Extract branching instructions from this block.
        branches: List[Instruction] = [
            inst for inst in block.instructions if inst.is_branch_instruction()
        ]

        if len(branches) == 0:
            # No branches, i.e. the next block is this node's exit block.
            exit_block = blocks[block.index + 1]

            # Recursively analyze.
            exit_node = get_block_analysis(exit_block)
            return BasicNode(block, exit_node)
        elif len(branches) == 1:
            # There is a branch, so emit a ConditionalNode.
            branch = branches[0]

            # Get the jump target.
            branch_label = branch.args[-1]
            assert isinstance(branch_label, JumpTarget)

            # Get the block associated with the jump target.
            branch_block = find_block_by_label(branch_label)
            assert branch_block is not None

            # Recursively analyze.
            branch_node = get_block_analysis(branch_block)
            is_constant_branch = branch.mnemonic == 'b'
            if is_constant_branch:
                # A constant branch becomes a basic edge to our branch target.
                return BasicNode(block, branch_node)
            else:
                # A conditional branch means the fallthrough block is the next block.
                assert len(blocks) > block.index + 1
                fallthrough_block = blocks[block.index + 1]
                # Recursively analyze this too.
                fallthrough_node = get_block_analysis(fallthrough_block)
                return ConditionalNode(block, branch_node, fallthrough_node)
        else:
            # Shouldn't be possible.
            sys.exit(1)

    # Traverse through the block tree.
    entrance_block = blocks[0]
    get_block_analysis(entrance_block)

    # Sort the nodes chronologically.
    nodes.sort(key=lambda node: node.block.index)
    return FlowAnalysis(nodes)

def decompile(filename: str, f: typing.TextIO) -> None:
    program: Program = Program(filename)

    for line in f:
        # Strip comments and whitespace
        line = re.sub(r'/\*.*\*/', '', line)
        line = re.sub(r'#.*$', '', line)
        line = line.strip()

        if line == '':
            continue
        elif line.startswith('.') and line.endswith(':'):
            # Label.
            label_name: str = line.strip('.:')
            program.new_label(label_name)
        elif line.startswith('.'):
            # Assembler directive.
            pass
        elif line.startswith('glabel'):
            # Function label.
            function_name: str = line.split(' ')[1]
            program.new_function(function_name)
        else:
            # Instruction.
            instruction: Instruction = parse_instruction(line)
            program.new_instruction(instruction)

    print(program.functions[1])

    print("\n\n### FLOW ANALYSIS")
    flow_analysis = do_flow_analysis(program.functions[1])
    for node in flow_analysis.nodes:
        print(node)

def main(filename: str) -> None:
    with open(filename, 'r') as f:
        decompile(filename, f)

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] in ['-h, --help']:
        print(f"USAGE: {sys.argv[0]} [filename]")
    else:
        main(sys.argv[1])
