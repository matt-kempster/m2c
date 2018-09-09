import attr

import typing
from typing import List, Union, Iterator, Optional, Dict, Set, Tuple, Callable, Any

from parse_instruction import *
from parse_file import *

@attr.s
class Block:
    index: int = attr.ib()
    label: Optional[Label] = attr.ib()
    instructions: List[Instruction] = attr.ib(factory=list)

    # TODO: fix "Any" to be "BlockInfo" (currently annoying due to circular imports)
    block_info: Optional[Any] = None

    def add_block_info(self, block_info: Any):
        self.block_info = block_info

    def __str__(self):
        if self.label:
            name = f'{self.index} ({self.label.name})'
        else:
            name = self.index
        inst_str = '\n'.join(str(instruction) for instruction in self.instructions)
        return f'# {name}\n{inst_str}\n'

class BlockBuilder:
    def __init__(self):
        self.curr_index: int = 0
        self.curr_label: Optional[Label] = None
        self.curr_instructions: List[Instruction] = []
        self.blocks: List[Block] = []

    def new_block(self) -> Optional[Block]:
        if len(self.curr_instructions) == 0:
            return None

        block = Block(self.curr_index, self.curr_label, self.curr_instructions)
        self.blocks.append(block)

        self.curr_index += 1
        self.curr_label = None
        self.curr_instructions = []

        return block

    def add_instruction(self, instruction: Instruction) -> None:
        self.curr_instructions.append(instruction)

    def set_label(self, label) -> None:
        self.curr_label = label

    def get_blocks(self) -> List[Block]:
        return self.blocks


temp_label_counter = 0
def generate_temp_label():
    global temp_label_counter
    temp_label_counter += 1
    return 'Ltemp' + str(temp_label_counter)


# Branch-likely instructions only evaluate their delay slots when they are
# taken, making control flow more complex. However, on the IRIX compiler they
# only occur in a very specific pattern:
#
# ...
# <branch likely instr> .label
#  X
# ...
# X
# .label:
# ...
#
# which this function transforms back into a regular branch pattern by moving
# the label one step back and replacing the delay slot by a nop.
#
# Branch-likely instructions that do not appear in this pattern are kept.
def normalize_likely_branches(function: Function) -> Function:
    label_prev_instr : Dict[str, Instruction] = {}
    for item in function.body:
        if isinstance(item, Instruction):
            prev_instr = item
        elif isinstance(item, Label):
            label_prev_instr[item.name] = prev_instr

    insert_label_before : Dict[int, str] = {}
    new_body : List[Tuple[Union[Instruction, Label], Union[Instruction, Label]]] = []

    body_iter: Iterator[Union[Instruction, Label]] = iter(function.body)
    for item in body_iter:
        orig_item = item
        if isinstance(item, Instruction) and item.is_branch_likely_instruction():
            before_target = label_prev_instr[item.get_branch_target().target]
            next_item = next(body_iter)
            orig_next_item = next_item
            if isinstance(next_item, Instruction) and str(before_target) == str(next_item):
                if id(before_target) not in insert_label_before:
                    insert_label_before[id(before_target)] = generate_temp_label()
                new_target = JumpTarget(insert_label_before[id(before_target)])
                item = Instruction(item.mnemonic[:-1], item.args[:-1] + [new_target])
                next_item = Instruction('nop', [])
            new_body.append((orig_item, item))
            new_body.append((orig_next_item, next_item))
        else:
            new_body.append((orig_item, item))

    new_function = Function(name=function.name)
    for (orig_item, new_item) in new_body:
        if id(orig_item) in insert_label_before:
            new_function.new_label(insert_label_before[id(orig_item)])
        new_function.body.append(new_item)

    return new_function


def prune_unreferenced_labels(function: Function) -> Function:
    labels_used : Set[str] = set()
    for item in function.body:
        if isinstance(item, Instruction) and item.is_branch_instruction():
            labels_used.add(item.get_branch_target().target)

    new_function = Function(name=function.name)
    for item in function.body:
        if not (isinstance(item, Label) and item.name not in labels_used):
            new_function.body.append(item)

    return new_function


def build_blocks(function: Function) -> List[Block]:
    function = normalize_likely_branches(function)
    function = prune_unreferenced_labels(function)

    block_builder = BlockBuilder()

    body_iter: Iterator[Union[Instruction, Label]] = iter(function.body)
    for item in body_iter:
        if isinstance(item, Label):
            # Split blocks at labels.
            block_builder.new_block()
            block_builder.set_label(item)
        elif isinstance(item, Instruction):
            # TODO: Should this behavior be reverted to its original behavior
            # (leaving the delay slot after the branch/jump)? This way may be
            # harder to test and produce hidden bugs.
            if item.is_delay_slot_instruction():
                # Handle the delay slot by taking the next instruction first.
                # TODO: On -O2-compiled code, the delay slot instruction is
                # sometimes a jump target, which makes things difficult.
                next_item = next(body_iter)
                assert isinstance(next_item, Instruction), "Delay slot instruction must not be a jump target"
                block_builder.add_instruction(next_item)
                # Now we take the original instruction.
            block_builder.add_instruction(item)

            assert not item.is_branch_likely_instruction(), "Not yet able to handle general branch-likely instructions"

            if item.is_branch_instruction():
                # Split blocks at branches.
                block_builder.new_block()
    # Split the last block.
    block_builder.new_block()

    return block_builder.get_blocks()


def is_loop_edge(node: 'Node', edge: 'Node'):
    # Loops are represented by backwards jumps.
    return edge.block.index < node.block.index

@attr.s(frozen=True)
class BasicNode:
    block: Block = attr.ib()
    successor: 'Node' = attr.ib()
    parents: List['Node'] = attr.ib(factory=list)

    def add_parent(self, parent: 'Node'):
        self.parents.append(parent)

    def is_loop(self):
        return is_loop_edge(self, self.successor)

    def __str__(self):
        return ''.join([
            f'{self.block}\n',
            f'# {self.block.index} -> {self.successor.block.index}',
            " (loop)" if self.is_loop() else ""])


@attr.s(frozen=True)
class ConditionalNode:
    block: Block = attr.ib()
    conditional_edge: 'Node' = attr.ib()  # forward-declare types
    fallthrough_edge: 'Node' = attr.ib()
    parents: List['Node'] = attr.ib(factory=list)

    def add_parent(self, parent: 'Node'):
        self.parents.append(parent)

    def is_loop(self):
        return is_loop_edge(self, self.conditional_edge)

    def __str__(self):
        return ''.join([
            f'{self.block}\n',
            f'# {self.block.index} -> ',
            f'cond: {self.conditional_edge.block.index}',
            ' (loop)' if self.is_loop() else '',
            ', ',
            f'def: {self.fallthrough_edge.block.index}'])


@attr.s(frozen=True)
class ReturnNode:
    block: Block = attr.ib()
    parents: List['Node'] = attr.ib(factory=list)

    def add_parent(self, parent: 'Node'):
        self.parents.append(parent)

    def __str__(self):
        return f'{self.block}\n# {self.block.index} -> ret'

Node = Union[
    BasicNode,
    ConditionalNode,
    ReturnNode,
]

def build_graph_from_block(
    block: Block, blocks: List[Block], nodes: List[Node]
) -> Node:
    # Don't reanalyze blocks.
    for node in nodes:
        if node.block == block:
            return node
    new_node: Optional[Node] = None

    def find_block_by_label(label: JumpTarget):
        for block in blocks:
            if block.label and block.label.name == label.target:
                return block

    # Extract branching instructions from this block.
    branches: List[Instruction] = [
        inst for inst in block.instructions if inst.is_branch_instruction()
    ]
    assert len(branches) in [0, 1], "too many branch instructions in one block"

    if len(branches) == 0:
        # No branches, i.e. the next block is this node's successor block.
        successor_block = blocks[block.index + 1]

        # Recursively analyze.
        successor = build_graph_from_block(successor_block, blocks, nodes)
        new_node = BasicNode(block, successor)
        # Keep track of parents.
        successor.add_parent(new_node)
    elif len(branches) == 1:
        # There is a branch, so emit a ConditionalNode.
        branch = branches[0]

        # Get the block associated with the jump target.
        branch_label = branch.get_branch_target()
        branch_block = find_block_by_label(branch_label)
        assert branch_block is not None

        # Recursively analyze.
        branch_node = build_graph_from_block(branch_block, blocks, nodes)
        is_constant_branch = branch.mnemonic == 'b'
        if is_constant_branch:
            # A constant branch becomes a basic edge to our branch target.
            new_node = BasicNode(block, branch_node)
            # Keep track of parents.
            branch_node.add_parent(new_node)
        else:
            # A conditional branch means the fallthrough block is the next block.
            assert len(blocks) > block.index + 1
            fallthrough_block = blocks[block.index + 1]
            # Recursively analyze this too.
            fallthrough_node = build_graph_from_block(fallthrough_block, blocks, nodes)
            new_node = ConditionalNode(block, branch_node, fallthrough_node)
            branch_node.add_parent(new_node)
            # Keep track of parents.
            fallthrough_node.add_parent(new_node)

    assert new_node is not None

    nodes.append(new_node)
    return new_node


def build_nodes(function: Function, blocks: List[Block]):
    # Base case: build the ReturnNode first (to avoid a special case later).
    return_block: Block = blocks[-1]
    return_node: ReturnNode = ReturnNode(return_block)
    graph: List[Node] = [return_node]

    # Traverse through the block tree.
    entrance_block = blocks[0]
    build_graph_from_block(entrance_block, blocks, graph)

    # Sort the nodes by index.
    graph.sort(key=lambda node: node.block.index)
    return graph


@attr.s(frozen=True)
class FlowGraph:
    nodes: List[Node] = attr.ib()


def build_callgraph(function: Function):
    # First build blocks...
    blocks = build_blocks(function)
    # Now build edges.
    nodes = build_nodes(function, blocks)

    return FlowGraph(nodes)


def visualize_callgraph(flow_graph: FlowGraph):
    import graphviz as g
    dot = g.Digraph()
    for node in flow_graph.nodes:
        dot.node(str(node.block.index))
        if isinstance(node, BasicNode):
            dot.edge(str(node.block.index), str(node.successor.block.index), color='green')
        elif isinstance(node, ConditionalNode):
            dot.edge(str(node.block.index), str(node.fallthrough_edge.block.index), color='blue')
            dot.edge(str(node.block.index), str(node.conditional_edge.block.index), color='red')
        else:
            pass
    dot.render('graphviz_render.gv')
