import attr

import typing
from typing import List, Union, Iterator, Optional, Dict, Callable, Any

from parse_instruction import *
from parse_file import *


@attr.s(frozen=True)
class Block:
    index: int = attr.ib()
    label: Optional[Label] = attr.ib()
    instructions: List[Instruction] = attr.ib(factory=list)

    # block_info: Optional['BlockInfo'] = None
    #
    # def add_block_info(self, block_info: 'BlockInfo'):
    #     self.block_info = block_info

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


def build_blocks(function: Function) -> List[Block]:
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
                block_builder.add_instruction(typing.cast(Instruction, next(body_iter)))
                # Now we take the original instruction.
            block_builder.add_instruction(item)

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
        branch_node = build_graph_from_block(branch_block, blocks, nodes)
        is_constant_branch = branch.mnemonic == 'b'
        if is_constant_branch:
            # A constant branch becomes a basic edge to our branch target.
            new_node = BasicNode(block, branch_node)
        else:
            # A conditional branch means the fallthrough block is the next block.
            assert len(blocks) > block.index + 1
            fallthrough_block = blocks[block.index + 1]
            # Recursively analyze this too.
            fallthrough_node = build_graph_from_block(fallthrough_block, blocks, nodes)
            new_node = ConditionalNode(block, branch_node, fallthrough_node)

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
