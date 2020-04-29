import typing
from copy import copy
from typing import List, Optional, Tuple

import attr

from .flow_graph import BasicNode, ConditionalNode, FlowGraph, Node, compute_dominators
from .parse_instruction import Instruction


def unroll_loop(flow_graph: FlowGraph, start: ConditionalNode) -> Optional[FlowGraph]:
    node_1 = start.fallthrough_edge
    node_7 = start.conditional_edge

    if not isinstance(node_1, ConditionalNode):
        return None
    node_2 = node_1.fallthrough_edge
    node_5 = node_1.conditional_edge

    if not isinstance(node_2, BasicNode):
        return None
    node_3 = node_2.successor

    if not (
        isinstance(node_3, ConditionalNode)
        and node_3.is_loop()
        and node_3.conditional_edge.block.index == node_3.block.index
    ):
        return None
    node_4 = node_3.fallthrough_edge

    if not (
        isinstance(node_4, ConditionalNode)
        and node_4.fallthrough_edge.block.index == node_5.block.index
        and node_4.conditional_edge.block.index == node_7.block.index
    ):
        return None

    if not isinstance(node_5, BasicNode):
        return None
    node_6 = node_5.successor

    if not (
        isinstance(node_6, ConditionalNode)
        and node_6.is_loop()
        and node_6.conditional_edge.block.index == node_6.block.index
        and node_6.fallthrough_edge.block.index == node_7.block.index
    ):
        return None

    modified_node_3 = attr.evolve(
        node_3, fallthrough_edge=node_7, marked_to_remove_remainder_op=True,
    )
    modified_node_3.conditional_edge = modified_node_3
    modified_node_2 = attr.evolve(node_2, successor=modified_node_3)
    modified_node_3.parents = [modified_node_2, modified_node_3]
    node_7.parents = [modified_node_3]

    new_instructions_1 = copy(node_1.block.instructions)
    branches = list(
        filter(lambda instr: instr.is_branch_instruction(), new_instructions_1)
    )
    assert len(branches) == 1
    del new_instructions_1[new_instructions_1.index(branches[0])]
    # TODO: also remove & 3 here
    andis = list(filter(lambda instr: instr.mnemonic == "andi", new_instructions_1))
    assert len(andis) == 1
    new_instructions_1[new_instructions_1.index(andis[0])] = Instruction(
        mnemonic="move", args=[andis[0].args[0], andis[0].args[1]]
    )

    new_block_1 = attr.evolve(node_1.block, instructions=new_instructions_1)
    modified_node_1 = attr.evolve(
        node_1.to_basic_node(successor=modified_node_2), block=new_block_1
    )
    modified_node_2.parents = [modified_node_1]

    new_instructions_0 = copy(start.block.instructions)
    branches = list(
        filter(lambda instr: instr.is_branch_instruction(), new_instructions_0)
    )
    assert len(branches) == 1
    del new_instructions_0[new_instructions_0.index(branches[0])]
    new_block_0 = attr.evolve(start.block, instructions=new_instructions_0)
    modified_node_0 = attr.evolve(
        start.to_basic_node(successor=modified_node_1), block=new_block_0
    )
    modified_node_1.parents = [modified_node_0]

    # TODO: does copy() work?
    new_nodes = copy(flow_graph.nodes)
    # TODO: do we need to reinterpret .parents?
    new_nodes[new_nodes.index(node_3)] = modified_node_3
    new_nodes[new_nodes.index(node_2)] = modified_node_2
    new_nodes[new_nodes.index(node_1)] = modified_node_1
    new_nodes[new_nodes.index(start)] = modified_node_0

    del new_nodes[new_nodes.index(node_4)]
    del new_nodes[new_nodes.index(node_5)]
    del new_nodes[new_nodes.index(node_6)]

    compute_dominators(new_nodes)
    return attr.evolve(flow_graph, nodes=new_nodes)


def munge_unrolled_loops(flow_graph: FlowGraph) -> FlowGraph:
    # TODO: This is horrible, probably not what I want.
    # What if knocking out nodes 4, 5, 6 just reveals another
    # set of nodes that look identical? We will incorrectly
    # be merging two adjacent for-loops.
    changed: bool = True
    while changed:
        changed = False
        for node in flow_graph.nodes:
            if not isinstance(node, ConditionalNode):
                continue
            new_flow_graph = unroll_loop(flow_graph, node)
            if new_flow_graph:
                flow_graph = new_flow_graph
                changed = True
                break
    return flow_graph


def munge_flowgraph(flow_graph: FlowGraph) -> FlowGraph:
    return munge_unrolled_loops(flow_graph)
