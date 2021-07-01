from typing import Dict, List, Optional, Tuple, Union

from .flow_graph import (
    BasicNode,
    ConditionalNode,
    FlowGraph,
    Node,
    compute_relations,
)
from .parse_instruction import Instruction


def replace_node_references(
    flow_graph: FlowGraph, replace_this: Node, with_this: Node
) -> None:
    for node_to_modify in flow_graph.nodes:
        node_to_modify.replace_any_children(replace_this, with_this)


def remove_node(flow_graph: FlowGraph, to_delete: Node, new_child: Node) -> None:
    flow_graph.nodes.remove(to_delete)
    replace_node_references(flow_graph, to_delete, new_child)


def replace_node(flow_graph: FlowGraph, replace_this: Node, with_this: Node) -> None:
    replacement_index = flow_graph.nodes.index(replace_this)
    flow_graph.nodes[replacement_index] = with_this
    replace_node_references(flow_graph, replace_this, with_this)


PatternGraph = Dict[int, Union[int, Tuple[int, int]]]

IDO_O2_SIMPLE_LOOP: PatternGraph = {
    0: (1, 7),
    1: (2, 5),
    2: 3,
    3: (4, 3),
    4: (5, 7),
    5: 6,
    6: (7, 6),
}


def detect_pattern(
    pattern: PatternGraph, flow_graph: FlowGraph, start: Node
) -> Optional[Tuple[Node, ...]]:
    indices = [node.block.index for node in flow_graph.nodes]
    assert sorted(indices) == indices, "FlowGraphs should be sorted"

    offset = start.block.index
    for label in pattern.keys():
        try:
            node = flow_graph.nodes[label + offset]
            target = pattern[label]
            if isinstance(target, int):
                if not (
                    isinstance(node, BasicNode)
                    and node.successor is flow_graph.nodes[offset + target]
                ):
                    return None
            else:
                (fallthrough, conditional) = target
                if not (
                    isinstance(node, ConditionalNode)
                    and node.conditional_edge is flow_graph.nodes[offset + conditional]
                    and node.fallthrough_edge is flow_graph.nodes[offset + fallthrough]
                ):
                    return None
        except IndexError:
            return None

    all_nodes_in_pattern = (
        {offset + label for label in pattern.keys()}
        | {offset + label[0] for label in pattern.values() if isinstance(label, tuple)}
        | {offset + label[1] for label in pattern.values() if isinstance(label, tuple)}
    )
    return tuple(
        node
        for i, node in enumerate(flow_graph.nodes)
        if node is not start and i in all_nodes_in_pattern
    )


def remove_and_replace_nodes(flow_graph: FlowGraph, nodes: Tuple[Node, ...]) -> None:
    (node_1, node_2, node_3, node_4, node_5, node_6, node_7) = nodes
    new_node_1 = BasicNode(node_1.block, node_1.emit_goto, node_2)
    replace_node(flow_graph, node_1, new_node_1)
    remove_node(flow_graph, node_4, node_7)
    remove_node(flow_graph, node_5, node_7)
    remove_node(flow_graph, node_6, node_7)  # TODO: assert didn't execute anything?.


def reroll_loop(flow_graph: FlowGraph, start: ConditionalNode) -> bool:
    nodes = detect_pattern(IDO_O2_SIMPLE_LOOP, flow_graph, start)
    if nodes is None:
        return False
    (node_1, node_2, node_3, node_4, node_5, node_6, node_7) = nodes

    def modify_node_1_instructions(instructions: List[Instruction]) -> bool:
        # First, we check that the node has the instructions we
        # think it has.
        branches = [instr for instr in instructions if instr.is_branch_instruction()]
        if len(branches) != 1:
            return False
        andi_instrs = [instr for instr in instructions if instr.mnemonic == "andi"]
        if len(andi_instrs) != 1:
            return False
        # We are now free to modify the instructions, as we have verified
        # that this node fits the criteria.
        instructions.remove(branches[0])
        andi = andi_instrs[0]
        move = Instruction.derived("move", [andi.args[0], andi.args[1]], andi)
        instructions[instructions.index(andi)] = move
        return True

    if not modify_node_1_instructions(node_1.block.instructions):
        return False

    remove_and_replace_nodes(flow_graph, nodes)

    return True


def reroll_loops(flow_graph: FlowGraph) -> FlowGraph:
    changed: bool = True
    while changed:
        changed = False
        for node in flow_graph.nodes:
            if not isinstance(node, ConditionalNode):
                continue
            changed = reroll_loop(flow_graph, node)
            if changed:
                compute_relations(flow_graph.nodes)
                break
    return flow_graph
