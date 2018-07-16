from contextlib import contextmanager
import queue

import typing
from typing import List, Union, Iterator, Optional, Dict, Callable, Any, Set

from parse_instruction import *
from flow_graph import *
from parse_file import *
from translate import *

@contextmanager
def write_if(condition, indent):
    space = ' ' * indent
    print(f'{space}if ({str(condition)})')  # TODO: need str() call?
    print(f'{space}{{')
    yield
    print(f'{space}}}')

@contextmanager
def write_else(indent):
    space = ' ' * indent
    print(f'{space}else')
    print(f'{space}{{')
    yield
    print(f'{space}}}')


def write_node(node: Node, indent: int):
    assert node.block.block_info is not None
    for item in node.block.block_info.to_write:
        print(f'{" " * indent}{item};')


def write_conditional_subgraph(start: ConditionalNode, end: Node, indent: int) -> None:
    """
    Output the subgraph between "start" and "end" at indent level "indent",
    given that "start" is a ConditionalNode; this program will intelligently
    output if/else relationships.
    """
    if_block_info = start.block.block_info
    assert if_block_info is not None

    # If one of the output edges is the end, it's a "fake" if-statement. That
    # is, it actually just resides one indentation level above the start node.
    if start.conditional_edge == end:
        assert start.fallthrough_edge != end  # otherwise two edges point to one node
        if_condition = UnaryOp(op='!', expr=if_block_info.branch_condition)
        # If the conditional edge isn't real, then the "fallthrough_edge" is
        # actually within the inner if-statement. This means we have to negate
        # the fallthrough edge and go down that path.
        with write_if(if_condition, indent):
            write_flowgraph_between(start.fallthrough_edge, end, indent + 4)
    else:
        if_condition = if_block_info.branch_condition
        # We know for sure that the conditional_edge is written as an if-
        # statement.
        with write_if(if_condition, indent):
            write_flowgraph_between(start.conditional_edge, end, indent + 4)
        # If our fallthrough_edge is "real", then we have to write its contents
        # as an else statement.
        if start.fallthrough_edge != end:
            else_block = start.fallthrough_edge.block
            with write_else(indent):
                write_flowgraph_between(start.fallthrough_edge, end, indent + 4)

reachable_without: Dict[typing.Tuple[int, int, int], bool] = {}

def end_reachable_without(start, end, without):
    """Return whether "end" is reachable from "start" if "without" were removed.
    """
    trip = (start.block.index, end.block.index, without.block.index)
    if trip in reachable_without:
        return reachable_without[trip]
    if without == end:
        # Can't get to the end if it is removed.
        ret = False
    if start == end:
        # Already there! (Base case.)
        ret = True
    else:
        assert not isinstance(start, ReturnNode)  # because then, start == end
        if isinstance(start, BasicNode):
            # If the successor is removed, we can't make it. Otherwise, try
            # to reach the end.
            ret = (start.successor != without and
                    end_reachable_without(start.successor, end, without))
        elif isinstance(start, ConditionalNode):
            # If one edge or the other is removed, you have to go the other route.
            if start.conditional_edge == without:
                assert start.fallthrough_edge != without
                ret = end_reachable_without(start.fallthrough_edge, end, without)
            elif start.fallthrough_edge == without:
                ret = end_reachable_without(start.conditional_edge, end, without)
            else:
                # Both routes are acceptable.
                ret = (end_reachable_without(start.conditional_edge, end, without) or
                        end_reachable_without(start.fallthrough_edge, end, without))
    reachable_without[trip] = ret
    return ret

immpdom: Dict[int, Node] = {}

def immediate_postdominator(start: Node, end: Node) -> Node:
    """
    Find the immediate postdominator of "start", where "end" is an exit node
    from the control flow graph.
    """
    if start.block.index in immpdom:
        return immpdom[start.block.index]
    stack: List[Node] = []
    postdominators: List[Node] = []
    stack.append(start)
    while stack:
        # Get potential postdominator.
        node = stack.pop()
        if node.block.index > end.block.index:
            # Don't go beyond the end.
            continue
        # Add children of node.
        if isinstance(node, BasicNode):
            stack.append(node.successor)
        elif isinstance(node, ConditionalNode):
            stack.append(node.conditional_edge)
            stack.append(node.fallthrough_edge)
        # If removing the node means the end becomes unreachable,
        # the node is a postdominator.
        if not end_reachable_without(start, end, node):
            postdominators.append(node)
    assert postdominators  # at least "end" should be a postdominator
    # Get the earliest postdominator
    postdominators.sort(key=lambda node: node.block.index)
    immpdom[start.block.index] = postdominators[0]
    return postdominators[0]


def write_flowgraph_between(start: Node, end: Node, indent: int) -> None:
    """
    Output a section of a flow graph that has already been translated to our
    symbolic AST. All nodes between start and end, including start but NOT end,
    will be printed out using if-else statements and block info at the given
    level of indentation.
    """
    curr_start = start

    # We will split this graph into subgraphs, where the entrance and exit nodes
    # of that subgraph are at the same indentation level. "curr_start" will
    # iterate through these nodes, which are commonly referred to as
    # articulation nodes.
    while curr_start != end:
        # Since curr_start != end, and since there should only be one
        # ReturnNode, double-check that we haven't done anything wrong.
        assert not isinstance(curr_start, ReturnNode)

        # Write the current node.
        write_node(curr_start, indent)

        if isinstance(curr_start, BasicNode):
            # In a BasicNode, the successor is the next articulation node.
            curr_start = curr_start.successor
        elif isinstance(curr_start, ConditionalNode):
            # A ConditionalNode means we need to find the next articulation
            # node. This means we need to find the "immediate postdominator"
            # of the current node, where "postdominator" means we have to go
            # through it, and "immediate" means we aren't skipping any.
            curr_end = immediate_postdominator(curr_start, end)
            # We also need to handle the if-else block here; this does the
            # outputting of the subgraph between curr_start and the next
            # articulation node.
            write_conditional_subgraph(curr_start, curr_end, indent)
            # Move on.
            curr_start = curr_end

def write_flowgraph(flow_graph: FlowGraph) -> None:
    start_node: Node = flow_graph.nodes[0]
    return_node: Node = flow_graph.nodes[-1]
    assert isinstance(return_node, ReturnNode)

    print("Here's the whole function!\n")
    write_flowgraph_between(start_node, return_node, 0)
    write_node(return_node, 0)  # this node is not printed in the above routine
