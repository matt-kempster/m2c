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
    if_block_info = start.block.block_info
    assert if_block_info is not None

    if start.conditional_edge == end:
        assert start.fallthrough_edge != end
        if_condition = UnaryOp(op='!', expr=if_block_info.branch_condition)
        with write_if(if_condition, indent):
            write_flowgraph_between(start.fallthrough_edge, end, indent + 4)
    else:
        if_condition = if_block_info.branch_condition
        with write_if(if_condition, indent):
            write_flowgraph_between(start.conditional_edge, end, indent + 4)
        if start.fallthrough_edge != end:
            else_block = start.fallthrough_edge.block
            with write_else(indent):
                write_flowgraph_between(start.fallthrough_edge, end, indent + 4)


def end_reachable_without(start, end, without):
    if without == end:
        return False
    if start == end:
        return True
    else:
        assert not isinstance(start, ReturnNode)  # because then, start == end
        if isinstance(start, BasicNode):
            return (start.successor != without and
                    end_reachable_without(start.successor, end, without))
        elif isinstance(start, ConditionalNode):
            if start.conditional_edge == without:
                assert start.fallthrough_edge != without
                return end_reachable_without(start.fallthrough_edge, end, without)
            elif start.fallthrough_edge == without:
                return end_reachable_without(start.conditional_edge, end, without)
            else:
                return any([
                    end_reachable_without(start.conditional_edge, end, without),
                    end_reachable_without(start.fallthrough_edge, end, without)
                ])

def immediate_postdominator(start: Node, end: Node) -> Node:
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
    return postdominators[0]


def write_flowgraph_between(start: Node, end: Node, indent: int) -> None:
    curr_start = start

    while curr_start != end:
        write_node(curr_start, indent)

        if isinstance(curr_start, BasicNode):
            curr_start = curr_start.successor
        elif isinstance(curr_start, ConditionalNode):
            # curr_end = immediate_postdominator(curr_start.conditional_edge,
            #                                    curr_start.fallthrough_edge)
            curr_end = immediate_postdominator(curr_start, end)
            write_conditional_subgraph(curr_start, curr_end, indent)
            curr_start = curr_end
        else:
            assert isinstance(curr_start, ReturnNode)
            print('// the return node should be here?')

def write_flowgraph(flow_graph: FlowGraph) -> None:
    start_node: Node = flow_graph.nodes[0]
    return_node: Node = flow_graph.nodes[-1]
    assert isinstance(return_node, ReturnNode)

    print("Here's the big one! The whole function!\n")
    write_flowgraph_between(start_node, return_node, 0)
    write_node(return_node, 0)
