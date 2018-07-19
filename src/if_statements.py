# from contextlib import contextmanager
import queue

import typing
from typing import List, Union, Iterator, Optional, Dict, Callable, Any, Set

from parse_instruction import *
from flow_graph import *
from parse_file import *
from translate import *

@attr.s
class IfElseStatement:
    condition = attr.ib()
    indent: int = attr.ib()
    if_body = attr.ib()
    else_body = attr.ib(default=None)

    def __str__(self):
        space = ' ' * self.indent
        if_str = '\n'.join([
            f'{space}if ({(self.condition.simplify())})',
            f'{space}{{',
            str(self.if_body),  # has its own indentation
            f'{space}}}',
        ])
        if self.else_body is not None:
            else_str = '\n'.join([
                f'{space}else',
                f'{space}{{',
                str(self.else_body),
                f'{space}}}',
            ])
            if_str = if_str + '\n' + else_str
        return if_str

@attr.s
class Statement:
    indent: int = attr.ib()
    contents = attr.ib()
    is_comment: bool = attr.ib(default=False)

    def __str__(self):
        return f'{" " * self.indent}{self.contents}{"" if self.is_comment else ";"}'

@attr.s
class Body:
    statements: List[Union[Statement, IfElseStatement]] = attr.ib(factory=list)

    def add_node(self, node: Node, indent: int):
        # Add node header comment
        self.statements.append(
            Statement(indent, f'// Node {node.block.index}', is_comment=True))
        # Add node contents
        assert node.block.block_info is not None
        for item in node.block.block_info.to_write:
            self.statements.append(Statement(indent, str(item)))

    def add_statement(self, statement: Statement):
        self.statements.append(statement)

    def add_if_else(self, if_else: IfElseStatement):
        self.statements.append(if_else)

    def __str__(self):
        return '\n'.join([str(statement) for statement in self.statements])


def build_conditional_subgraph(
    flow: FlowGraph, start: ConditionalNode, end: Node, indent: int
) -> IfElseStatement:
    """
    Output the subgraph between "start" and "end" at indent level "indent",
    given that "start" is a ConditionalNode; this program will intelligently
    output if/else relationships.
    """
    if_block_info = start.block.block_info
    assert if_block_info is not None

    # If one of the output edges is the end, it's a "fake" if-statement. That
    # is, it actually just resides one indentation level above the start node.
    else_body = None
    if start.conditional_edge == end:
        assert start.fallthrough_edge != end  # otherwise two edges point to one node
        # If the conditional edge isn't real, then the "fallthrough_edge" is
        # actually within the inner if-statement. This means we have to negate
        # the fallthrough edge and go down that path.
        if_condition = if_block_info.branch_condition.negated()
        if_body = build_flowgraph_between(flow, start.fallthrough_edge, end, indent + 4)
    elif start.fallthrough_edge == end:
        # Only an if block, so this is easy.
        # I think this can only happen in the case where the other branch has
        # an early return.
        if_condition = if_block_info.branch_condition
        if_body = build_flowgraph_between(flow, start.conditional_edge, end, indent + 4)
    else:
        # Both an if and an else block are present. We should write them in
        # chronological order (based on the original MIPS file). The
        # fallthrough edge will always be first, so write it that way.
        if_condition = if_block_info.branch_condition.negated()
        if_body = build_flowgraph_between(flow, start.fallthrough_edge, end, indent + 4)
        else_body = build_flowgraph_between(flow, start.conditional_edge, end, indent + 4)

    return IfElseStatement(if_condition, indent, if_body=if_body, else_body=else_body)

# TODO: Make sure this memoizing dictionary is not used incorrectly, i.e.
# between analyzing different functions. It should likely be scoped differently.
reachable_without: Dict[typing.Tuple[int, int, int], bool] = {}

def end_reachable_without(flow: FlowGraph, start, end, without):
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
            # to reach the end. A small caveat is premature returns. We
            # actually don't want to allow a premature return to occur in
            # this case, because otherwise the immediate postdominator will
            # always end up as the return node.
            is_premature_return = (
                start.successor == flow.nodes[-1] and
                # You'd think a premature return would actually be the block
                # with index = (end_index - 1). That is:
                #       start.block.index != flow.nodes[-1].block.index - 1
                # However, this is not so; some functions have a dead
                # penultimate block with a superfluous unreachable return. The
                # way around this is to just check whether this is the
                # penultimate block, not by index, but by position in the flow
                # graph list:
                start != flow.nodes[-2]
            )
            ret = (start.successor != without and
                    not is_premature_return and
                    end_reachable_without(flow, start.successor, end, without))
        elif isinstance(start, ConditionalNode):
            # If one edge or the other is removed, you have to go the other route.
            if start.conditional_edge == without:
                assert start.fallthrough_edge != without
                ret = end_reachable_without(flow, start.fallthrough_edge, end, without)
            elif start.fallthrough_edge == without:
                ret = end_reachable_without(flow, start.conditional_edge, end, without)
            else:
                # Both routes are acceptable.
                ret = (end_reachable_without(flow, start.conditional_edge, end, without) or
                        end_reachable_without(flow, start.fallthrough_edge, end, without))
    reachable_without[trip] = ret
    return ret

def immediate_postdominator(flow: FlowGraph, start: Node, end: Node) -> Node:
    """
    Find the immediate postdominator of "start", where "end" is an exit node
    from the control flow graph.
    """
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
        if not end_reachable_without(flow, start, end, node):
            postdominators.append(node)
    assert postdominators  # at least "end" should be a postdominator
    # Get the earliest postdominator
    postdominators.sort(key=lambda node: node.block.index)
    return postdominators[0]


def build_flowgraph_between(
    flow: FlowGraph, start: Node, end: Node, indent: int
) -> Body:
    """
    Output a section of a flow graph that has already been translated to our
    symbolic AST. All nodes between start and end, including start but NOT end,
    will be printed out using if-else statements and block info at the given
    level of indentation.
    """
    curr_start = start
    body = Body()

    # We will split this graph into subgraphs, where the entrance and exit nodes
    # of that subgraph are at the same indentation level. "curr_start" will
    # iterate through these nodes, which are commonly referred to as
    # articulation nodes.
    while curr_start != end:
        # Since curr_start != end, and since there should only be one
        # ReturnNode, double-check that we haven't done anything wrong.
        assert not isinstance(curr_start, ReturnNode)

        # Write the current node.
        body.add_node(curr_start, indent)

        if isinstance(curr_start, BasicNode):
            # In a BasicNode, the successor is the next articulation node,
            # unless the node is trying to prematurely return, in which case
            # let it do that.
            if isinstance(curr_start.successor, ReturnNode):
                body.add_statement(Statement(indent, 'return'))
                break
            else:
                curr_start = curr_start.successor
        elif isinstance(curr_start, ConditionalNode):
            # A ConditionalNode means we need to find the next articulation
            # node. This means we need to find the "immediate postdominator"
            # of the current node, where "postdominator" means we have to go
            # through it, and "immediate" means we aren't skipping any.
            curr_end = immediate_postdominator(flow, curr_start, end)
            # We also need to handle the if-else block here; this does the
            # outputting of the subgraph between curr_start and the next
            # articulation node.
            body.add_if_else(
                build_conditional_subgraph(flow, curr_start, curr_end, indent))
            # Move on.
            curr_start = curr_end
    return body

def write_function(function_info: FunctionInfo) -> None:
    global reachable_without  # TODO: global variables are bad, this is temporary

    reachable_without = {}
    flow_graph = function_info.flow_graph
    start_node: Node = flow_graph.nodes[0]
    return_node: Node = flow_graph.nodes[-1]
    assert isinstance(return_node, ReturnNode)

    print("Here's the whole function!\n")
    body: Body = build_flowgraph_between(flow_graph, start_node, return_node, 4)
    body.add_node(return_node, 4)  # this node is not created in the above routine

    print(f'{function_info.stack_info.function.name}(...) {{')
    for local_var in function_info.stack_info.local_vars:
        print(f'    (???) {local_var};')
    print(body)
    print('}')
