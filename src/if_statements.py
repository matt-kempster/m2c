# from contextlib import contextmanager
import queue

import typing
from typing import List, Union, Iterator, Optional, Dict, Callable, Any, Set

from parse_instruction import *
from flow_graph import *
from parse_file import *
from translate import *

@attr.s
class Context:
    flow_graph: FlowGraph = attr.ib()
    reachable_without: Dict[typing.Tuple[int, int, int], bool] = attr.ib(factory=dict)
    can_reach_return = attr.ib(default=False)

@attr.s
class IfElseStatement:
    condition: BinaryOp = attr.ib()
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
class SimpleStatement:
    indent: int = attr.ib()
    contents: str = attr.ib()

    def __str__(self):
        return f'{" " * self.indent}{self.contents}'

@attr.s
class Body:
    statements: List[Union[SimpleStatement, IfElseStatement]] = attr.ib(factory=list)

    def add_node(self, node: Node, indent: int) -> None:
        # Add node header comment
        self.add_comment(indent, f'// Node {node.block.index}')
        # Add node contents
        assert node.block.block_info is not None
        for item in node.block.block_info.to_write:
            self.statements.append(SimpleStatement(indent, str(item)))

    def add_statement(self, statement: SimpleStatement) -> None:
        self.statements.append(statement)

    def add_comment(self, indent: int, contents: str) -> None:
        self.add_statement(SimpleStatement(indent, f'// {contents}'))

    def add_if_else(self, if_else: IfElseStatement) -> None:
        self.statements.append(if_else)

    def __str__(self) -> str:
        return '\n'.join([str(statement) for statement in self.statements])


def build_conditional_subgraph(
    context: Context, start: ConditionalNode, end: Node, indent: int
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
        if_body = build_flowgraph_between(context, start.fallthrough_edge, end, indent + 4)
    elif start.fallthrough_edge == end:
        # Only an if block, so this is easy.
        # I think this can only happen in the case where the other branch has
        # an early return.
        if_condition = if_block_info.branch_condition
        if_body = build_flowgraph_between(context, start.conditional_edge, end, indent + 4)
    else:
        # We need to see if this is a compound if-statement, i.e. containing
        # && or ||.
        conds = get_number_of_if_conditions(context, start, end)
        if conds < 2:  # normal if-statement
            # Both an if and an else block are present. We should write them in
            # chronological order (based on the original MIPS file). The
            # fallthrough edge will always be first, so write it that way.
            if_condition = if_block_info.branch_condition.negated()
            if_body = build_flowgraph_between(context, start.fallthrough_edge, end, indent + 4)
            else_body = build_flowgraph_between(context, start.conditional_edge, end, indent + 4)
        else:  # multiple conditions in if-statement
            return get_full_if_condition(context, conds, start, end, indent)

    return IfElseStatement(if_condition, indent, if_body=if_body, else_body=else_body)

def end_reachable_without(context: Context, start, end, without):
    """Return whether "end" is reachable from "start" if "without" were removed.
    """
    trip = (start.block.index, end.block.index, without.block.index)
    if trip in context.reachable_without:
        return context.reachable_without[trip]
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
                start.successor == context.flow_graph.nodes[-1] and
                # You'd think a premature return would actually be the block
                # with index = (end_index - 1). That is:
                #   start.block.index != flow_graph.nodes[-1].block.index - 1
                # However, this is not so; some functions have a dead
                # penultimate block with a superfluous unreachable return. The
                # way around this is to just check whether this is the
                # penultimate block, not by index, but by position in the flow
                # graph list:
                start != context.flow_graph.nodes[-2]
            )
            ret = (start.successor != without and
                    not is_premature_return and
                    end_reachable_without(context, start.successor, end, without))
        elif isinstance(start, ConditionalNode):
            # If one edge or the other is removed, you have to go the other route.
            if start.conditional_edge == without:
                assert start.fallthrough_edge != without
                ret = end_reachable_without(context, start.fallthrough_edge, end, without)
            elif start.fallthrough_edge == without:
                ret = end_reachable_without(context, start.conditional_edge, end, without)
            else:
                # Both routes are acceptable.
                ret = (end_reachable_without(context, start.conditional_edge, end, without) or
                        end_reachable_without(context, start.fallthrough_edge, end, without))
    context.reachable_without[trip] = ret
    return ret

def immediate_postdominator(context: Context, start: Node, end: Node) -> Node:
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
        if not end_reachable_without(context, start, end, node):
            postdominators.append(node)
    assert postdominators  # at least "end" should be a postdominator
    # Get the earliest postdominator
    postdominators.sort(key=lambda node: node.block.index)
    return postdominators[0]


def count_non_postdominated_parents(context, child, curr_end):
    """
    Return the number of parents of "child" for whom "child" is NOT their
    immediate postdominator. This is useful for finding nodes that would be
    printed more than once under naive assumptions, i.e. if-conditions that
    contain multiple predicates in the form of && or ||.
    """
    count = 0
    for parent in child.parents:
        if immediate_postdominator(context, parent, curr_end) != child:
            count += 1
    # Either all this node's parents are immediately postdominated by it,
    # or none of them are. To be honest, I don't have much evidence for
    # this assertion, but if it fails, then the output of && and || will
    # likely be incorrect. (A suitable TODO, perhaps, is to prove this
    # mathematically.)
    assert count in [0, len(child.parents)]
    return count


def get_number_of_if_conditions(context, node, curr_end):
    """
    For a given ConditionalNode, this function will return k when the if-
    statement of the correspondant C code is "if (1 && 2 && ... && k)" or
    "if (1 || 2 || ... || k)", where the numbers are labels for clauses.
    (It remains unclear how a predicate that mixes && and || would behave.)
    """
    count1, count2 = map(
        lambda child: count_non_postdominated_parents(context, child, curr_end),
        [node.conditional_edge, node.fallthrough_edge]
    )
    # Return the nonzero count; the predicates will go through that path.
    # (TODO: I have a theory that we can just return count2 here.)
    if count1 != 0:
        return count1
    else:
        return count2

def join_conditions(conditions: List[BinaryOp], op: str, only_negate_last: bool):
    assert op in ['&&', '||']
    final_cond: Optional[BinaryOp] = None
    for i, cond in enumerate(conditions):
        if not only_negate_last or i == len(conditions) - 1:
            cond = cond.negated()
        if final_cond is None:
            final_cond = cond
        else:
            final_cond = BinaryOp(final_cond, op, cond)
    return final_cond

def get_full_if_condition(
    context: Context,
    count: int,
    start: ConditionalNode,
    curr_end,
    indent
) -> IfElseStatement:
    curr_node: Node = start
    prev_node: Optional[ConditionalNode] = None
    conditions: List[BinaryOp] = []
    # Get every condition.
    while count > 0:
        block_info = curr_node.block.block_info
        assert isinstance(block_info, BlockInfo)
        assert block_info.branch_condition is not None
        conditions.append(block_info.branch_condition)
        assert isinstance(curr_node, ConditionalNode)
        prev_node = curr_node
        curr_node = curr_node.fallthrough_edge
        count -= 1
    # At the end, if we end up at the conditional-edge after the very start,
    # then we know this was an || statement - if the start condition were true,
    # we would have skipped ahead to the body.
    if curr_node == start.conditional_edge:
        assert prev_node is not None
        return IfElseStatement(
            # Negate the last condition, for it must fall-through to the
            # body instead of jumping to it, hence it must jump OVER the body.
            join_conditions(conditions, '||', only_negate_last=True),
            indent,
            if_body=build_flowgraph_between(
                context, start.conditional_edge, curr_end, indent + 4),
            # The else-body is wherever the code jumps to instead of the
            # fallthrough (i.e. if-body).
            else_body=build_flowgraph_between(
                context, prev_node.conditional_edge, curr_end, indent + 4)
        )
    # Otherwise, we have an && statement.
    else:
        return IfElseStatement(
            # We negate everything, because the conditional edges will jump
            # OVER the if body.
            join_conditions(conditions, '&&', only_negate_last=False),
            indent,
            if_body=build_flowgraph_between(
                context, curr_node, curr_end, indent + 4),
            else_body=build_flowgraph_between(
                context, start.conditional_edge, curr_end, indent + 4)
        )

def handle_return(
    context: Context, body: Body, return_node: Node, indent: int
):
    ret_info = return_node.block.block_info
    if ret_info and Register('v0') in ret_info.final_register_states:
        ret = ret_info.final_register_states[Register('v0')]
        body.add_comment(indent, f'// (possible return value: {ret})')
    else:
        body.add_comment(indent, '// (function likely void)')

def build_flowgraph_between(
    context: Context, start: Node, end: Node, indent: int
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
                body.add_statement(SimpleStatement(indent, 'return;'))
                handle_return(context, body, curr_start.successor, indent)
                break
            else:
                curr_start = curr_start.successor
        elif isinstance(curr_start, ConditionalNode):
            # A ConditionalNode means we need to find the next articulation
            # node. This means we need to find the "immediate postdominator"
            # of the current node, where "postdominator" means we have to go
            # through it, and "immediate" means we aren't skipping any.
            curr_end = immediate_postdominator(context, curr_start, end)
            # We also need to handle the if-else block here; this does the
            # outputting of the subgraph between curr_start and the next
            # articulation node.
            body.add_if_else(
                build_conditional_subgraph(context, curr_start, curr_end, indent))
            # Move on.
            curr_start = curr_end

    if isinstance(curr_start, ReturnNode):
        context.can_reach_return = True

    return body

def write_function(function_info: FunctionInfo) -> None:
    context = Context(flow_graph=function_info.flow_graph)
    start_node: Node = context.flow_graph.nodes[0]
    return_node: Node = context.flow_graph.nodes[-1]
    assert isinstance(return_node, ReturnNode)

    print("Here's the whole function!\n")
    body: Body = build_flowgraph_between(context, start_node, return_node, 4)
    body.add_node(return_node, 4)  # this node is not created in the above routine
    if context.can_reach_return:
        handle_return(context, body, return_node, 4)

    print(f'{function_info.stack_info.function.name}(...) {{')
    for local_var in function_info.stack_info.local_vars:
        print(f'    (???) {local_var};')
    print(body)
    print('}')
