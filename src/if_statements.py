# from contextlib import contextmanager
import queue

import typing
from typing import List, Union, Iterator, Optional, Dict, Callable, Any, Set

from options import Options
from flow_graph import *
from translate import FunctionInfo, BlockInfo, BinaryOp, Type, simplify_condition, as_type

@attr.s
class Context:
    flow_graph: FlowGraph = attr.ib()
    options: Options = attr.ib()
    reachable_without: Dict[typing.Tuple[int, int, int], bool] = attr.ib(factory=dict)
    return_type: Type = attr.ib(factory=Type.any)

@attr.s
class IfElseStatement:
    condition: BinaryOp = attr.ib()
    indent: int = attr.ib()
    if_body: 'Body' = attr.ib()
    else_body: Optional['Body'] = attr.ib(default=None)

    def __str__(self) -> str:
        space = ' ' * self.indent
        # Avoid duplicate parentheses. TODO: make this cleaner and do it more
        # uniformly, not just here.
        condition = simplify_condition(self.condition)
        cond_str = str(condition)
        if not isinstance(condition, BinaryOp):
            cond_str = f'({cond_str})'
        if_str = '\n'.join([
            f'{space}if {cond_str}',
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

    def __str__(self) -> str:
        return f'{" " * self.indent}{self.contents}'

@attr.s
class Body:
    print_node_comment: bool = attr.ib()
    statements: List[Union[SimpleStatement, IfElseStatement]] = attr.ib(factory=list)

    def add_node(self, node: Node, indent: int, comment_empty: bool) -> None:
        assert isinstance(node.block.block_info, BlockInfo)
        to_write = node.block.block_info.to_write
        any_to_write = any(item.should_write() for item in to_write)

        # Add node header comment
        if self.print_node_comment and (any_to_write or comment_empty):
            self.add_comment(indent, f'Node {node.block.index}')
        # Add node contents
        for item in node.block.block_info.to_write:
            if item.should_write():
                self.statements.append(SimpleStatement(indent, str(item)))

    def add_statement(self, statement: SimpleStatement) -> None:
        self.statements.append(statement)

    def add_comment(self, indent: int, contents: str) -> None:
        self.add_statement(SimpleStatement(indent, f'// {contents}'))

    def add_if_else(self, if_else: IfElseStatement) -> None:
        self.statements.append(if_else)

    def __str__(self) -> str:
        return '\n'.join(str(statement) for statement in self.statements)


def build_conditional_subgraph(
    context: Context, start: ConditionalNode, end: Node, indent: int
) -> IfElseStatement:
    """
    Output the subgraph between "start" and "end" at indent level "indent",
    given that "start" is a ConditionalNode; this program will intelligently
    output if/else relationships.
    """
    if_block_info = start.block.block_info
    assert isinstance(if_block_info, BlockInfo)
    assert if_block_info.branch_condition is not None

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
        if_condition = if_block_info.branch_condition
        if not start.is_loop():
            # Only an if block, so this is easy.
            # I think this can only happen in the case where the other branch has
            # an early return.
            if_body = build_flowgraph_between(context, start.conditional_edge, end, indent + 4)
        else:
            # Don't want to follow the loop, otherwise we'd be trapped here.
            # Instead, write a goto for the beginning of the loop.
            label = f'loop_{start.conditional_edge.block.index}'
            if_body = Body(False, [SimpleStatement(indent + 4, f'goto {label};')])
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

def end_reachable_without(
    context: Context, start: Node, end: Node, without: Node
) -> bool:
    """Return whether "end" is reachable from "start" if "without" were removed.
    """
    if end == without or start == without:
        # Can't get to the end.
        return False
    if start == end:
        # Already there! (Base case.)
        return True

    key = (start.block.index, end.block.index, without.block.index)
    if key in context.reachable_without:
        return context.reachable_without[key]

    def reach(edge: Node) -> bool:
        return end_reachable_without(context, edge, end, without)

    if isinstance(start, BasicNode):
        ret = reach(start.successor)
    elif isinstance(start, ConditionalNode):
        # Going through the conditional node cannot help, since that is a
        # backwards arrow. There is no way to get to the end.
        ret = (reach(start.fallthrough_edge) or
            (not start.is_loop() and reach(start.conditional_edge)))
    else:
        assert isinstance(start, ReturnNode)
        ret = False

    context.reachable_without[key] = ret
    return ret

def immediate_postdominator(context: Context, start: Node, end: Node) -> Node:
    """
    Find the immediate postdominator of "start", where "end" is an exit node
    from the control flow graph.
    """
    stack: List[Node] = [start]
    postdominators: List[Node] = []
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
            if not node.is_loop():
                # If the node is a loop, then adding the conditional edge
                # here would cause this while loop to never end.
                stack.append(node.conditional_edge)
            stack.append(node.fallthrough_edge)
        # If removing the node means the end becomes unreachable,
        # the node is a postdominator.
        if node != start and not end_reachable_without(context, start, end, node):
            postdominators.append(node)
    assert postdominators  # at least "end" should be a postdominator
    # Get the earliest postdominator
    postdominators.sort(key=lambda node: node.block.index)
    return postdominators[0]


def count_non_postdominated_parents(
    context: Context, child: Node, curr_end: Node
) -> int:
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


def get_number_of_if_conditions(
    context: Context, node: ConditionalNode, curr_end: Node
) -> int:
    """
    For a given ConditionalNode, this function will return k when the if-
    statement of the correspondant C code is "if (1 && 2 && ... && k)" or
    "if (1 || 2 || ... || k)", where the numbers are labels for clauses.
    (It remains unclear how a predicate that mixes && and || would behave.)
    """
    count1 = count_non_postdominated_parents(context, node.conditional_edge,
                                             curr_end)
    count2 = count_non_postdominated_parents(context, node.fallthrough_edge,
                                             curr_end)

    # Return the nonzero count; the predicates will go through that path.
    # (TODO: I have a theory that we can just return count2 here.)
    if count1 != 0:
        return count1
    else:
        return count2

def join_conditions(
    conditions: List[BinaryOp], op: str, only_negate_last: bool
) -> BinaryOp:
    assert op in ['&&', '||']
    assert conditions
    final_cond: Optional[BinaryOp] = None
    for i, cond in enumerate(conditions):
        if not only_negate_last or i == len(conditions) - 1:
            cond = cond.negated()
        if final_cond is None:
            final_cond = cond
        else:
            final_cond = BinaryOp(final_cond, op, cond, type=Type.bool())
    assert final_cond is not None
    return final_cond

def get_full_if_condition(
    context: Context,
    count: int,
    start: ConditionalNode,
    curr_end: Node,
    indent: int
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

def write_return(
    context: Context, body: Body, node: ReturnNode, indent: int, last: bool
) -> None:
    body.add_node(node, indent, comment_empty=node.real)

    ret_info = node.block.block_info
    assert isinstance(ret_info, BlockInfo)

    ret = ret_info.return_value
    if ret is not None:
        ret = as_type(ret, context.return_type, True)
        body.add_statement(SimpleStatement(indent, f'return {ret};'))
    elif not last:
        body.add_statement(SimpleStatement(indent, 'return;'))


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
    body = Body(print_node_comment=context.options.debug)

    # We will split this graph into subgraphs, where the entrance and exit nodes
    # of that subgraph are at the same indentation level. "curr_start" will
    # iterate through these nodes, which are commonly referred to as
    # articulation nodes.
    while curr_start != end:
        # Write the current node (but return nodes are handled specially).
        if not isinstance(curr_start, ReturnNode):
            # We currently write loops as labels and gotos. If a node is
            # "looped to", in the sense that some parent makes a backwards jump
            # to it, then it needs a label.
            if any(node.block.index >= curr_start.block.index
                    for node in curr_start.parents):
                label = f'loop_{curr_start.block.index}'
                body.add_statement(SimpleStatement(0, f'{label}:'))
            body.add_node(curr_start, indent, comment_empty=True)

        if isinstance(curr_start, BasicNode):
            # In a BasicNode, the successor is the next articulation node.
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
        else:
            assert isinstance(curr_start, ReturnNode)
            # Write the return node, and break, because there is nothing more
            # to process.
            write_return(context, body, curr_start, indent, last=False)
            break

    return body

def write_function(function_info: FunctionInfo, options: Options) -> None:
    context = Context(flow_graph=function_info.flow_graph, options=options)
    start_node: Node = context.flow_graph.nodes[0]
    return_node: Node = context.flow_graph.nodes[-1]
    assert isinstance(return_node, ReturnNode)

    if options.debug:
        print("Here's the whole function!\n")
    body: Body = build_flowgraph_between(context, start_node, return_node, 4)

    write_return(context, body, return_node, 4, last=True)

    ret_type = 'void '
    if not context.return_type.is_any():
        ret_type = context.return_type.to_decl()
    fn_name = function_info.stack_info.function.name
    arg_strs = []
    for arg in function_info.stack_info.arguments:
        arg_strs.append(f'{arg.type.to_decl()}{arg}')
    arg_str = ', '.join(arg_strs) or 'void'
    print(f'{ret_type}{fn_name}({arg_str})\n{{')

    any_decl = False
    for local_var in function_info.stack_info.local_vars[::-1]:
        type_decl = local_var.type.to_decl()
        print(SimpleStatement(4, f'{type_decl}{local_var};'))
        any_decl = True
    for temp_var in function_info.stack_info.temp_vars:
        if temp_var.need_decl():
            expr = temp_var.expr
            type_decl = expr.type.to_decl()
            print(SimpleStatement(4, f'{type_decl}{expr.get_var_name()};'))
            any_decl = True
    for phi_var in function_info.stack_info.phi_vars:
        type_decl = phi_var.type.to_decl()
        print(SimpleStatement(4, f'{type_decl}{phi_var.get_var_name()};'))
        any_decl = True
    if any_decl:
        print()

    print(body)
    print('}')
