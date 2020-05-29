from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import attr

from .flow_graph import (
    BasicNode,
    Block,
    ConditionalNode,
    FlowGraph,
    Node,
    ReturnNode,
    SwitchNode,
)
from .options import CodingStyle, Options
from .translate import (
    BinaryOp,
    BlockInfo,
    CommaConditionExpr,
    Condition,
    Expression,
    FunctionInfo,
    Type,
    simplify_condition,
    stringify_expr,
)


@attr.s
class Context:
    flow_graph: FlowGraph = attr.ib()
    options: Options = attr.ib()
    reachable_without: Dict[Tuple[Node, Node], Set[Node]] = attr.ib(factory=dict)
    is_void: bool = attr.ib(default=True)
    switch_nodes: Dict[SwitchNode, int] = attr.ib(factory=dict)
    case_nodes: Dict[Node, List[Tuple[int, int]]] = attr.ib(
        factory=lambda: defaultdict(list)
    )
    goto_nodes: Set[Node] = attr.ib(factory=set)
    loop_nodes: Set[Node] = attr.ib(factory=set)
    emitted_nodes: Set[Node] = attr.ib(factory=set)
    has_warned: bool = attr.ib(default=False)


@attr.s
class IfElseStatement:
    condition: Condition = attr.ib()
    indent: int = attr.ib()
    coding_style: CodingStyle = attr.ib()
    if_body: "Body" = attr.ib()
    else_body: Optional["Body"] = attr.ib(default=None)

    def should_write(self) -> bool:
        return True

    def __str__(self) -> str:
        space = " " * self.indent
        condition = simplify_condition(self.condition)
        cond_str = stringify_expr(condition)
        brace_after_if = f"\n{space}{{" if self.coding_style.newline_after_if else " {"
        if_str = "\n".join(
            [
                f"{space}if ({cond_str}){brace_after_if}",
                str(self.if_body),  # has its own indentation
                f"{space}}}",
            ]
        )
        if self.else_body is not None:
            whitespace = f"\n{space}" if self.coding_style.newline_before_else else " "
            else_str = "\n".join(
                [f"{whitespace}else{brace_after_if}", str(self.else_body), f"{space}}}"]
            )
            if_str = if_str + else_str
        return if_str


@attr.s
class SimpleStatement:
    indent: int = attr.ib()
    contents: str = attr.ib()

    def should_write(self) -> bool:
        return True

    def __str__(self) -> str:
        return f'{" " * self.indent}{self.contents}'


@attr.s
class LabelStatement:
    indent: int = attr.ib()
    context: Context = attr.ib()
    node: Node = attr.ib()

    def should_write(self) -> bool:
        return (
            self.node in self.context.goto_nodes or self.node in self.context.case_nodes
        )

    def __str__(self) -> str:
        lines = []
        if self.node in self.context.case_nodes:
            for (switch, case) in self.context.case_nodes[self.node]:
                case_str = f"case {case}" if case != -1 else "default"
                switch_str = f" // switch {switch}" if switch != 0 else ""
                lines.append(f'{" " * self.indent}{case_str}:{switch_str}')
        if self.node in self.context.goto_nodes:
            lines.append(f"{label_for_node(self.context, self.node)}:")
        return "\n".join(lines)


Statement = Union[SimpleStatement, IfElseStatement, LabelStatement]


@attr.s
class Body:
    print_node_comment: bool = attr.ib()
    statements: List[Statement] = attr.ib(factory=list)

    def add_node(self, node: Node, indent: int, comment_empty: bool) -> None:
        assert isinstance(node.block.block_info, BlockInfo)
        to_write = node.block.block_info.to_write
        any_to_write = any(item.should_write() for item in to_write)

        # Add node header comment
        if self.print_node_comment and (any_to_write or comment_empty):
            self.add_comment(indent, f"Node {node.name()}")
        # Add node contents
        for item in node.block.block_info.to_write:
            if item.should_write():
                self.statements.append(SimpleStatement(indent, str(item)))

    def add_statement(self, statement: Statement) -> None:
        self.statements.append(statement)

    def add_comment(self, indent: int, contents: str) -> None:
        self.add_statement(SimpleStatement(indent, f"// {contents}"))

    def add_if_else(self, if_else: IfElseStatement) -> None:
        self.statements.append(if_else)

    def is_empty(self) -> bool:
        return not any(statement.should_write() for statement in self.statements)

    def __str__(self) -> str:
        return "\n".join(
            str(statement) for statement in self.statements if statement.should_write()
        )


def label_for_node(context: Context, node: Node) -> str:
    if node in context.loop_nodes:
        return f"loop_{node.block.index}"
    else:
        return f"block_{node.block.index}"


def emit_node(context: Context, node: Node, body: Body, indent: int) -> None:
    """Emit a node, together with a label for it (which is only printed if
    something jumps to it, e.g. currently for loops)."""
    if isinstance(node, ReturnNode) and not node.is_real():
        body.add_node(node, indent, comment_empty=False)
    else:
        body.add_statement(LabelStatement(max(indent - 4, 0), context, node))
        body.add_node(node, indent, comment_empty=True)


def emit_goto(context: Context, target: Node, body: Body, indent: int) -> None:
    label = label_for_node(context, target)
    context.goto_nodes.add(target)
    body.add_statement(SimpleStatement(indent, f"goto {label};"))


def emit_switch_jump(
    context: Context, node: SwitchNode, body: Body, indent: int
) -> None:
    block_info = node.block.block_info
    assert isinstance(block_info, BlockInfo)
    expr = block_info.switch_value
    assert expr is not None
    switch_index = context.switch_nodes.get(node, 0)
    comment = f" // switch {switch_index}" if switch_index else ""
    body.add_statement(
        SimpleStatement(indent, f"goto *{stringify_expr(expr)};{comment}")
    )


def emit_goto_or_early_return(
    context: Context, target: Node, body: Body, indent: int
) -> None:
    """Emit a goto to a node, *unless* that node is an early return, which we
    can't goto to since it's not a real node and won't ever be emitted."""
    if isinstance(target, ReturnNode) and not target.is_real():
        add_return_statement(context, body, target, indent, last=False)
    else:
        emit_goto(context, target, body, indent)


def end_reachable_without(
    context: Context, start: Node, end: Node, without: Node
) -> bool:
    """Return whether "end" is reachable from "start" if "without" were removed.
    """
    key = (start, without)
    if key in context.reachable_without:
        return end in context.reachable_without[key]

    reachable: Set[Node] = set()
    stack: List[Node] = [start]

    while stack:
        node = stack.pop()
        if node == without or node in reachable:
            continue
        reachable.add(node)
        if isinstance(node, BasicNode):
            stack.append(node.successor)
        elif isinstance(node, ConditionalNode):
            stack.append(node.fallthrough_edge)
            if not node.is_loop():
                # For compatibility with older code, don't add back edges.
                # (It would cause infinite loops before this was rewritten
                # iteratively, with a 'node in reachable' check avoiding
                # loops.) TODO: revisit this?
                stack.append(node.conditional_edge)
        elif isinstance(node, SwitchNode):
            stack.extend(node.cases)
        else:
            _: ReturnNode = node

    context.reachable_without[key] = reachable
    return end in reachable


def get_reachable_nodes(start: Node) -> List[Node]:
    reachable_nodes_set: Set[Node] = set()
    reachable_nodes: List[Node] = []
    stack: List[Node] = [start]
    while stack:
        node = stack.pop()
        if node in reachable_nodes_set:
            continue
        reachable_nodes_set.add(node)
        reachable_nodes.append(node)
        if isinstance(node, BasicNode):
            stack.append(node.successor)
        elif isinstance(node, ConditionalNode):
            if not node.is_loop():
                stack.append(node.conditional_edge)
            stack.append(node.fallthrough_edge)
        elif isinstance(node, SwitchNode):
            stack.extend(node.cases)
        else:
            _: ReturnNode = node
    return reachable_nodes


def immediate_postdominator(context: Context, start: Node, end: Node) -> Node:
    """
    Find the immediate postdominator of "start", where "end" is an exit node
    from the control flow graph.
    """
    # If the end is unreachable, we are computing immediate postdominators
    # of a subflow where every path ends in an early return. In this case we
    # need to replace our end node, or else every node will be treated as a
    # postdominator, and the earliest one might be within a conditional
    # expression. That in turn can result in nodes emitted multiple times.
    # (TODO: this is rather ad hoc, we should probably come up with a more
    # principled approach to early returns...)
    #
    # Note that we use a List instead of a Set here, since duplicated return
    # nodes may result in multiple nodes with the same block index, and sets
    # have non-deterministic iteration order.
    reachable_nodes: List[Node] = get_reachable_nodes(start)
    if end not in reachable_nodes:
        end = max(reachable_nodes, key=lambda n: n.block.index)

    stack: List[Node] = [start]
    seen: Set[Node] = set()
    postdominators: List[Node] = []
    while stack:
        # Get potential postdominator.
        node = stack.pop()
        if node in seen:
            # Don't revisit nodes.
            continue
        seen.add(node)
        # If removing the node means the end becomes unreachable,
        # the node is a postdominator.
        if node != start and not end_reachable_without(context, start, end, node):
            postdominators.append(node)
        else:
            # Otherwise, add the children of the node and continue the search.
            assert node != end
            if isinstance(node, BasicNode):
                stack.append(node.successor)
            elif isinstance(node, ConditionalNode):
                if not node.is_loop():
                    # This check is wonky, see end_reachable_without.
                    # It should be kept the same as in get_reachable_nodes.
                    stack.append(node.conditional_edge)
                stack.append(node.fallthrough_edge)
            elif isinstance(node, SwitchNode):
                stack.extend(node.cases)
            else:
                _: ReturnNode = node
    assert len(postdominators) == 1, "we should always find exactly one postdominator"
    return postdominators[0]


def is_and_statement(next_node: Node, bottom: Node) -> bool:
    return (
        not isinstance(next_node, ConditionalNode)
        or not (
            next_node.conditional_edge is bottom or next_node.fallthrough_edge is bottom
        )
        # A strange edge-case of our pattern-matching technology:
        # self-loops match the pattern. Avoiding that...
        or next_node.is_loop()
    )


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
    else_body: Optional[Body] = None
    if start.fallthrough_edge == end:
        if_condition = if_block_info.branch_condition
        if not start.is_loop():
            # Only an if block, so this is easy.
            # I think this can only happen in the case where the other branch has
            # an early return.
            if_body = build_flowgraph_between(
                context, start.conditional_edge, end, indent + 4
            )
        else:
            # Don't want to follow the loop, otherwise we'd be trapped here.
            # Instead, write a goto for the beginning of the loop.
            if_body = Body(False, [])
            emit_goto(context, start.conditional_edge, if_body, indent + 4)
    elif is_and_statement(start.fallthrough_edge, start.conditional_edge):
        # If the conditional edge isn't real, then the "fallthrough_edge" is
        # actually within the inner if-statement. This means we have to negate
        # the fallthrough edge and go down that path.
        assert start.block.block_info
        assert start.block.block_info.branch_condition
        if_condition = start.block.block_info.branch_condition.negated()

        if_body = build_flowgraph_between(
            context, start.fallthrough_edge, end, indent + 4
        )
        else_body = build_flowgraph_between(
            context, start.conditional_edge, end, indent + 4
        )
        if else_body.is_empty():
            else_body = None
    else:
        return get_andor_if_statement(context, start, end, indent)

    return IfElseStatement(
        if_condition, indent, context.options.coding_style, if_body, else_body
    )


def gather_any_comma_conditions(block_info: BlockInfo) -> Condition:
    branch_condition = block_info.branch_condition
    assert branch_condition is not None
    comma_statements = [
        statement for statement in block_info.to_write if statement.should_write()
    ]
    if comma_statements:
        assert not isinstance(branch_condition, CommaConditionExpr)
        return CommaConditionExpr(comma_statements, branch_condition)
    else:
        return branch_condition


def get_andor_if_statement(
    context: Context, start: ConditionalNode, end: Node, indent: int
) -> IfElseStatement:
    """
    This function detects if-conditions of any form.

    This is an example of an or-statement:

               +-------+      COND
               | start |---------------------+
               +-------+                     |
                   |                         |
                   | FALL                    |
                   v                         |
               +-------+      COND           |
               |   1   |-------------------+ |
               +-------+                   | |
                   |                       | |
                   | FALL                  | |
                   v                       | |
               +-------+      COND         | |
               |   2   |-----------------+ | |
               +-------+                 | | |
                   |                     | | |
                   | FALL                | | |
                   v                     | | |
    ________________________________________________
                                         . . .
                                         . . .
                                         . . .
    ________________________________________________
                   |                     | | |
                   | FALL                | | |
                   v                     v v v
               +-------+           +--------------+
               |   N   |---------->|    bottom    |
               +-------+           +--------------+
                   |                   |
                   | FALL              |
                   v                   |
     +-----------------+               |
     |                 |               |
     |  SOME SUBGRAPH  |               |
     |                 |               |
     +-----------------+               |
                   |                   |
                   |                   |
                   v                   v
               +---------------------------+
               |            end            |
               +---------------------------+

    """
    conditions: List[Condition] = []
    bottom = start.conditional_edge
    curr_node: ConditionalNode = start
    while True:
        # Collect conditions as we accumulate them:
        block_info = curr_node.block.block_info
        assert isinstance(block_info, BlockInfo)
        branch_condition = block_info.branch_condition
        assert branch_condition is not None
        if not conditions:
            # The first condition in an if-statement will have
            # unrelated statements in its to_write list. Circumvent
            # emitting them twice by just using branch_condition:
            conditions.append(branch_condition)
        else:
            # Make sure to write down each block's statement list,
            # even inside an and/or group.
            conditions.append(gather_any_comma_conditions(block_info))

        # The next node will tell us whether we are in an &&/|| statement...
        next_node = curr_node.fallthrough_edge

        else_body: Optional[Body]
        if is_and_statement(next_node, bottom):
            # We reached the end of an and-statement.
            # TODO: The last one - or more - might've been part
            # of a while loop.

            else_body = build_flowgraph_between(context, bottom, end, indent + 4)
            if else_body.is_empty():
                else_body = None
            return IfElseStatement(
                # We negate everything, because the conditional edges will jump
                # OVER the if body.
                join_conditions(conditions, "&&", only_negate_last=False),
                indent,
                context.options.coding_style,
                if_body=build_flowgraph_between(context, next_node, end, indent + 4),
                else_body=else_body,
            )
        assert isinstance(next_node, ConditionalNode)  # from is_and_statement()

        if next_node.fallthrough_edge is bottom:
            assert next_node.block.block_info
            next_node_condition = next_node.block.block_info.branch_condition
            assert next_node_condition
            # We were following an or-statement.
            if_body = build_flowgraph_between(context, bottom, end, indent + 4)
            else_body = build_flowgraph_between(
                context, next_node.conditional_edge, end, indent + 4
            )
            if else_body.is_empty():
                else_body = None
            return IfElseStatement(
                # Negate the last condition, for it must fall-through to the
                # body instead of jumping to it, hence it must jump OVER the body.
                join_conditions(
                    conditions + [next_node_condition], "||", only_negate_last=True
                ),
                indent,
                context.options.coding_style,
                if_body=if_body,
                # The else-body is wherever the code jumps to instead of the
                # fallthrough (i.e. if-body).
                else_body=else_body,
            )

        # Otherwise, still in the middle of an and-or statement.
        assert next_node.conditional_edge is bottom
        curr_node = next_node
    assert False


def join_conditions(
    conditions: List[Condition], op: str, only_negate_last: bool
) -> Condition:
    assert op in ["&&", "||"]
    assert conditions
    final_cond: Optional[Condition] = None
    for i, cond in enumerate(conditions):
        if not only_negate_last or i == len(conditions) - 1:
            cond = cond.negated()
        if final_cond is None:
            final_cond = cond
        else:
            final_cond = BinaryOp(final_cond, op, cond, type=Type.bool())
    assert final_cond is not None
    return final_cond


def add_return_statement(
    context: Context, body: Body, node: ReturnNode, indent: int, last: bool
) -> None:
    emit_node(context, node, body, indent)

    ret_info = node.block.block_info
    assert isinstance(ret_info, BlockInfo)

    ret = ret_info.return_value
    if ret is not None:
        ret_str = stringify_expr(ret)
        body.add_statement(SimpleStatement(indent, f"return {ret_str};"))
        context.is_void = False
    elif not last:
        body.add_statement(SimpleStatement(indent, "return;"))


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
            # If a node is ever encountered twice, we can emit a goto to the
            # first place we emitted it. Since nodes represent positions in the
            # assembly, and we use phi's for preserved variable contents, this
            # will end up semantically equivalent. This can happen sometimes
            # when early returns/continues/|| are not detected correctly, and
            # hints at that situation better than if we just blindly duplicate
            # the block.
            if curr_start in context.emitted_nodes:
                emit_goto(context, curr_start, body, indent)
                break
            context.emitted_nodes.add(curr_start)

            # Emit the node, and possibly a label for it.
            emit_node(context, curr_start, body, indent)

        if curr_start.emit_goto:
            # If we have decided to emit a goto here, then we should just fall
            # through to the next node, after writing a goto/conditional goto/
            # return.
            block_info = curr_start.block.block_info
            assert isinstance(block_info, BlockInfo)
            if isinstance(curr_start, BasicNode):
                emit_goto_or_early_return(context, curr_start.successor, body, indent)
            elif isinstance(curr_start, ConditionalNode):
                target = curr_start.conditional_edge
                if_body = Body(print_node_comment=False)
                emit_goto_or_early_return(context, target, if_body, indent + 4)
                assert block_info.branch_condition is not None
                body.add_if_else(
                    IfElseStatement(
                        block_info.branch_condition,
                        indent,
                        context.options.coding_style,
                        if_body=if_body,
                        else_body=None,
                    )
                )
            elif isinstance(curr_start, SwitchNode):
                emit_switch_jump(context, curr_start, body, indent)
            else:  # ReturnNode
                add_return_statement(context, body, curr_start, indent, last=False)

            # Advance to the next node in block order. This may skip over
            # unreachable blocks -- hopefully none too important.
            index = context.flow_graph.nodes.index(curr_start)
            fallthrough = context.flow_graph.nodes[index + 1]
            if isinstance(curr_start, ConditionalNode):
                assert fallthrough == curr_start.fallthrough_edge
            curr_start = fallthrough
            continue

        # Switch nodes are always marked emit_goto.
        assert not isinstance(curr_start, SwitchNode)

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
                build_conditional_subgraph(context, curr_start, curr_end, indent)
            )
            # Move on.
            curr_start = curr_end
        else:  # ReturnNode
            # Write the return node, and break, because there is nothing more
            # to process.
            add_return_statement(context, body, curr_start, indent, last=False)
            break

    return body


def build_naive(context: Context, nodes: List[Node]) -> Body:
    """Naive procedure for generating output with only gotos for control flow.

    Used for --no-ifs, when the regular if_statements code fails."""

    body = Body(print_node_comment=context.options.debug)

    def emit_successor(node: Node, cur_index: int) -> None:
        if (
            cur_index + 1 < len(nodes)
            and nodes[cur_index + 1] == node
            and not (isinstance(node, ReturnNode) and not node.is_real())
        ):
            # Fallthrough is fine
            return
        emit_goto_or_early_return(context, node, body, 4)

    for i, node in enumerate(nodes):
        block_info = node.block.block_info
        assert isinstance(block_info, BlockInfo)
        if isinstance(node, ReturnNode):
            # Do not emit return nodes; they are often duplicated and don't
            # have a well-defined position, so we emit them next to where they
            # are jumped to instead.
            pass
        elif isinstance(node, BasicNode):
            emit_node(context, node, body, 4)
            emit_successor(node.successor, i)
        elif isinstance(node, SwitchNode):
            emit_node(context, node, body, 4)
            emit_switch_jump(context, node, body, 4)
        else:  # ConditionalNode
            emit_node(context, node, body, 4)
            if_body = Body(print_node_comment=False)
            emit_goto_or_early_return(context, node.conditional_edge, if_body, 8)
            assert block_info.branch_condition is not None
            body.add_if_else(
                IfElseStatement(
                    block_info.branch_condition,
                    4,
                    context.options.coding_style,
                    if_body=if_body,
                    else_body=None,
                )
            )
            emit_successor(node.fallthrough_edge, i)

    return body


def build_body(
    context: Context, function_info: FunctionInfo, options: Options,
) -> Body:
    start_node: Node = context.flow_graph.entry_node()
    return_node: Optional[ReturnNode] = context.flow_graph.return_node()
    if return_node is None:
        fictive_block = Block(-1, None, "", [])
        return_node = ReturnNode(fictive_block, False, index=-1)

    num_switches = len(
        [node for node in context.flow_graph.nodes if isinstance(node, SwitchNode)]
    )
    switch_index = 0
    for node in context.flow_graph.nodes:
        if isinstance(node, SwitchNode):
            assert node.cases, "jtbl list must not be empty"
            if num_switches > 1:
                switch_index += 1
            context.switch_nodes[node] = switch_index
            most_common = max(node.cases, key=node.cases.count)
            has_common = False
            if node.cases.count(most_common) > 1:
                context.case_nodes[most_common].append((switch_index, -1))
                has_common = True
            for index, target in enumerate(node.cases):
                if has_common and target == most_common:
                    continue
                context.case_nodes[target].append((switch_index, index))
        elif isinstance(node, ConditionalNode) and node.is_loop():
            context.loop_nodes.add(node.conditional_edge)
        elif isinstance(node, BasicNode) and node.is_loop():
            context.loop_nodes.add(node.successor)

    if options.debug:
        print("Here's the whole function!\n")
    body: Body
    if options.ifs:
        body = build_flowgraph_between(context, start_node, return_node, 4)
    else:
        body = build_naive(context, context.flow_graph.nodes)

    if return_node.index != -1:
        add_return_statement(context, body, return_node, 4, last=True)

    return body


def get_function_text(function_info: FunctionInfo, options: Options) -> str:
    context = Context(flow_graph=function_info.flow_graph, options=options)
    body: Body = build_body(context, function_info, options)

    function_lines: List[str] = []

    fn_name = function_info.stack_info.function.name
    arg_strs = []
    for arg in function_info.stack_info.arguments:
        arg_strs.append(arg.type.to_decl(str(arg)))
    if function_info.stack_info.is_variadic:
        arg_strs.append("...")
    arg_str = ", ".join(arg_strs) or "void"

    fn_header = f"{fn_name}({arg_str})"

    if context.is_void:
        fn_header = f"void {fn_header}"
    else:
        fn_header = function_info.return_type.to_decl(fn_header)
    whitespace = "\n" if options.coding_style.newline_after_function else " "
    function_lines.append(f"{fn_header}{whitespace}{{")

    any_decl = False
    for local_var in function_info.stack_info.local_vars[::-1]:
        type_decl = local_var.type.to_decl(str(local_var))
        function_lines.append(str(SimpleStatement(4, f"{type_decl};")))
        any_decl = True
    temp_decls = set()
    for temp_var in function_info.stack_info.temp_vars:
        if temp_var.need_decl():
            expr = temp_var.expr
            type_decl = expr.type.to_decl(str(expr.var))
            temp_decls.add(f"{type_decl};")
            any_decl = True
    for decl in sorted(list(temp_decls)):
        function_lines.append(str(SimpleStatement(4, decl)))
    for phi_var in function_info.stack_info.phi_vars:
        type_decl = phi_var.type.to_decl(phi_var.get_var_name())
        function_lines.append(str(SimpleStatement(4, f"{type_decl};")))
        any_decl = True
    if any_decl:
        function_lines.append("")

    function_lines.append(str(body))
    function_lines.append("}")
    full_function_text: str = "\n".join(function_lines)
    return full_function_text
