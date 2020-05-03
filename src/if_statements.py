from typing import Dict, List, Optional, Set, Tuple, Union

import attr

from .error import DecompFailure
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
    case_nodes: Dict[Node, List[Tuple[int, int]]] = attr.ib(factory=dict)
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


@attr.s
class DoWhileLoop:
    indent: int = attr.ib()
    coding_style: CodingStyle = attr.ib()

    body: "Body" = attr.ib()
    condition: Optional[Condition] = attr.ib(default=None)

    def should_write(self) -> bool:
        return True

    def __str__(self) -> str:
        space = " " * self.indent
        brace_after_do = f"\n{space}{{" if self.coding_style.newline_after_if else " {"

        cond = stringify_expr(self.condition).rstrip(";") if self.condition else ""
        body = str(self.body)
        string_components = [
            f"{space}do{brace_after_do}\n{body}",
            f"{space}}} while ({cond});",
        ]
        return "\n".join(string_components)


Statement = Union[SimpleStatement, IfElseStatement, LabelStatement, DoWhileLoop]


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

    def add_do_while_loop(self, do_while_loop: DoWhileLoop) -> None:
        self.statements.append(do_while_loop)

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


def create_goto(context: Context, target: Node, indent: int) -> SimpleStatement:
    label = label_for_node(context, target)
    context.goto_nodes.add(target)
    return SimpleStatement(indent, f"goto {label};")


def emit_switch_jump(
    context: Context, expr: Expression, body: Body, indent: int
) -> None:
    body.add_statement(SimpleStatement(indent, f"goto *{stringify_expr(expr)};"))


def emit_goto_or_early_return(
    context: Context, target: Node, body: Body, indent: int
) -> None:
    """Emit a goto to a node, *unless* that node is an early return, which we
    can't goto to since it's not a real node and won't ever be emitted."""
    if isinstance(target, ReturnNode) and not target.is_real():
        add_return_statement(context, body, target, indent, last=False)
    else:
        emit_goto(context, target, body, indent)


def build_conditional_subgraph(
    context: Context, start: ConditionalNode, end: Node, indent: int
) -> IfElseStatement:
    """
    Output the subgraph between "start" and "end" at indent level "indent",
    given that "start" is a ConditionalNode; this program will intelligently
    output if/else relationships.
    """
    # print(f"build_conditional_subgraph({start.block.index}, {end.block.index})")
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
        if_body = build_flowgraph_between(
            context, start.fallthrough_edge, end, indent + 4
        )
    elif start.fallthrough_edge == end:
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
    else:
        # We need to see if this is a compound if-statement, i.e. containing
        # && or ||.
        conds = get_number_of_if_conditions(context, start, end)
        if conds < 2:  # normal if-statement
            # Both an if and an else block are present. We should write them in
            # chronological order (based on the original MIPS file). The
            # fallthrough edge will always be first, so write it that way.
            if_condition = if_block_info.branch_condition.negated()
            if_body = build_flowgraph_between(
                context, start.fallthrough_edge, end, indent + 4
            )
            else_body = build_flowgraph_between(
                context, start.conditional_edge, end, indent + 4
            )
        else:  # multiple conditions in if-statement
            return get_full_if_condition(context, conds, start, end, indent)

    return IfElseStatement(
        if_condition,
        indent,
        context.options.coding_style,
        if_body=if_body,
        else_body=else_body,
    )


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


def get_reachable_nodes(start: Node) -> Set[Node]:
    reachable_nodes: Set[Node] = set()
    stack: List[Node] = [start]
    while stack:
        node = stack.pop()
        if node in reachable_nodes:
            continue
        reachable_nodes.add(node)
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
    reachable_nodes = get_reachable_nodes(start)
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
    # Ideally, either all this node's parents are immediately postdominated by
    # it, or none of them are. In practice this doesn't always hold, and then
    # output of && and || may not be correct.
    if count not in [0, len(child.parents)] and not context.has_warned:
        context.has_warned = True
        print(
            "Warning: confusing control flow, output may have incorrect && "
            "and || detection. Run with --no-andor to disable detection and "
            "print gotos instead.\n"
        )
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
    if not context.options.andor_detection:
        # If &&/|| detection is disabled, short-circuit this logic and return
        # 1 instead.
        return 1

    count1 = count_non_postdominated_parents(context, node.conditional_edge, curr_end)
    count2 = count_non_postdominated_parents(context, node.fallthrough_edge, curr_end)

    # Return the nonzero count; the predicates will go through that path.
    # (TODO: I have a theory that we can just return count2 here.)
    if count1 != 0:
        return count1
    else:
        return count2


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


def get_full_if_condition(
    context: Context, count: int, start: ConditionalNode, curr_end: Node, indent: int
) -> IfElseStatement:
    curr_node: Node = start
    prev_node: Optional[ConditionalNode] = None
    conditions: List[Condition] = []
    # Get every condition.
    for i in range(count):
        if not isinstance(curr_node, ConditionalNode):
            raise DecompFailure(
                "Complex control flow; node assumed to be "
                "part of &&/|| wasn't. Run with --no-andor to disable "
                "detection of &&/|| and try again."
            )
        block_info = curr_node.block.block_info
        assert isinstance(block_info, BlockInfo)
        branch_condition = block_info.branch_condition
        assert branch_condition is not None

        # Make sure to write down each block's statement list,
        # even inside an and/or group.
        if i == 0:
            # The first condition in an if-statement will have
            # unrelated statements in its to_write list. Circumvent
            # emitting them twice by just using branch_condition:
            conditions.append(branch_condition)
        else:
            comma_statements = [
                statement
                for statement in block_info.to_write
                if statement.should_write()
            ]
            if comma_statements:
                assert not isinstance(branch_condition, CommaConditionExpr)
                comma_condition = CommaConditionExpr(comma_statements, branch_condition)
                conditions.append(comma_condition)
            else:
                conditions.append(branch_condition)
        prev_node = curr_node
        curr_node = curr_node.fallthrough_edge

    # At the end, if we end up at the conditional-edge after the very start,
    # then we know this was an || statement - if the start condition were true,
    # we would have skipped ahead to the body.
    if curr_node == start.conditional_edge:
        assert prev_node is not None
        return IfElseStatement(
            # Negate the last condition, for it must fall-through to the
            # body instead of jumping to it, hence it must jump OVER the body.
            join_conditions(conditions, "||", only_negate_last=True),
            indent,
            context.options.coding_style,
            if_body=build_flowgraph_between(
                context, start.conditional_edge, curr_end, indent + 4
            ),
            # The else-body is wherever the code jumps to instead of the
            # fallthrough (i.e. if-body).
            else_body=build_flowgraph_between(
                context, prev_node.conditional_edge, curr_end, indent + 4
            ),
        )
    # Otherwise, we have an && statement.
    else:
        return IfElseStatement(
            # We negate everything, because the conditional edges will jump
            # OVER the if body.
            join_conditions(conditions, "&&", only_negate_last=False),
            indent,
            context.options.coding_style,
            if_body=build_flowgraph_between(context, curr_node, curr_end, indent + 4),
            else_body=build_flowgraph_between(
                context, start.conditional_edge, curr_end, indent + 4
            ),
        )


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


def pattern_match_simple_do_while_loop(
    context: Context, start: ConditionalNode, indent: int
) -> Optional[DoWhileLoop]:
    if not start.is_self_loop():
        return None

    assert start.block.block_info
    assert start.block.block_info.branch_condition

    loop_body = Body(False, [])
    emit_node(context, start, loop_body, indent + 4)
    return DoWhileLoop(
        indent,
        context.options.coding_style,
        loop_body,
        start.block.block_info.branch_condition,
    )


def get_do_while_loop_between(
    context: Context, start: ConditionalNode, end: ConditionalNode, indent: int
) -> DoWhileLoop:
    assert end.block.block_info
    assert end.block.block_info.branch_condition

    # TODO: fallthrough_edge needs to be the right thing here
    # the real detection is conditional has a reverse arrow...
    # (a self-arrow IS a reverse arrow, so we can even consolidate with the
    # above function)
    loop_body = build_flowgraph_between(
        context, start.fallthrough_edge, end, indent + 4
    )
    emit_node(context, end, loop_body, indent + 4)

    return DoWhileLoop(
        indent,
        context.options.coding_style,
        loop_body,
        end.block.block_info.branch_condition,  # TODO: negated?
    )


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
    # print(f"build_flowgraph_between({start.block.index}, {end.block.index})")

    # We will split this graph into subgraphs, where the entrance and exit nodes
    # of that subgraph are at the same indentation level. "curr_start" will
    # iterate through these nodes, which are commonly referred to as
    # articulation nodes.
    while curr_start != end:
        # print(f"...curr_start={curr_start.block.index}")
        # Write the current node (but return nodes are handled specially).
        if not isinstance(curr_start, ReturnNode):
            # Before we do anything else, we pattern-match the subgraph
            # rooted at curr_start against certain predefined subgraphs
            # that emit do-while-loops:
            if isinstance(curr_start, ConditionalNode):
                # loops = curr_start.loop_edges()
                # for loop in loops:
                #     postdominated
                do_while_loop = pattern_match_simple_do_while_loop(
                    context, curr_start, indent
                )
                if do_while_loop:
                    body.add_do_while_loop(do_while_loop)
                    curr_start = curr_start.fallthrough_edge
                    continue

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
                assert block_info.switch_value is not None
                emit_switch_jump(context, block_info.switch_value, body, indent)
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
            # Once again, before anything else, we pattern match against "big"
            # do-while loops.
            # loops = curr_start.loop_edges()
            # loops = list(
            #     filter(
            #         lambda n: isinstance(n, ConditionalNode) and not n.is_self_loop(),
            #         loops,
            #     )
            # )
            # if loops:
            #     curr_end = sorted(loops, key=lambda n: n.block.index, reverse=True)[0]
            #     body.add_do_while_loop(
            #         get_do_while_loop_between(context, curr_start, curr_end, indent)
            #     )
            #     curr_start = curr_end.fallthrough_edge
            #     continue

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
            assert block_info.switch_value is not None
            emit_switch_jump(context, block_info.switch_value, body, 4)
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
            most_common = max(node.cases, key=node.cases.count)
            context.case_nodes[most_common] = [(switch_index, -1)]
            for index, target in enumerate(node.cases):
                if target == most_common:
                    continue
                if target not in context.case_nodes:
                    context.case_nodes[target] = []
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
