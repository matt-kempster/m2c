from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import attr

from .flow_graph import (
    BasicNode,
    ConditionalNode,
    FlowGraph,
    Node,
    ReturnNode,
    SwitchNode,
    TerminalNode,
)
from .options import Options
from .translate import (
    BinaryOp,
    BlockInfo,
    CommaConditionExpr,
    Condition,
    Formatter,
    FunctionInfo,
    Statement as TrStatement,
    Type,
    simplify_condition,
    format_expr,
)


@attr.s
class Context:
    flow_graph: FlowGraph = attr.ib()
    fmt: Formatter = attr.ib()
    options: Options = attr.ib()
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
    if_body: "Body" = attr.ib()
    else_body: Optional["Body"] = attr.ib(default=None)

    def should_write(self) -> bool:
        return True

    def format(self, fmt: Formatter) -> str:
        space = fmt.indent(0, "")
        condition = simplify_condition(self.condition)
        cond_str = format_expr(condition, fmt)
        after_ifelse = f"\n{space}" if fmt.coding_style.newline_after_if else " "
        before_else = f"\n{space}" if fmt.coding_style.newline_before_else else " "
        with fmt.indented():
            if_str = "\n".join(
                [
                    f"{space}if ({cond_str}){after_ifelse}{{",
                    self.if_body.format(fmt),  # has its own indentation
                    f"{space}}}",
                ]
            )
        if self.else_body is not None and not self.else_body.is_empty():
            sub_if = self.else_body.get_lone_if_statement()
            if sub_if:
                sub_if_str = sub_if.format(fmt).lstrip()
                else_str = f"{before_else}else {sub_if_str}"
            else:
                with fmt.indented():
                    else_str = "\n".join(
                        [
                            f"{before_else}else{after_ifelse}{{",
                            self.else_body.format(fmt),
                            f"{space}}}",
                        ]
                    )
            if_str = if_str + else_str
        return if_str

    def negated(self) -> "IfElseStatement":
        """Return an IfElseStatement with the condition negated and the if/else
        bodies swapped. The caller must check that `else_body` is present."""
        assert self.else_body is not None, "checked by caller"
        return attr.evolve(
            self,
            condition=self.condition.negated(),
            if_body=self.else_body,
            else_body=self.if_body,
        )


@attr.s
class SwitchStatement:
    jump: "SimpleStatement" = attr.ib()
    body: "Body" = attr.ib()

    def should_write(self) -> bool:
        return True

    def format(self, fmt: Formatter) -> str:
        lines = [
            self.jump.format(fmt),
            self.body.format(fmt),
        ]
        return "\n".join(lines)


@attr.s
class SimpleStatement:
    contents: Union[str, TrStatement] = attr.ib()
    is_jump: bool = attr.ib(default=False)

    def should_write(self) -> bool:
        return bool(self.contents)

    def format(self, fmt: Formatter) -> str:
        if isinstance(self.contents, str):
            return fmt.indent(0, self.contents)
        else:
            return fmt.indent(0, self.contents.format(fmt))


@attr.s
class LabelStatement:
    context: Context = attr.ib()
    node: Node = attr.ib()

    def should_write(self) -> bool:
        return (
            self.node in self.context.goto_nodes or self.node in self.context.case_nodes
        )

    def format(self, fmt: Formatter) -> str:
        lines = []
        if self.node in self.context.case_nodes:
            for (switch, case) in self.context.case_nodes[self.node]:
                case_str = f"case {case}" if case != -1 else "default"
                switch_str = f" // switch {switch}" if switch != 0 else ""
                lines.append(fmt.indent(-1, f"{case_str}:{switch_str}"))
        if self.node in self.context.goto_nodes:
            lines.append(f"{label_for_node(self.context, self.node)}:")
        return "\n".join(lines)


Statement = Union[
    SimpleStatement,
    IfElseStatement,
    LabelStatement,
    SwitchStatement,
]


@attr.s
class Body:
    print_node_comment: bool = attr.ib()
    statements: List[Statement] = attr.ib(factory=list)

    def extend(self, other: "Body") -> None:
        """Add the contents of `other` into ourselves"""
        self.print_node_comment |= other.print_node_comment
        self.statements.extend(other.statements)

    def add_node(self, node: Node, comment_empty: bool) -> None:
        assert isinstance(node.block.block_info, BlockInfo)
        to_write = node.block.block_info.to_write
        any_to_write = any(item.should_write() for item in to_write)

        # Add node header comment
        if self.print_node_comment and (any_to_write or comment_empty):
            self.add_comment(f"Node {node.name()}")
        # Add node contents
        for item in node.block.block_info.to_write:
            if item.should_write():
                self.statements.append(SimpleStatement(item))

    def add_statement(self, statement: Statement) -> None:
        self.statements.append(statement)

    def add_comment(self, contents: str) -> None:
        self.add_statement(SimpleStatement(f"// {contents}"))

    def add_if_else(self, if_else: IfElseStatement) -> None:
        if if_else.if_body.ends_in_jump():
            # Transform `if (A) { B; return C; } else { D; }`
            # into `if (A) { B; return C; } D;`,
            # which reduces indentation to make the output more readable
            self.statements.append(attr.evolve(if_else, else_body=None))
            if if_else.else_body is not None:
                self.extend(if_else.else_body)
            return

        self.statements.append(if_else)

    def add_switch(self, switch: SwitchStatement) -> None:
        self.add_statement(switch)

    def is_empty(self) -> bool:
        return not any(statement.should_write() for statement in self.statements)

    def ends_in_jump(self) -> bool:
        """
        Returns True if the body ends in an unconditional jump, which
        may allow for some syntax transformations.
        This function is conservative: it only returns True if we're
        *sure* if the control flow won't continue past the Body boundary.
        """
        for statement in self.statements[::-1]:
            if not statement.should_write():
                continue
            return isinstance(statement, SimpleStatement) and statement.is_jump
        return False

    def get_lone_if_statement(self) -> Optional[IfElseStatement]:
        """If the body consists solely of one IfElseStatement, return it, else None."""
        ret: Optional[IfElseStatement] = None
        for statement in self.statements:
            if statement.should_write():
                if not isinstance(statement, IfElseStatement) or ret:
                    return None
                ret = statement
        return ret

    def elide_empty_returns(self) -> None:
        """Remove `return;` statements from the end of the body.
        If the final statement is an if-else block, recurse into it."""
        for statement in self.statements[::-1]:
            if (
                isinstance(statement, SimpleStatement)
                and statement.contents == "return;"
            ):
                statement.contents = ""
            if not statement.should_write():
                continue
            if isinstance(statement, IfElseStatement):
                statement.if_body.elide_empty_returns()
                if statement.else_body is not None:
                    statement.else_body.elide_empty_returns()
            # We could also do this to SwitchStatements, but the generally
            # preferred style is to keep the final return/break
            break

    def format(self, fmt: Formatter) -> str:
        return "\n".join(
            statement.format(fmt)
            for statement in self.statements
            if statement.should_write()
        )


def label_for_node(context: Context, node: Node) -> str:
    if node in context.loop_nodes:
        return f"loop_{node.block.index}"
    else:
        return f"block_{node.block.index}"


def emit_node(context: Context, node: Node, body: Body) -> bool:
    """
    Try to emit a node for the first time, together with a label for it.
    The label is only printed if something jumps to it, e.g. a loop.

    For return nodes, it's preferred to emit multiple copies, rather than
    goto'ing a single return statement.

    For other nodes that were already emitted, instead emit a goto.
    Since nodes represent positions in assembly, and we use phi's for preserved
    variable contents, this will end up semantically equivalent. This can happen
    sometimes when early returns/continues/|| are not detected correctly, and
    this hints at that situation better than if we just blindly duplciate the block
    """
    if node in context.emitted_nodes:
        if not isinstance(node, ReturnNode):
            emit_goto(context, node, body)
            return False
    else:
        body.add_statement(LabelStatement(context, node))
        context.emitted_nodes.add(node)

    body.add_node(node, comment_empty=True)
    if isinstance(node, ReturnNode):  # and not node.is_real():
        emit_return(context, node, body)
    return True


def emit_goto(context: Context, target: Node, body: Body) -> None:
    assert not isinstance(target, TerminalNode), "cannot goto a TerminalNode"
    label = label_for_node(context, target)
    context.goto_nodes.add(target)
    body.add_statement(SimpleStatement(f"goto {label};", is_jump=True))


def switch_jump(context: Context, node: SwitchNode) -> SimpleStatement:
    block_info = node.block.block_info
    assert isinstance(block_info, BlockInfo)
    expr = block_info.switch_value
    assert expr is not None
    switch_index = context.switch_nodes.get(node, 0)
    comment = f" // switch {switch_index}" if switch_index else ""
    return SimpleStatement(
        f"goto *{format_expr(expr, context.fmt)};{comment}", is_jump=True
    )


def emit_goto_or_early_return(context: Context, target: Node, body: Body) -> None:
    """
    Emit a goto to a node, *unless* that node is a return or terminal node.
    This is similar to `emit_node`, but won't write the node body here unless
    the node is a return.
    """
    if isinstance(target, ReturnNode):
        emit_node(context, target, body)
    else:
        emit_goto(context, target, body)


def build_conditional_subgraph(
    context: Context, start: ConditionalNode, end: Node
) -> IfElseStatement:
    """
    Output the subgraph between `start` and `end`, including the branch condition
    in the ConditionalNode `start`.
    This function will intelligently output if/else relationships.
    """
    if_block_info = start.block.block_info
    assert isinstance(if_block_info, BlockInfo)
    assert if_block_info.branch_condition is not None

    else_body: Optional[Body] = None
    fallthrough_node: Node = start.fallthrough_edge
    conditional_node: Node = start.conditional_edge

    if fallthrough_node is end:
        # This case is quite rare, and either indicates an early return, an
        # empty if, or some sort of loop. In the loop case, we expect
        # build_flowgraph_between to end up noticing that the label has already
        # been seen and emit a goto, but in rare cases this might not happen.
        # If so it seems fine to emit the loop here.
        if_condition = if_block_info.branch_condition
        if_body = build_flowgraph_between(context, conditional_node, end)
    elif (
        isinstance(fallthrough_node, ConditionalNode)
        and (
            fallthrough_node.conditional_edge is conditional_node
            or fallthrough_node.fallthrough_edge is conditional_node
        )
        and context.options.andor_detection
    ):
        # The fallthrough node is also a conditional, with an edge pointing to
        # the same target as our conditional edge. This case comes up for
        # &&-statements and ||-statements, but also sometimes for regular
        # if-statements (a degenerate case of an &&/|| statement).
        return get_andor_if_statement(context, start, end)
    else:
        # This case is the most common. Since we missed the if above, we will
        # assume that taking the conditional edge does not perform any other
        # ||/&&-chain checks, but instead represents skipping the if body.
        # Thus, we split into an if-body and an else-body, though the latter
        # (for one reason or another) can still be empty.
        assert start.block.block_info
        assert start.block.block_info.branch_condition
        if_condition = start.block.block_info.branch_condition.negated()

        if_body = build_flowgraph_between(context, fallthrough_node, end)
        else_body = build_flowgraph_between(context, conditional_node, end)

    return IfElseStatement(if_condition, if_body, else_body)


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
    context: Context, start: ConditionalNode, end: Node
) -> IfElseStatement:
    """
    This function detects &&-statements, ||-statements, and
    degenerate forms of those - i.e. singular if-statements.

    As generated by IDO, &&-statements and ||-statements are
    emitted in a very particular way. In the below ASCII art,
    if (X) is a "COND", then the subgraph is an &&-statement,
    and 'alternative' represents the if body and 'bottom'
    the else body. Otherwise, i.e. if (X) is "FALL", then it
    is an ||-statement, and 'bottom' is the if body and
    'alternative' the else.

          +-------+      COND
          | start |---------------------+
          +-------+                     |
              | FALL                    |
              v                         |
          +-------+      COND           |
          |   1   |-------------------+ |
          +-------+                   | |
              | FALL                  | |
              v                       | |
    ___________________________________________
                                    . . .
                                    . . .
                                    . . .
    ___________________________________________
              |                     | | |
              | FALL                | | |
              v                     | | |
          +-------+    COND         | | |
          |  N-1  |---------------+ | | |
          +-------+               | | | |
              | FALL              | | | |
              v                   v v v v
          +-------+    (X)    +--------------+
          |   N   |---------->|    bottom    |
          +-------+           +--------------+
              |                   |
              | (Y)               |
              v                   |
     +-----------------+          |
     |   alternative   |          |
     +-----------------+          |
              |                   |
              v                   v
          +---------------------------+
          |            end            |
          +---------------------------+

    """
    conditions: List[Condition] = []
    condition_nodes: List[ConditionalNode] = []
    bottom = start.conditional_edge
    curr_node: ConditionalNode = start
    while True:
        # Collect conditions as we accumulate them:
        block_info = curr_node.block.block_info
        assert isinstance(block_info, BlockInfo)
        branch_condition = block_info.branch_condition
        assert branch_condition is not None
        if not conditions:
            # The first condition in an if-statement will have unrelated
            # statements in its to_write list, which our caller will already
            # have emitted. Avoid emitting them twice.
            conditions.append(branch_condition)
        else:
            # Include the statements in the condition by using a comma expression.
            conditions.append(gather_any_comma_conditions(block_info))
        condition_nodes.append(curr_node)

        # The next node will tell us whether we are in an &&/|| statement.
        # Our strategy will be:
        #   - Check if we have reached the end of an &&-statement
        #   - Check if we have reached the end of an ||-statement
        #   - Otherwise, assert we still fit the criteria of an &&/|| statement.
        next_node = curr_node.fallthrough_edge

        if (
            not isinstance(next_node, ConditionalNode)
            or (
                next_node.conditional_edge is not bottom
                and next_node.fallthrough_edge is not bottom
            )
            # An edge-case of our pattern-matching technology: without
            # this, self-loops match the pattern indefinitely, since a
            # self-loop node's conditional edge points to itself.
            or next_node.loop
            or len(next_node.parents) > 1
        ):
            # We reached the end of an && statement.

            if bottom is end:
                # If we don't need to emit an 'else', only emit && conditions up
                # to the first comma-statement condition, to avoid too much of
                # the output being sucked into if conditions.
                index = next(
                    (
                        i
                        for i, cond in enumerate(conditions)
                        if isinstance(cond, CommaConditionExpr)
                    ),
                    None,
                )

                if index is not None:
                    context.emitted_nodes |= set(condition_nodes[:index])
                    if_body = build_flowgraph_between(
                        context, condition_nodes[index], end
                    )
                    return IfElseStatement(
                        join_conditions(
                            [cond.negated() for cond in conditions[:index]], "&&"
                        ),
                        if_body=if_body,
                    )

            context.emitted_nodes |= set(condition_nodes)
            if_body = build_flowgraph_between(context, next_node, end)
            else_body = build_flowgraph_between(context, bottom, end)
            return IfElseStatement(
                # We negate everything, because the conditional edges will jump
                # OVER the if body.
                join_conditions([cond.negated() for cond in conditions], "&&"),
                if_body=if_body,
                else_body=else_body,
            )

        if next_node.fallthrough_edge is bottom:
            # End of || statement.
            assert next_node.block.block_info
            next_node_condition = gather_any_comma_conditions(
                next_node.block.block_info
            )
            context.emitted_nodes |= set(condition_nodes) | {next_node}
            if_body = build_flowgraph_between(context, bottom, end)
            else_body = build_flowgraph_between(
                context, next_node.conditional_edge, end
            )
            return IfElseStatement(
                # Negate the last condition, for it must fall-through to the
                # body instead of jumping to it, hence it must jump OVER the body.
                join_conditions(conditions + [next_node_condition.negated()], "||"),
                if_body=if_body,
                # The else-body is wherever the code jumps to instead of the
                # fallthrough (i.e. if-body).
                else_body=else_body,
            )

        # Otherwise, still in the middle of an &&/|| condition.
        assert next_node.conditional_edge is bottom
        curr_node = next_node
    assert False


def join_conditions(conditions: List[Condition], op: str) -> Condition:
    assert op in ["&&", "||"]
    assert conditions
    ret: Condition = conditions[0]
    for cond in conditions[1:]:
        ret = BinaryOp(ret, op, cond, type=Type.bool())
    return ret


def emit_return(context: Context, node: ReturnNode, body: Body) -> None:
    ret_info = node.block.block_info
    assert isinstance(ret_info, BlockInfo)

    ret = ret_info.return_value
    if ret is not None:
        ret_str = format_expr(ret, context.fmt)
        body.add_statement(SimpleStatement(f"return {ret_str};", is_jump=True))
        context.is_void = False
    else:
        body.add_statement(SimpleStatement("return;", is_jump=True))


def build_switch_between(
    context: Context,
    start: SwitchNode,
    end: Node,
) -> SwitchStatement:
    """Output the subgraph between `start` and `end`, including the jump
    in the SwitchStatement `start`, but not including `end`"""
    body = Body(print_node_comment=context.options.debug)
    jump = switch_jump(context, start)
    emitted_cases: Set[Node] = set()

    for case in start.cases:
        if case in context.emitted_nodes:
            continue
        if case == end:
            # We are not responsible for emitting `end`
            continue

        body.extend(build_flowgraph_between(context, case, end))
        # TODO: This could be a `break;` statement
        if not body.ends_in_jump():
            emit_goto_or_early_return(context, end, body)
    return SwitchStatement(jump, body)


def build_flowgraph_between(context: Context, start: Node, end: Node) -> Body:
    """
    Output a section of a flow graph that has already been translated to our
    symbolic AST. All nodes between start and end, including start but NOT end,
    will be printed out using if-else statements and block info
    """
    curr_start: Node = start
    body = Body(print_node_comment=context.options.debug)

    # We will split this graph into subgraphs, where the entrance and exit nodes
    # of that subgraph are at the same indentation level. "curr_start" will
    # iterate through these nodes by taking the immediate postdominators,
    # which are commonly referred to as articulation nodes.
    while curr_start != end:
        assert not isinstance(curr_start, TerminalNode)

        # Write the current node, or a goto, to the body
        if not emit_node(context, curr_start, body):
            # If the node was already witten, emit_node will use a goto
            # and return False. After the jump, there control flow will
            # continue from there (hopefully hitting `end`!)
            break

        if curr_start.emit_goto:
            # If we have decided to emit a goto here, then we should just fall
            # through to the next node by index, after writing a goto.
            print(f">>> GOTO {curr_start.name()}")
            emit_goto(context, curr_start, body)

            # Advance to the next node in block order. This may skip over
            # unreachable blocks -- hopefully none too important.
            index = context.flow_graph.nodes.index(curr_start)
            fallthrough = context.flow_graph.nodes[index + 1]
            if isinstance(curr_start, ConditionalNode):
                assert fallthrough == curr_start.fallthrough_edge
            curr_start = fallthrough
            continue

        # The interval to process is [curr_start, curr_start.immediate_postdominator)
        curr_end = curr_start.immediate_postdominator
        assert curr_end is not None

        # For nodes with branches, curr_end is not a direct successor of curr_start
        if isinstance(curr_start, ConditionalNode):
            body.add_if_else(build_conditional_subgraph(context, curr_start, curr_end))
        elif isinstance(curr_start, SwitchNode):
            body.add_switch(build_switch_between(context, curr_start, curr_end))
        else:
            # No branch, but check that we didn't skip any nodes
            assert curr_start.children() == {curr_end}

        # Move on.
        curr_start = curr_end
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
        emit_goto(context, node, body)

    for i, node in enumerate(nodes):
        if isinstance(node, ReturnNode):
            # Do not emit duplicated (non-real) return nodes; they don't have
            # a well-defined position, so we emit them next to where they are
            # jumped to instead.
            if node.is_real():
                emit_node(context, node, body)
        elif isinstance(node, BasicNode):
            emit_node(context, node, body)
            emit_successor(node.successor, i)
        elif isinstance(node, SwitchNode):
            emit_node(context, node, body)
            body.add_statement(switch_jump(context, node))
        elif isinstance(node, ConditionalNode):
            emit_node(context, node, body)
            if_body = Body(print_node_comment=True)
            emit_goto(context, node.conditional_edge, if_body)
            block_info = node.block.block_info
            assert isinstance(block_info, BlockInfo)
            assert block_info.branch_condition is not None
            body.add_if_else(
                IfElseStatement(
                    block_info.branch_condition,
                    if_body=if_body,
                    else_body=None,
                )
            )
            emit_successor(node.fallthrough_edge, i)
        else:
            assert isinstance(node, TerminalNode)

    return body


def build_body(context: Context, options: Options) -> Body:
    start_node: Node = context.flow_graph.entry_node()
    terminal_node: Node = context.flow_graph.terminal_node()
    is_reducible = context.flow_graph.is_reducible()

    num_switches = len(
        [node for node in context.flow_graph.nodes if isinstance(node, SwitchNode)]
    )
    switch_index = 0
    for node in context.flow_graph.nodes:
        if node.loop is not None:
            context.loop_nodes.add(node)
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
                # (jump table entry 0 is never covered by 'default'; the compiler would
                # do 'switch (x - 1)' in that case)
                if has_common and target == most_common and index != 0:
                    continue
                context.case_nodes[target].append((switch_index, index))

    if options.debug:
        print("Here's the whole function!\n")

    body: Body
    if options.ifs and is_reducible:
        body = build_flowgraph_between(context, start_node, terminal_node)
        body.elide_empty_returns()
    else:
        body = build_naive(context, context.flow_graph.nodes)

    # Check no nodes were skipped: build_flowgraph_between should hit every node in
    # well-formed (reducible) graphs; and build_naive explicitly emits every node
    unemitted_nodes = (
        set(context.flow_graph.nodes)
        - context.emitted_nodes
        - {context.flow_graph.terminal_node()}
    )
    for node in unemitted_nodes:
        body.add_comment(f"bug: did not emit code for node #{node.name()}")

    return body


def get_function_text(function_info: FunctionInfo, options: Options) -> str:
    fmt = options.formatter()
    context = Context(flow_graph=function_info.flow_graph, options=options, fmt=fmt)
    body: Body = build_body(context, options)

    function_lines: List[str] = []

    fn_name = function_info.stack_info.function.name
    arg_strs = []
    for arg in function_info.stack_info.arguments:
        arg_strs.append(arg.type.to_decl(arg.format(fmt), fmt))
    if function_info.stack_info.is_variadic:
        arg_strs.append("...")
    arg_str = ", ".join(arg_strs) or "void"

    fn_header = f"{fn_name}({arg_str})"

    if context.is_void:
        fn_header = f"void {fn_header}"
    else:
        fn_header = function_info.return_type.to_decl(fn_header, fmt)
    whitespace = "\n" if fmt.coding_style.newline_after_function else " "
    function_lines.append(f"{fn_header}{whitespace}{{")

    any_decl = False

    with fmt.indented():
        for local_var in function_info.stack_info.local_vars[::-1]:
            type_decl = local_var.type.to_decl(local_var.format(fmt), fmt)
            function_lines.append(SimpleStatement(f"{type_decl};").format(fmt))
            any_decl = True

        # With reused temps (no longer used), we can get duplicate declarations,
        # hence the use of a set here.
        temp_decls = set()
        for temp_var in function_info.stack_info.temp_vars:
            if temp_var.need_decl():
                expr = temp_var.expr
                type_decl = expr.type.to_decl(expr.var.format(fmt), fmt)
                temp_decls.add(f"{type_decl};")
                any_decl = True
        for decl in sorted(temp_decls):
            function_lines.append(SimpleStatement(decl).format(fmt))

        for phi_var in function_info.stack_info.phi_vars:
            type_decl = phi_var.type.to_decl(phi_var.get_var_name(), fmt)
            function_lines.append(SimpleStatement(f"{type_decl};").format(fmt))
            any_decl = True

        for reg_var in function_info.stack_info.reg_vars.values():
            if reg_var.register not in function_info.stack_info.used_reg_vars:
                continue
            type_decl = reg_var.type.to_decl(reg_var.format(fmt), fmt)
            function_lines.append(SimpleStatement(f"{type_decl};").format(fmt))
            any_decl = True

        if any_decl:
            function_lines.append("")

        function_lines.append(body.format(fmt))
        function_lines.append("}")
    full_function_text: str = "\n".join(function_lines)
    return full_function_text
