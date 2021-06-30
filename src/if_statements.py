from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Set, Tuple, Union

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
)
from .translate import Statement as TrStatement
from .translate import Type, format_expr, simplify_condition


@dataclass
class Context:
    flow_graph: FlowGraph
    fmt: Formatter
    options: Options
    is_void: bool = True
    switch_nodes: Dict[SwitchNode, int] = field(default_factory=dict)
    case_nodes: Dict[Node, List[Tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    goto_nodes: Set[Node] = field(default_factory=set)
    loop_nodes: Set[Node] = field(default_factory=set)
    emitted_nodes: Set[Node] = field(default_factory=set)
    has_warned: bool = False


@dataclass
class IfElseStatement:
    condition: Condition
    if_body: "Body"
    else_body: Optional["Body"] = None

    def should_write(self) -> bool:
        return True

    def format(self, fmt: Formatter) -> str:
        space = fmt.indent("")
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
        return replace(
            self,
            condition=self.condition.negated(),
            if_body=self.else_body,
            else_body=self.if_body,
        )


@dataclass
class SwitchStatement:
    jump: "SimpleStatement"
    body: "Body"

    def should_write(self) -> bool:
        return True

    def format(self, fmt: Formatter) -> str:
        lines = [
            self.jump.format(fmt),
            self.body.format(fmt),
        ]
        return "\n".join(lines)


@dataclass
class SimpleStatement:
    contents: Optional[Union[str, TrStatement]]
    is_jump: bool = False

    def should_write(self) -> bool:
        return self.contents is not None

    def format(self, fmt: Formatter) -> str:
        if self.contents is None:
            return ""
        elif isinstance(self.contents, str):
            return fmt.indent(self.contents)
        else:
            return fmt.indent(self.contents.format(fmt))

    def clear(self) -> None:
        self.contents = None


@dataclass
class LabelStatement:
    context: Context
    node: Node

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
                lines.append(fmt.indent(f"{case_str}:{switch_str}", -1))
        if self.node in self.context.goto_nodes:
            lines.append(f"{label_for_node(self.context, self.node)}:")
        return "\n".join(lines)


@dataclass
class DoWhileLoop:
    body: "Body"
    condition: Condition

    def should_write(self) -> bool:
        return True

    def format(self, fmt: Formatter) -> str:
        space = fmt.indent("")
        after_do = f"\n{space}" if fmt.coding_style.newline_after_if else " "
        cond = format_expr(self.condition, fmt)
        with fmt.indented():
            return "\n".join(
                [
                    f"{space}do{after_do}{{",
                    self.body.format(fmt),
                    f"{space}}} while ({cond});",
                ]
            )


Statement = Union[
    SimpleStatement,
    IfElseStatement,
    LabelStatement,
    SwitchStatement,
    DoWhileLoop,
]


@dataclass
class Body:
    print_node_comment: bool
    statements: List[Statement] = field(default_factory=list)

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
            self.statements.append(replace(if_else, else_body=None))
            if if_else.else_body is not None:
                self.extend(if_else.else_body)
            return

        self.statements.append(if_else)

    def add_do_while_loop(self, do_while_loop: DoWhileLoop) -> None:
        self.statements.append(do_while_loop)

    def add_switch(self, switch: SwitchStatement) -> None:
        self.add_statement(switch)

    def is_empty(self) -> bool:
        return not any(statement.should_write() for statement in self.statements)

    def ends_in_jump(self) -> bool:
        """
        Returns True if the body ends in an unconditional jump (`goto` or `return`),
        which may allow for some syntax transformations.
        For example, this is True for bodies ending in a ReturnNode, because
        `return ...;` statements are marked with is_jump.
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
                statement.clear()
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
    this hints at that situation better than if we just blindly duplicate the block
    """
    if node in context.emitted_nodes:
        # TODO: Treating ReturnNode as a special case and emitting it repeatedly
        # hides the fact that we failed to fold the control flow. Maybe remove?
        if not isinstance(node, ReturnNode):
            emit_goto(context, node, body)
            return False
        else:
            body.add_comment(
                f"Duplicate return node #{node.name()}. Try simplifying control flow for better match"
            )
    else:
        body.add_statement(LabelStatement(context, node))
        context.emitted_nodes.add(node)

    body.add_node(node, comment_empty=True)
    if isinstance(node, ReturnNode):
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
    elif not isinstance(target, TerminalNode):
        emit_goto(context, target, body)


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


def build_conditional_subgraph(
    context: Context, start: ConditionalNode, end: Node
) -> IfElseStatement:
    """
    Output the subgraph between `start` and `end`, including the branch condition
    in the ConditionalNode `start`.
    This function detects &&-statements, ||-statements, and degenerate forms of
    those (i.e. singular if-statements or self-loops)

    As generated by IDO and GCC, &&-statements and ||-statements are emitted in a
    very particular way. A "chain" ConditionalNodes will exit, where each node
    falls through to the next node in the chain.
    Each conditional edge (branch) from the nodes in this chain will be to one of:
        - The head of the if block body (`if_node`)
        - The head of the else block body (`else_node`)
        - A *later* node in the chain

    The `if_node` and `else_node` are mostly symmetric, modulo some negations.
    However, we know IDO likes to emit the assembly for basic blocks in approximately
    the same order that they appear in the C source. So, we generally call the
    `if_node` the node that is the fallthrough of the final ConditionNode, because
    its contents will generally be earlier in the assembly.
    (The exception is when `if_node` is `end`, in which case the if body would be empty)
    """

    # Find the longest fallthrough chain of ConditionalNodes.
    # This is the starting point for finding the single complex and/or conditional.
    # Exclude any loop nodes (except `start`) and nodes not postdominated by `end`.
    # The conditional edges will be checked in later step
    chained_cond_nodes: List[ConditionalNode] = []
    curr_node: ConditionalNode = start
    while True:
        if end not in curr_node.fallthrough_edge.postdominators:
            break
        chained_cond_nodes.append(curr_node)
        if not context.options.andor_detection:
            # and/or detection is disabled; so limit the chain length to 1 node
            break
        if not isinstance(curr_node.fallthrough_edge, ConditionalNode):
            break
        curr_node = curr_node.fallthrough_edge
        if curr_node.loop or any(
            p not in chained_cond_nodes for p in curr_node.parents
        ):
            break

    # We want to take the largest chain of ConditionalNodes where each node's edges
    # are only to other nodes forward in the chain, the `if_node`, or the `else_node`.
    # We start with the largest chain computed above, and then trim it until it
    # meets this criteria. The resulting chain will always have at least one node.
    if_node: Node
    else_node: Optional[Node]
    while True:
        assert chained_cond_nodes
        if_node = chained_cond_nodes[-1].fallthrough_edge
        else_node = chained_cond_nodes[-1].conditional_edge

        # If the else body would be empty (`else_node is end`), then compute which
        # nodes are part of a top-level && expression (if it exists).
        # If any of these nodes would add comma expressions, we'll split them into
        # separate if statements instead of bundling them into this condition.
        potential_partition_nodes = set()
        if else_node is end:
            node: Node = chained_cond_nodes[0]
            while node in chained_cond_nodes:
                assert isinstance(node, ConditionalNode)
                potential_partition_nodes.add(node)
                if node.conditional_edge == end:
                    # `!node && node.conditional_edge`
                    node = node.fallthrough_edge
                else:
                    # `node && node.conditional_edge`
                    node = node.conditional_edge
            if node is not if_node:
                potential_partition_nodes = set()

        # Set of nodes that are permitted to be the target of conditional_edge
        allowed_nodes = set(chained_cond_nodes) | {if_node, else_node}
        for node in chained_cond_nodes:
            block_info = node.block.block_info
            assert block_info
            if node.conditional_edge not in allowed_nodes or (
                node is not start
                and node.conditional_edge is end
                and node in potential_partition_nodes
                and next((True for b in block_info.to_write if b.should_write()), False)
            ):
                # Shorten the chain by removing the last node, then try again
                chained_cond_nodes.pop()
                break
            # Conditional edges must point "forward," so remove `node` from the allowed set
            allowed_nodes.remove(node)
        else:
            # No changes were made: we found the largest chain that matches the constraints
            break

    def traverse(node: Node, if_node: Node, else_node: Node) -> Tuple[Node, Condition]:
        or_terms = []
        # Iterate through the chain of conditionals, until we hit one of the ends.
        # The `... or node is start` lets us reuse this logic to handle self loops:
        # although there aren't any loops *inside* the chain, `start` is allowed to
        # be a loop, and it may point to itself.
        while node not in (if_node, else_node) or node is start:
            assert (
                isinstance(node, ConditionalNode) and node in chained_cond_nodes
            ), f"{(node, if_node, else_node)}"

            block_info = node.block.block_info
            assert block_info
            if node is start:
                # The first condition in an if-statement will have unrelated
                # statements in its to_write list, which our caller will already
                # have emitted. Avoid emitting them twice.
                cond = block_info.branch_condition
            else:
                # Otherwise, these statements will be added to the condition
                cond = gather_any_comma_conditions(block_info)
                context.emitted_nodes.add(node)

            if node.conditional_edge is if_node:
                # The node means: "If `cond`, goto `if_node`", so append `cond` to `or_terms` as-is
                or_terms.append(cond)
                node = node.fallthrough_edge
            elif node.fallthrough_edge is if_node:
                # The node means: "If `!cond`, goto `if_node`": so append `!cond` to `or_terms`...
                or_terms.append(cond.negated())
                if node.conditional_edge is else_node:
                    # We're done, we hit the bottom!
                    # (In this case, the recursion would fail because `node == else_node`)
                    node = node.fallthrough_edge
                else:
                    # Here, if any 1 condition in `or_terms` is true, can reach `if_node`.
                    # But if they're all false, we end up at `node.conditional_edge`.
                    #
                    # This section looks for node structures with the following form:
                    # ```
                    #   if (or_terms) {
                    #       if (tail_cond) {            // tail_cond starts with the `if_node` condition
                    #           else_node;
                    #       } else {
                    #           node.conditional_edge
                    #       }
                    #   } else {
                    #       node.conditional_edge
                    #   }
                    # ```
                    #
                    # ...and re-writes them into a single if statement:
                    # ```
                    #   if (!or_terms || !tail_cond) {
                    #       node.conditional_edge;
                    #   } else {
                    #       else_node;                  // this is the returned `node` from traverse(...)
                    #   }
                    # ```
                    assert if_node in chained_cond_nodes
                    next_node, tail_cond = traverse(
                        if_node, else_node, node.conditional_edge
                    )
                    if next_node is else_node:
                        tail_cond = tail_cond.negated()
                        node = next_node
                    else:
                        assert next_node is node.conditional_edge
                    or_terms = [
                        join_conditions(or_terms, "||").negated(),
                        tail_cond,
                    ]
                break
            else:
                # Both branches from the node point to other nodes, so we need to create
                # a parenthetical expression, similar to above.
                # This section looks for node structures with the following form:
                # ```
                #   if (or_terms) {
                #       if_node;
                #   } else {
                #       if (tail_cond) {            // `tail_cond` starts with the `node` condition
                #           next_node;
                #       } else {
                #           if_node;
                #       }
                #   }
                # ```
                #
                # ...and re-writes them into a single if statement:
                # ```
                #   if (or_terms || !tail_cond) {
                #       if_node;
                #   } else {
                #       next_node;                  // this is the returned `node` from traverse(...)
                #   }
                # ```
                node, tail_cond = traverse(node, node.conditional_edge, if_node)
                or_terms.append(tail_cond.negated())
        assert node in (if_node, else_node), f"{(node, if_node, else_node)}"
        return node, join_conditions(or_terms, "||")

    final_node, cond = traverse(start, if_node, else_node)
    assert final_node is if_node

    if else_node is end:
        # No need to emit an `else` block
        else_node = None
    elif if_node is end:
        # This is rare, and either indicates an empty if, or some sort of loop
        cond = cond.negated()
        if_node, else_node = else_node, if_node

    # Build the if & else bodies
    if_body = build_flowgraph_between(context, if_node, end)
    else_body: Optional[Body] = None
    if else_node:
        else_body = build_flowgraph_between(context, else_node, end)

    return IfElseStatement(cond, if_body, else_body)


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


def detect_loop(context: Context, start: Node, end: Node) -> Optional[DoWhileLoop]:
    assert start.loop

    # Find the the condition for the do-while, if it exists
    condition: Optional[Condition] = None
    for node in start.loop.backedges:
        if (
            node in start.postdominators
            and isinstance(node, ConditionalNode)
            and node.fallthrough_edge == end
        ):
            assert node.block.block_info
            assert node.block.block_info.branch_condition
            condition = node.block.block_info.branch_condition
            new_end = node
            break
    if not condition:
        return None

    loop_body = build_flowgraph_between(
        context,
        start,
        new_end,
        skip_loop_detection=True,
    )
    emit_node(context, new_end, loop_body)

    return DoWhileLoop(loop_body, condition)


def build_flowgraph_between(
    context: Context, start: Node, end: Node, skip_loop_detection: bool = False
) -> Body:
    """
    Output a section of a flow graph that has already been translated to our
    symbolic AST. All nodes between start and end, including start but NOT end,
    will be printed out using if-else statements and block info.

    `skip_loop_detection` is used to prevent infinite recursion, since (in the
    case of loops) this function can be recursively called by itself (via
    `detect_loop`) with the same `start` argument.
    """
    curr_start: Node = start
    body = Body(print_node_comment=context.options.debug)

    # We will split this graph into subgraphs, where the entrance and exit nodes
    # of that subgraph are at the same indentation level. "curr_start" will
    # iterate through these nodes by taking the immediate postdominators,
    # which are commonly referred to as articulation nodes.
    while curr_start != end:
        assert not isinstance(curr_start, TerminalNode)

        if (
            not skip_loop_detection
            and curr_start.loop
            and not curr_start in context.emitted_nodes
        ):
            # Find the immediate postdominator to the whole loop,
            # i.e. the first node outside the loop body
            imm_pdom: Node = curr_start
            while imm_pdom in curr_start.loop.nodes:
                assert imm_pdom.immediate_postdominator is not None
                imm_pdom = imm_pdom.immediate_postdominator

            # Construct the do-while loop
            do_while_loop = detect_loop(context, curr_start, imm_pdom)
            if do_while_loop:
                body.add_do_while_loop(do_while_loop)

                # Move on.
                curr_start = imm_pdom
                continue

        # Write the current node, or a goto, to the body
        if not emit_node(context, curr_start, body):
            # If the node was already witten, emit_node will use a goto
            # and return False. After the jump, there control flow will
            # continue from there (hopefully hitting `end`!)
            break

        if curr_start.emit_goto:
            # If we have decided to emit a goto here, then we should just fall
            # through to the next node by index, after writing a goto.
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
            # No branch, but double check that we didn't skip any nodes.
            # If the check fails, then the immediate_postdominator computation was wrong
            assert curr_start.children() == [curr_end], (
                f"While emitting flowgraph between {start.name()}:{end.name()}, "
                f"skipped nodes while stepping from {curr_start.name()} to {curr_end.name()}."
            )

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
        if isinstance(node, ReturnNode) and not node.is_real():
            continue
        body.add_comment(
            f"bug: did not emit code for node #{node.name()}; contents below:"
        )
        emit_node(context, node, body)

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
            if reg_var.reg not in function_info.stack_info.used_reg_vars:
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
