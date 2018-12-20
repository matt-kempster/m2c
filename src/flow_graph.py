import copy
import typing
from typing import (Any, Callable, Dict, Iterator, List, Optional, Set, Tuple,
                    Union)

import attr

from error import DecompFailure
from parse_file import Function, Label
from parse_instruction import (AsmLiteral, Instruction, JumpTarget, Register,
                               parse_instruction)


@attr.s(cmp=False)
class Block:
    index: int = attr.ib()
    label: Optional[Label] = attr.ib()
    approx_label_name: Optional[str] = attr.ib()
    instructions: List[Instruction] = attr.ib(factory=list)

    # TODO: fix "Any" to be "BlockInfo" (currently annoying due to circular imports)
    block_info: Optional[Any] = None

    def add_block_info(self, block_info: Any) -> None:
        self.block_info = block_info

    def clone(self) -> 'Block':
        return copy.deepcopy(self)

    def __str__(self) -> str:
        if self.approx_label_name:
            name = f'{self.index} ({self.approx_label_name})'
        else:
            name = f'{self.index}'
        inst_str = '\n'.join(str(instruction) for instruction in self.instructions)
        return f'# {name}\n{inst_str}\n'

@attr.s(cmp=False)
class BlockBuilder:
    curr_index: int = attr.ib(default=0)
    curr_label: Optional[Label] = attr.ib(default=None)
    last_label: Optional[Label] = attr.ib(default=None)
    label_counter: int = attr.ib(default=0)
    curr_instructions: List[Instruction] = attr.ib(factory=list)
    blocks: List[Block] = attr.ib(factory=list)

    def new_block(self) -> Optional[Block]:
        if len(self.curr_instructions) == 0:
            return None

        label_name = None
        if self.last_label is not None:
            label_name = self.last_label.name
            if self.label_counter > 0:
                label_name += f'.{self.label_counter}'
        block = Block(self.curr_index, self.curr_label, label_name,
                self.curr_instructions)
        self.blocks.append(block)

        self.curr_index += 1
        self.curr_label = None
        self.label_counter += 1
        self.curr_instructions = []

        return block

    def add_instruction(self, instruction: Instruction) -> None:
        self.curr_instructions.append(instruction)

    def set_label(self, label: Label) -> None:
        if label == self.curr_label:
            # It's okay to repeat a label (e.g. once with glabel, once as a
            # standard label -- this often occurs for switches).
            return
        # We could support multiple labels at the same position, and output
        # empty blocks. For now we don't, however.
        assert not self.curr_label, "A block can not have more than one label"
        self.curr_label = label
        self.last_label = label
        self.label_counter = 0

    def get_blocks(self) -> List[Block]:
        return self.blocks


temp_label_counter: int = 0
def generate_temp_label(name: str) -> str:
    global temp_label_counter
    temp_label_counter += 1
    return 'Ltemp' + str(temp_label_counter) + ('_' + name if name else '')


# Branch-likely instructions only evaluate their delay slots when they are
# taken, making control flow more complex. However, on the IRIX compiler they
# only occur in a very specific pattern:
#
# ...
# <branch likely instr> .label
#  X
# ...
# X
# .label:
# ...
#
# which this function transforms back into a regular branch pattern by moving
# the label one step back and replacing the delay slot by a nop.
#
# Branch-likely instructions that do not appear in this pattern are kept.
def normalize_likely_branches(function: Function) -> Function:
    label_prev_instr: Dict[str, Instruction] = {}
    label_before_instr: Dict[int, str] = {}
    prev_instr: Optional[Instruction] = None
    prev_label: Optional[Label] = None
    for item in function.body:
        if isinstance(item, Instruction):
            if prev_label is not None:
                label_before_instr[id(item)] = prev_label.name
                prev_label = None
            prev_instr = item
        elif isinstance(item, Label):
            assert prev_instr is not None
            label_prev_instr[item.name] = prev_instr
            prev_label = item

    insert_label_before: Dict[int, str] = {}
    new_body: List[Tuple[Union[Instruction, Label], Union[Instruction, Label]]] = []

    body_iter: Iterator[Union[Instruction, Label]] = iter(function.body)
    for item in body_iter:
        orig_item = item
        if isinstance(item, Instruction) and item.is_branch_likely_instruction():
            old_label = item.get_branch_target().target
            before_target = label_prev_instr[old_label]
            next_item = next(body_iter)
            orig_next_item = next_item
            if isinstance(next_item, Instruction) and str(before_target) == str(next_item):
                if id(before_target) not in label_before_instr:
                    new_label = generate_temp_label(old_label)
                    label_before_instr[id(before_target)] = new_label
                    insert_label_before[id(before_target)] = new_label
                new_target = JumpTarget(label_before_instr[id(before_target)])
                item = Instruction(item.mnemonic[:-1], item.args[:-1] + [new_target])
                next_item = Instruction('nop', [])
            new_body.append((orig_item, item))
            new_body.append((orig_next_item, next_item))
        else:
            new_body.append((orig_item, item))

    new_function = Function(name=function.name)
    for (orig_item, new_item) in new_body:
        if id(orig_item) in insert_label_before:
            new_function.new_label(insert_label_before[id(orig_item)])
        new_function.body.append(new_item)

    return new_function


def prune_unreferenced_labels(function: Function) -> Function:
    labels_used : Set[str] = set(l.name for l in function.jumptable_labels)
    for item in function.body:
        if isinstance(item, Instruction) and item.is_branch_instruction():
            labels_used.add(item.get_branch_target().target)

    new_function = Function(name=function.name)
    for item in function.body:
        if not (isinstance(item, Label) and item.name not in labels_used):
            new_function.body.append(item)

    return new_function


# Detect and simplify various standard patterns emitted by the IRIX compiler.
# Currently handled:
# - checks for x/0 and INT_MIN/-1 after division (removed)
# - unsigned to float conversion (converted to a made-up instruction)
def simplify_standard_patterns(function: Function) -> Function:
    BodyPart = Union[Instruction, Label]

    div_pattern: List[str] = [
        "bnez",
        "nop",
        "break 7",
        "",
        "li $at, -1",
        "bne",
        "li $at, 0x80000000",
        "bne",
        "nop",
        "break 6",
        "",
    ]

    divu_pattern: List[str] = [
        "bnez",
        "nop",
        "break 7",
        "",
    ]

    utf_pattern: List[str] = [
        "bgez",
        "cvt.s.w",
        "li $at, 0x4f800000",
        "mtc1",
        "nop",
        "add.s",
        "",
    ]

    def get_li_imm(ins: Instruction) -> Optional[int]:
        if ins.mnemonic == 'lui' and isinstance(ins.args[1], AsmLiteral):
            return (ins.args[1].value & 0xffff) << 16
        if (ins.mnemonic in ['addi', 'addiu'] and
                ins.args[1] == Register('zero') and
                isinstance(ins.args[2], AsmLiteral)):
            val = ins.args[2].value & 0xffff
            if val >= 0x8000:
                val -= 0x10000
            return val & 0xffffffff
        if (ins.mnemonic == 'ori' and ins.args[1] == Register('zero') and
                isinstance(ins.args[2], AsmLiteral)):
            return ins.args[2].value & 0xffff
        return None

    def matches_pattern(actual: List[BodyPart], pattern: List[str]) -> bool:
        def match_one(actual: BodyPart, expected: str) -> bool:
            if not isinstance(actual, Instruction):
                return (expected == "")
            ins = actual
            exp = parse_instruction(expected)
            if not exp.args:
                return ins.mnemonic == exp.mnemonic
            if str(ins) == str(exp):
                return True
            # A bit of an ugly hack, but since 'li' can be spelled many ways...
            return (exp.mnemonic == 'li' and exp.args[0] == ins.args[0] and
                    isinstance(exp.args[1], AsmLiteral) and
                    (exp.args[1].value & 0xffffffff) == get_li_imm(ins))

        return (len(actual) == len(pattern) and
            all(match_one(a, e) for (a, e) in zip(actual, pattern)))

    def try_replace_div(i: int) -> Optional[Tuple[List[BodyPart], int]]:
        actual = function.body[i:i + len(div_pattern)]
        if not matches_pattern(actual, div_pattern):
            return None
        label1 = typing.cast(Label, actual[3])
        label2 = typing.cast(Label, actual[10])
        bnez = typing.cast(Instruction, actual[0])
        bne1 = typing.cast(Instruction, actual[5])
        bne2 = typing.cast(Instruction, actual[7])
        if (bnez.get_branch_target().target != label1.name or
                bne1.get_branch_target().target != label2.name and
                bne2.get_branch_target().target != label2.name):
            return None
        return ([], i + len(div_pattern) - 1)

    def try_replace_divu(i: int) -> Optional[Tuple[List[BodyPart], int]]:
        actual = function.body[i:i + len(divu_pattern)]
        if not matches_pattern(actual, divu_pattern):
            return None
        label = typing.cast(Label, actual[3])
        bnez = typing.cast(Instruction, actual[0])
        if bnez.get_branch_target().target != label.name:
            return None
        return ([], i + len(divu_pattern) - 1)

    def try_replace_utf_conv(i: int) -> Optional[Tuple[List[BodyPart], int]]:
        actual = function.body[i:i + len(utf_pattern)]
        if not matches_pattern(actual, utf_pattern):
            return None
        label = typing.cast(Label, actual[6])
        bgez = typing.cast(Instruction, actual[0])
        if bgez.get_branch_target().target != label.name:
            return None
        cvt_instr = typing.cast(Instruction, actual[1])
        new_instr = Instruction(mnemonic="cvt.s.u", args=cvt_instr.args)
        return ([new_instr], i + len(utf_pattern) - 1)

    def no_replacement(i: int) -> Tuple[List[BodyPart], int]:
        return ([function.body[i]], i + 1)

    new_function = Function(name=function.name)
    i = 0
    while i < len(function.body):
        repl, i = (try_replace_div(i) or try_replace_divu(i) or
                try_replace_utf_conv(i) or no_replacement(i))
        new_function.body.extend(repl)
    return new_function


def build_blocks(function: Function) -> List[Block]:
    function = normalize_likely_branches(function)
    function = prune_unreferenced_labels(function)
    function = simplify_standard_patterns(function)
    function = prune_unreferenced_labels(function)

    block_builder = BlockBuilder()

    body_iter: Iterator[Union[Instruction, Label]] = iter(function.body)
    def process(item: Union[Instruction, Label]) -> None:
        process_after: List[Union[Instruction, Label]] = []
        if isinstance(item, Label):
            # Split blocks at labels.
            block_builder.new_block()
            block_builder.set_label(item)
        elif isinstance(item, Instruction):
            process_after = []
            if item.is_delay_slot_instruction():
                next_item = next(body_iter)
                if isinstance(next_item, Label):
                    # Delay slot is a jump target, so we need the delay slot
                    # instruction to be in two blocks at once... In most cases,
                    # we can just duplicate it. (This happens from time to time
                    # in -O2-compiled code.)
                    label = next_item
                    next_item = next(body_iter)

                    assert isinstance(next_item, Instruction), "Cannot have two labels in a row"

                    # (Best-effort check for whether the instruction can be
                    # executed twice in a row.)
                    r = next_item.args[0] if next_item.args else None
                    if all(a != r for a in next_item.args[1:]):
                        process_after.append(label)
                        process_after.append(next_item)
                    else:
                        msg = [
                            f"Label {label.name} refers to a delay slot; this is currently not supported.",
                            "Please modify the assembly to work around it (e.g. copy the instruction",
                            "to all jump sources and move the label, or add a nop to the delay slot)."
                        ]
                        if label.name.startswith('Ltemp'):
                            msg += [
                                "",
                                "This label was auto-generated as the target for a branch-likely",
                                "instruction (e.g. beql); you can also try to turn that into a non-likely",
                                "branch if that's equivalent in this context, i.e., if it's okay to",
                                "execute its delay slot unconditionally."
                            ]
                        raise DecompFailure('\n'.join(msg))

                if next_item.is_delay_slot_instruction():
                    raise DecompFailure(f"Two delay slot instructions in a row is not supported:\n{item}\n{next_item}")

                if item.is_branch_likely_instruction():
                    raise DecompFailure("Not yet able to handle general branch-likely instruction:\n" \
                            f"{item}\n\n" \
                            "Only branch-likely instructions which can be turned into non-likely\n" \
                            "versions pointing one step up are currently supported. Try rewriting\n" \
                            "the assembly using non-branch-likely instructions.")

                if item.mnemonic in ['jal', 'jr', 'jalr']:
                    # Move the delay slot instruction to before the call/return
                    # so it passes correct arguments.
                    # TODO: do something about jalr $r/jr $r where $r is
                    # clobbered in the delay slot instruction...
                    block_builder.add_instruction(next_item)
                    block_builder.add_instruction(item)
                else:
                    block_builder.add_instruction(item)
                    block_builder.add_instruction(next_item)

                if item.is_jump_instruction():
                    # Split blocks at jumps, after the next instruction.
                    block_builder.new_block()
            else:
                block_builder.add_instruction(item)

        for item in process_after:
            process(item)

    for item in body_iter:
        process(item)

    # Throw away whatever is past the last "jr $ra" and return what we have.
    return block_builder.get_blocks()


def is_loop_edge(node: 'Node', edge: 'Node') -> bool:
    # Loops are represented by backwards jumps.
    return typing.cast(bool, (edge.block.index <= node.block.index))

@attr.s(cmp=False)
class BaseNode:
    block: Block = attr.ib()
    parents: List['Node'] = attr.ib(init=False, factory=list)
    dominators: Set['Node'] = attr.ib(init=False, factory=set)
    immediate_dominator: Optional['Node'] = attr.ib(init=False, default=None)
    immediately_dominates: List['Node'] = attr.ib(init=False, factory=list)

    def add_parent(self, parent: 'Node') -> None:
        self.parents.append(parent)

    def name(self) -> str:
        return str(self.block.index)

@attr.s(cmp=False)
class BasicNode(BaseNode):
    successor: 'Node' = attr.ib()

    def is_loop(self) -> bool:
        return is_loop_edge(self, self.successor)

    def __str__(self) -> str:
        return ''.join([
            f'{self.block}\n',
            f'# {self.block.index} -> {self.successor.block.index}',
            " (loop)" if self.is_loop() else ""])


@attr.s(cmp=False)
class ConditionalNode(BaseNode):
    conditional_edge: 'Node' = attr.ib()
    fallthrough_edge: 'Node' = attr.ib()

    def is_loop(self) -> bool:
        return is_loop_edge(self, self.conditional_edge)

    def __str__(self) -> str:
        return ''.join([
            f'{self.block}\n',
            f'# {self.block.index} -> ',
            f'cond: {self.conditional_edge.block.index}',
            ' (loop)' if self.is_loop() else '',
            ', ',
            f'def: {self.fallthrough_edge.block.index}'])


@attr.s(cmp=False)
class ReturnNode(BaseNode):
    index: int = attr.ib()

    def name(self) -> str:
        name = super().name()
        return name if self.is_real() else f'{name}.{self.index}'

    def is_real(self) -> bool:
        return self.index == 0

    def __str__(self) -> str:
        return f'{self.block}\n# {self.block.index} -> ret'

Node = Union[
    BasicNode,
    ConditionalNode,
    ReturnNode,
]

def build_graph_from_block(
    block: Block,
    blocks: List[Block],
    nodes: List[Node]
) -> Node:
    # Don't reanalyze blocks.
    for node in nodes:
        if node.block == block:
            return node

    new_node: Node
    dummy_node: Any = None

    def find_block_by_label(label: JumpTarget) -> Optional[Block]:
        for block in blocks:
            if block.label and block.label.name == label.target:
                return block
        return None

    # Extract branching instructions from this block.
    jumps: List[Instruction] = [
        inst for inst in block.instructions if inst.is_jump_instruction()
    ]
    assert len(jumps) in [0, 1], "too many jump instructions in one block"

    if len(jumps) == 0:
        # No jumps, i.e. the next block is this node's successor block.
        new_node = BasicNode(block, dummy_node)
        nodes.append(new_node)

        # Recursively analyze.
        next_block = blocks[block.index + 1]
        new_node.successor = build_graph_from_block(next_block, blocks, nodes)

        # Keep track of parents.
        new_node.successor.add_parent(new_node)
    elif len(jumps) == 1:
        # There is a jump. This is either a ReturnNode if it's "jr $ra", a
        # BasicNode if it's an unconditional branch, or a ConditionalNode.
        jump = jumps[0]

        if jump.mnemonic == 'jr':
            # If jump.args[0] != Register('ra'), this is a switch, which is not
            # supported. Delay that error until code emission though.
            new_node = ReturnNode(block, index=0)
            nodes.append(new_node)
            return new_node

        assert jump.is_branch_instruction()

        # Get the block associated with the jump target.
        branch_label = jump.get_branch_target()
        branch_block = find_block_by_label(branch_label)
        assert branch_block is not None, \
                f"Cannot find branch target {branch_label.target}"

        is_constant_branch = jump.mnemonic == 'b'
        if is_constant_branch:
            # A constant branch becomes a basic edge to our branch target.
            new_node = BasicNode(block, dummy_node)
            nodes.append(new_node)
            # Recursively analyze.
            new_node.successor = build_graph_from_block(branch_block, blocks,
                    nodes)
            # Keep track of parents.
            new_node.successor.add_parent(new_node)
        else:
            # A conditional branch means the fallthrough block is the next
            # block if the branch isn't.
            new_node = ConditionalNode(block, dummy_node, dummy_node)
            nodes.append(new_node)
            # Recursively analyze this too.
            next_block = blocks[block.index + 1]
            new_node.conditional_edge = build_graph_from_block(branch_block,
                    blocks, nodes)
            new_node.fallthrough_edge = build_graph_from_block(next_block,
                    blocks, nodes)
            # Keep track of parents.
            new_node.conditional_edge.add_parent(new_node)
            new_node.fallthrough_edge.add_parent(new_node)

    return new_node


def is_trivial_return_block(block: Block) -> bool:
    # A heuristic for when a block is a simple "early-return" block.
    # This could be improved.
    stores = ['sb', 'sh', 'sw', 'swc1', 'sdc1', 'swr', 'swl', 'jal']
    return_regs = [Register('v0'), Register('f0')]
    for instr in block.instructions:
        if instr.mnemonic in stores:
            return False
        if any(reg in instr.args for reg in return_regs):
            return False
    return True


def build_nodes(function: Function, blocks: List[Block]) -> List[Node]:
    graph: List[Node] = []

    # Traverse through the block tree.
    entry_block = blocks[0]
    build_graph_from_block(entry_block, blocks, graph)

    # Sort the nodes by index.
    graph.sort(key=lambda node: node.block.index)
    return graph


def is_premature_return(node: Node, edge: Node, nodes: List[Node]) -> bool:
    """Check whether a given edge in the flow graph is an early return."""
    if edge != nodes[-1]:
        return False
    if not is_trivial_return_block(edge.block):
        # Only trivial return blocks can be used for premature returns,
        # hopefully.
        return False
    if not isinstance(node, BasicNode):
        # We only treat BasicNode's as being able to return early right now.
        # (Handling ConditionalNode's seems to cause assertion failures --
        # might need changes to build_flowgraph_between.)
        return False
    # The only node that is allowed to point to the return node is the node
    # before it in the flow graph list. (You'd think it would be the node
    # with index = return_node.index - 1, but that's not necessarily the
    # case -- some functions have a dead penultimate block with a
    # superfluous unreachable return.)
    return node != nodes[-2]


def duplicate_premature_returns(nodes: List[Node]) -> None:
    """For each jump to an early return node, create a duplicate return node
    for it to jump to. This ensures nice nesting for the if_statements code,
    and avoids a phi node for the return value."""
    extra_nodes: List[Node] = []
    index = 0
    for node in nodes:
        if (isinstance(node, BasicNode) and
                is_premature_return(node, node.successor, nodes)):
            assert isinstance(node.successor, ReturnNode)
            node.successor.parents.remove(node)
            index += 1
            n = ReturnNode(node.successor.block.clone(), index=index)
            node.successor = n
            n.add_parent(node)
            extra_nodes.append(n)
    nodes += extra_nodes
    nodes.sort(key=lambda node: node.block.index)


def compute_dominators(nodes: List[Node]) -> None:
    entry = nodes[0]
    entry.dominators = {entry}
    for n in nodes[1:]:
        n.dominators = set(nodes)

    changes = True
    while changes:
        changes = False
        for n in nodes[1:]:
            assert n.parents, f"no predecessors for node: {n}"
            nset = n.dominators
            for p in n.parents:
                nset = nset.intersection(p.dominators)
            nset = {n}.union(nset)
            if len(nset) < len(n.dominators):
                n.dominators = nset
                changes = True

    for n in nodes[1:]:
        doms = n.dominators.difference({n})
        n.immediate_dominator = max(doms, key=lambda d: len(d.dominators))
        n.immediate_dominator.immediately_dominates.append(n)
    for n in nodes:
        n.immediately_dominates.sort(key=lambda x: x.block.index)

@attr.s(frozen=True)
class FlowGraph:
    nodes: List[Node] = attr.ib()

    def entry_node(self) -> Node:
        return self.nodes[0]

    def return_node(self) -> Node:
        candidates = [n for n in self.nodes
                if isinstance(n, ReturnNode) and n.is_real()]
        return max(candidates, key=lambda n: n.block.index)

def build_flowgraph(function: Function) -> FlowGraph:
    blocks = build_blocks(function)
    nodes = build_nodes(function, blocks)
    duplicate_premature_returns(nodes)
    compute_dominators(nodes)
    return FlowGraph(nodes)


def visualize_flowgraph(flow_graph: FlowGraph) -> None:
    import graphviz as g
    dot = g.Digraph()
    for node in flow_graph.nodes:
        dot.node(node.name())
        if isinstance(node, BasicNode):
            dot.edge(node.name(), node.successor.name(), color='green')
        elif isinstance(node, ConditionalNode):
            dot.edge(node.name(), node.fallthrough_edge.name(), color='blue')
            dot.edge(node.name(), node.conditional_edge.name(), color='red')
        else:
            pass
    dot.render('graphviz_render.gv')
    print("Rendered to graphviz_render.gv.pdf")
    print("Key: green = successor, red = conditional edge, blue = fallthrough edge")
