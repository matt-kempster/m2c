import abc
import copy
from dataclasses import dataclass, field
import typing
from typing import (
    Any,
    Callable,
    Counter,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from .error import DecompFailure
from .options import Formatter
from .parse_file import AsmData, Function, Label
from .parse_instruction import (
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmLiteral,
    Instruction,
    InstructionMeta,
    JumpTarget,
    Macro,
    Register,
    parse_instruction,
)


@dataclass(eq=False)
class Block:
    index: int
    label: Optional[Label]
    approx_label_name: str
    instructions: List[Instruction]

    # block_info is actually an Optional[BlockInfo], set by translate.py for
    # non-TerminalNode's, but due to circular dependencies we cannot type it
    # correctly. To access it, use the get_block_info method from translate.py.
    block_info: object = None

    def add_block_info(self, block_info: object) -> None:
        assert self.block_info is None
        self.block_info = block_info

    def clone(self) -> "Block":
        return copy.deepcopy(self)

    def __str__(self) -> str:
        name = f"{self.index} ({self.approx_label_name})"
        inst_str = "\n".join("    " + str(instr) for instr in self.instructions)
        return f"# {name}\n{inst_str}\n"


@dataclass(eq=False)
class BlockBuilder:
    curr_index: int = 0
    curr_label: Optional[Label] = None
    last_label_name: str = "initial"
    label_counter: int = 0
    curr_instructions: List[Instruction] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)

    def new_block(self) -> Optional[Block]:
        if len(self.curr_instructions) == 0:
            return None

        label_name = self.last_label_name
        if self.label_counter > 0:
            label_name += f".{self.label_counter}"
        block = Block(
            self.curr_index, self.curr_label, label_name, self.curr_instructions
        )
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
        if self.curr_label:
            raise DecompFailure(
                "A block is currently not allowed to have more than one label,\n"
                f"but {self.curr_label.name}/{label.name} is given two."
            )
        self.curr_label = label
        self.last_label_name = label.name
        self.label_counter = 0

    def get_blocks(self) -> List[Block]:
        return self.blocks


def verify_no_trailing_delay_slot(function: Function) -> None:
    last_ins: Optional[Instruction] = None
    for item in function.body:
        if isinstance(item, Instruction):
            last_ins = item
    if last_ins and last_ins.is_delay_slot_instruction():
        raise DecompFailure(f"Last instruction is missing a delay slot:\n{last_ins}")


def invert_branch_mnemonic(mnemonic: str) -> str:
    inverses = {
        "beq": "bne",
        "bne": "beq",
        "beqz": "bnez",
        "bnez": "beqz",
        "bgez": "bltz",
        "bgtz": "blez",
        "blez": "bgtz",
        "bltz": "bgez",
        "bc1t": "bc1f",
        "bc1f": "bc1t",
    }
    return inverses[mnemonic]


def normalize_likely_branches(function: Function) -> Function:
    """Branch-likely instructions only evaluate their delay slots when they are
    taken, making control flow more complex. However, on the IDO compiler they
    only occur in a very specific pattern:

    ...
    <branch likely instr> .label
     X
    ...
    X
    .label:
    ...

    which this function transforms back into a regular branch pattern by moving
    the label one step back and replacing the delay slot by a nop.

    Branch-likely instructions that do not appear in this pattern are kept.

    We also do this for b instructions, which sometimes occur in the same pattern,
    and also fix up the pattern

    <branch likely instr> .label
     X
    .label:

    which GCC emits."""
    label_prev_instr: Dict[str, Optional[Instruction]] = {}
    label_before_instr: Dict[int, str] = {}
    instr_before_instr: Dict[int, Instruction] = {}
    prev_instr: Optional[Instruction] = None
    prev_label: Optional[Label] = None
    prev_item: Union[Instruction, Label, None] = None
    for item in function.body:
        if isinstance(item, Instruction):
            if prev_label is not None:
                label_before_instr[id(item)] = prev_label.name
                prev_label = None
            if isinstance(prev_item, Instruction):
                instr_before_instr[id(item)] = prev_item
            prev_instr = item
        elif isinstance(item, Label):
            label_prev_instr[item.name] = prev_instr
            prev_label = item
            prev_instr = None
        prev_item = item

    insert_label_before: Dict[int, str] = {}
    new_body: List[Tuple[Union[Instruction, Label], Union[Instruction, Label]]] = []

    body_iter: Iterator[Union[Instruction, Label]] = iter(function.body)
    for item in body_iter:
        orig_item = item
        if isinstance(item, Instruction) and (
            item.is_branch_likely_instruction() or item.mnemonic == "b"
        ):
            old_label = item.get_branch_target().target
            if old_label not in label_prev_instr:
                raise DecompFailure(
                    f"Unable to parse branch: label {old_label} does not exist in function {function.name}"
                )
            before_target = label_prev_instr[old_label]
            before_before_target = (
                instr_before_instr.get(id(before_target))
                if before_target is not None
                else None
            )
            next_item = next(body_iter)
            orig_next_item = next_item
            if (
                item.mnemonic == "b"
                and before_before_target is not None
                and before_before_target.is_delay_slot_instruction()
            ):
                # Don't treat 'b' instructions as branch likelies if doing so would
                # introduce a label in a delay slot.
                new_body.append((item, item))
                new_body.append((next_item, next_item))
            elif (
                isinstance(next_item, Instruction)
                and before_target is next_item
                and item.mnemonic != "b"
            ):
                mn_inverted = invert_branch_mnemonic(item.mnemonic[:-1])
                item = Instruction.derived(mn_inverted, item.args, item)
                new_nop = Instruction.derived("nop", [], item)
                new_body.append((orig_item, item))
                new_body.append((new_nop, new_nop))
                new_body.append((orig_next_item, next_item))
            elif (
                isinstance(next_item, Instruction)
                and before_target is not None
                and before_target is not next_item
                and str(before_target) == str(next_item)
                and (item.mnemonic != "b" or next_item.mnemonic != "nop")
            ):
                if id(before_target) not in label_before_instr:
                    new_label = old_label + "_before"
                    label_before_instr[id(before_target)] = new_label
                    insert_label_before[id(before_target)] = new_label
                new_target = JumpTarget(label_before_instr[id(before_target)])
                mn_unlikely = item.mnemonic[:-1] or "b"
                item = Instruction.derived(
                    mn_unlikely, item.args[:-1] + [new_target], item
                )
                next_item = Instruction.derived("nop", [], item)
                new_body.append((orig_item, item))
                new_body.append((orig_next_item, next_item))
            else:
                new_body.append((item, item))
                new_body.append((next_item, next_item))
        else:
            new_body.append((orig_item, item))

    new_function = function.bodyless_copy()
    for (orig_item, new_item) in new_body:
        if id(orig_item) in insert_label_before:
            new_function.new_label(insert_label_before[id(orig_item)])
        new_function.body.append(new_item)

    return new_function


def prune_unreferenced_labels(function: Function, asm_data: AsmData) -> Function:
    labels_used: Set[str] = {
        label.name
        for label in function.body
        if isinstance(label, Label) and label.name in asm_data.mentioned_labels
    }
    for item in function.body:
        if isinstance(item, Instruction) and item.is_branch_instruction():
            labels_used.add(item.get_branch_target().target)

    new_function = function.bodyless_copy()
    for item in function.body:
        if not (isinstance(item, Label) and item.name not in labels_used):
            new_function.body.append(item)

    return new_function


def simplify_standard_patterns(function: Function) -> Function:
    """Detect and simplify various standard patterns emitted by IDO and GCC.
    Currently handled:
    - checks for x/0 and INT_MIN/-1 after division (removed)
    - gcc sqrt nan check (removed)
    - division or modulo by power of two (converted to made-up instructions)
    - unsigned to float conversion (converted to a made-up instruction)
    - float/double to unsigned conversion (converted to a made-up instruction)"""
    BodyPart = Union[Instruction, Label]
    PatternPart = Union[Instruction, Label, None]
    Pattern = List[Tuple[PatternPart, bool]]

    def make_pattern(*parts: str) -> Pattern:
        ret: Pattern = []
        for part in parts:
            optional = part.endswith("*")
            part = part.rstrip("*")
            if part == "?":
                ret.append((None, optional))
            elif part.endswith(":"):
                ret.append((Label(""), optional))
            else:
                ins = parse_instruction(part, InstructionMeta.missing())
                ret.append((ins, optional))
        return ret

    div_pattern = make_pattern(
        "bnez $x, .A",
        "?",  # nop or div
        "break",
        ".A:",
        "li $at, -1",
        "bne $x, $at, .B",
        "li $at, 0x80000000",
        "bne $y, $at, .B",
        "nop",
        "break",
        ".B:",
    )

    divu_pattern = make_pattern(
        "bnez $x, .A",
        "nop",
        "break",
        ".A:",
    )

    mod_p2_pattern = make_pattern(
        "bgez $x, .A",
        "andi $y, $x, LIT",
        "beqz $y, .A",
        "nop",
        "addiu $y, $y, LIT",
        ".A:",
    )

    div_p2_pattern_1 = make_pattern(
        "bgez $x, .A",
        "sra $y, $x, LIT",
        "addiu $at, $x, LIT",
        "sra $y, $at, LIT",
        ".A:",
    )

    div_p2_pattern_2 = make_pattern(
        "bgez $x, .A",
        "move $at, $x",
        "addiu $at, $x, LIT",
        ".A:",
        "sra $x, $at, LIT",
    )

    div_2_s16_pattern = make_pattern(
        "sll $x, $x, LIT",
        "sra $y, $x, LIT",
        "srl $x, $x, 0x1f",
        "addu $y, $y, $x",
        "sra $y, $y, 1",
    )

    div_2_s32_pattern = make_pattern(
        "srl $x, $y, 0x1f",
        "addu $x, $y, $x",
        "sra $x, $x, 1",
    )

    utf_pattern = make_pattern(
        "bgez $x, .A",
        "cvt.s.w",
        "li $at, 0x4f800000",
        "mtc1",
        "nop",
        "add.s",
        ".A:",
    )

    ftu_pattern = make_pattern(
        "cfc1 $y, $31",
        "nop",
        "andi",
        "andi*",  # (skippable)
        "?",  # bnez or bneql
        "?",
        "li*",
        "mtc1",
        "mtc1*",
        "li",
        "?",  # sub.fmt ?, X, ?
        "ctc1",
        "nop",
        "?",  # cvt.w.fmt ?, ?
        "cfc1",
        "nop",
        "andi",
        "andi*",
        "bnez",
        "nop",
        "mfc1",
        "li",
        "b",
        "or",
        ".A:",
        "b",
        "li",
        "?",  # label: (moved one step down if bneql)
        "?",  # mfc1
        "nop",
        "bltz",
        "nop",
    )

    lwc1_twice_pattern = make_pattern("lwc1", "lwc1")
    swc1_twice_pattern = make_pattern("swc1", "swc1")

    gcc_sqrt_pattern = make_pattern(
        "sqrt.s",
        "c.eq.s",
        "nop",
        "bc1t",
        "?",
        "jal sqrtf",
        "nop",
    )

    def matches_pattern(actual: List[BodyPart], pattern: Pattern) -> int:
        symbolic_registers: Dict[str, Register] = {}
        symbolic_labels: Dict[str, str] = {}

        def match_one(actual: BodyPart, exp: PatternPart) -> bool:
            if exp is None:
                return True
            if isinstance(exp, Label):
                name = symbolic_labels.get(exp.name)
                return isinstance(actual, Label) and (
                    name is None or actual.name == name
                )
            if not isinstance(actual, Instruction):
                return False
            ins = actual
            if ins.mnemonic != exp.mnemonic:
                return False
            if exp.args:
                if len(exp.args) != len(ins.args):
                    return False
                for (e, a) in zip(exp.args, ins.args):
                    if isinstance(e, AsmLiteral):
                        if not isinstance(a, AsmLiteral) or e.value != a.value:
                            return False
                    elif isinstance(e, Register):
                        if not isinstance(a, Register):
                            return False
                        if len(e.register_name) <= 1:
                            if e.register_name not in symbolic_registers:
                                symbolic_registers[e.register_name] = a
                            elif symbolic_registers[e.register_name] != a:
                                return False
                        elif e.register_name != a.register_name:
                            return False
                    elif isinstance(e, AsmGlobalSymbol):
                        if e.symbol_name == "LIT" and not isinstance(a, AsmLiteral):
                            return False
                    elif isinstance(e, JumpTarget):
                        if not isinstance(a, JumpTarget):
                            return False
                        if e.target not in symbolic_labels:
                            symbolic_labels[e.target] = a.target
                        elif symbolic_labels[e.target] != a.target:
                            return False
                    else:
                        assert False, f"bad pattern part: {exp}"
            return True

        actuali = 0
        for (pat, optional) in pattern:
            if actuali < len(actual) and match_one(actual[actuali], pat):
                actuali += 1
            elif not optional:
                return 0
        return actuali

    def create_div_p2(bgez: Instruction, sra: Instruction) -> Instruction:
        assert isinstance(sra.args[2], AsmLiteral)
        shift = sra.args[2].value & 0x1F
        return Instruction.derived(
            "div.fictive", [sra.args[0], bgez.args[0], AsmLiteral(2 ** shift)], sra
        )

    def try_replace_div(i: int) -> Optional[Tuple[List[BodyPart], int]]:
        actual = function.body[i : i + len(div_pattern)]
        if not matches_pattern(actual, div_pattern):
            return None
        return ([actual[1]], i + len(div_pattern) - 1)

    def try_replace_divu(i: int) -> Optional[Tuple[List[BodyPart], int]]:
        actual = function.body[i : i + len(divu_pattern)]
        if not matches_pattern(actual, divu_pattern):
            return None
        return ([], i + len(divu_pattern) - 1)

    def try_replace_div_p2_1(i: int) -> Optional[Tuple[List[BodyPart], int]]:
        # Division by power of two where input reg != output reg
        actual = function.body[i : i + len(div_p2_pattern_1)]
        if not matches_pattern(actual, div_p2_pattern_1):
            return None
        bnez = typing.cast(Instruction, actual[0])
        div = create_div_p2(bnez, typing.cast(Instruction, actual[3]))
        return ([div], i + len(div_p2_pattern_1) - 1)

    def try_replace_div_p2_2(i: int) -> Optional[Tuple[List[BodyPart], int]]:
        # Division by power of two where input reg = output reg
        actual = function.body[i : i + len(div_p2_pattern_2)]
        if not matches_pattern(actual, div_p2_pattern_2):
            return None
        bnez = typing.cast(Instruction, actual[0])
        div = create_div_p2(bnez, typing.cast(Instruction, actual[4]))
        return ([div], i + len(div_p2_pattern_2))

    def try_replace_div_2_s16(i: int) -> Optional[Tuple[List[BodyPart], int]]:
        actual = function.body[i : i + len(div_2_s16_pattern)]
        if not matches_pattern(actual, div_2_s16_pattern):
            return None
        sll1 = typing.cast(Instruction, actual[0])
        sra1 = typing.cast(Instruction, actual[1])
        sra = typing.cast(Instruction, actual[4])
        if sll1.args[2] != sra1.args[2]:
            return None
        div = Instruction.derived(
            "div.fictive", [sra.args[0], sra.args[0], AsmLiteral(2)], sra
        )
        return ([sll1, sra1, div], i + len(div_2_s16_pattern))

    def try_replace_div_2_s32(i: int) -> Optional[Tuple[List[BodyPart], int]]:
        actual = function.body[i : i + len(div_2_s32_pattern)]
        if not matches_pattern(actual, div_2_s32_pattern):
            return None
        addu = typing.cast(Instruction, actual[1])
        sra = typing.cast(Instruction, actual[2])
        div = Instruction.derived(
            "div.fictive", [sra.args[0], addu.args[1], AsmLiteral(2)], sra
        )
        return ([div], i + len(div_2_s32_pattern))

    def try_replace_mod_p2(i: int) -> Optional[Tuple[List[BodyPart], int]]:
        actual = function.body[i : i + len(mod_p2_pattern)]
        if not matches_pattern(actual, mod_p2_pattern):
            return None
        andi = typing.cast(Instruction, actual[1])
        val = (typing.cast(AsmLiteral, andi.args[2]).value & 0xFFFF) + 1
        mod = Instruction.derived(
            "mod.fictive", [andi.args[0], andi.args[1], AsmLiteral(val)], andi
        )
        return ([mod], i + len(mod_p2_pattern) - 1)

    def try_replace_utf_conv(i: int) -> Optional[Tuple[List[BodyPart], int]]:
        actual = function.body[i : i + len(utf_pattern)]
        if not matches_pattern(actual, utf_pattern):
            return None
        cvt_instr = typing.cast(Instruction, actual[1])
        new_instr = Instruction.derived("cvt.s.u.fictive", cvt_instr.args, cvt_instr)
        return ([new_instr], i + len(utf_pattern) - 1)

    def try_replace_ftu_conv(i: int) -> Optional[Tuple[List[BodyPart], int]]:
        actual = function.body[i : i + len(ftu_pattern)]
        consumed = matches_pattern(actual, ftu_pattern)
        if not consumed:
            return None
        sub = next(
            x
            for x in actual
            if isinstance(x, Instruction) and x.mnemonic.startswith("sub")
        )
        cfc = actual[0]
        assert isinstance(cfc, Instruction)
        fmt = sub.mnemonic.split(".")[-1]
        args = [cfc.args[0], sub.args[1]]
        if fmt == "s":
            new_instr = Instruction.derived("cvt.u.s.fictive", args, cfc)
        else:
            new_instr = Instruction.derived("cvt.u.d.fictive", args, cfc)
        return ([new_instr], i + consumed)

    def try_replace_mips1_double_load_store(
        i: int,
    ) -> Optional[Tuple[List[BodyPart], int]]:
        # TODO: sometimes the instructions aren't consecutive.
        actual = function.body[i : i + 2]
        if not matches_pattern(actual, lwc1_twice_pattern) and not matches_pattern(
            actual, swc1_twice_pattern
        ):
            return None
        a, b = actual
        assert isinstance(a, Instruction)
        assert isinstance(b, Instruction)
        ra, rb = a.args[0], b.args[0]
        ma, mb = a.args[1], b.args[1]
        # TODO: verify that the memory locations are consecutive as well (a bit
        # annoying with macros...)
        if not (
            isinstance(ra, Register)
            and ra.is_float()
            and ra.other_f64_reg() == rb
            and isinstance(ma, AsmAddressMode)
            and isinstance(mb, AsmAddressMode)
            and ma.rhs == mb.rhs
        ):
            return None
        num = int(ra.register_name[1:])
        if num % 2 == 1:
            ra, rb = rb, ra
            ma, mb = mb, ma
        # Store the even-numbered register (ra) into the low address (mb).
        new_args = [ra, mb]
        new_mn = "ldc1" if a.mnemonic == "lwc1" else "sdc1"
        new_instr = Instruction.derived(new_mn, new_args, a)
        return ([new_instr], i + 2)

    def try_replace_gcc_sqrt(i: int) -> Optional[Tuple[List[BodyPart], int]]:
        actual = function.body[i : i + len(gcc_sqrt_pattern)]
        consumed = matches_pattern(actual, gcc_sqrt_pattern)
        if not consumed:
            return None
        sqrt = actual[0]
        assert isinstance(sqrt, Instruction)
        new_instr = Instruction.derived("sqrt.s", sqrt.args, sqrt)
        return ([new_instr], i + consumed)

    def no_replacement(i: int) -> Tuple[List[BodyPart], int]:
        return ([function.body[i]], i + 1)

    new_function = function.bodyless_copy()
    i = 0
    while i < len(function.body):
        repl, i = (
            try_replace_div(i)
            or try_replace_divu(i)
            or try_replace_div_p2_1(i)
            or try_replace_div_p2_2(i)
            or try_replace_div_2_s32(i)
            or try_replace_div_2_s16(i)
            or try_replace_mod_p2(i)
            or try_replace_utf_conv(i)
            or try_replace_ftu_conv(i)
            or try_replace_mips1_double_load_store(i)
            or try_replace_gcc_sqrt(i)
            or no_replacement(i)
        )
        new_function.body.extend(repl)
    return new_function


def build_blocks(function: Function, asm_data: AsmData) -> List[Block]:
    verify_no_trailing_delay_slot(function)
    function = normalize_likely_branches(function)
    function = prune_unreferenced_labels(function, asm_data)
    function = simplify_standard_patterns(function)
    function = prune_unreferenced_labels(function, asm_data)

    block_builder = BlockBuilder()

    body_iter: Iterator[Union[Instruction, Label]] = iter(function.body)
    branch_likely_counts: Counter[str] = Counter()

    def process(item: Union[Instruction, Label]) -> None:
        if isinstance(item, Label):
            # Split blocks at labels.
            block_builder.new_block()
            block_builder.set_label(item)
            return

        if not item.is_delay_slot_instruction():
            block_builder.add_instruction(item)
            return

        process_after: List[Union[Instruction, Label]] = []
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
                    "to all jump sources and move the label, or add a nop to the delay slot).",
                ]
                if "_before" in label.name:
                    msg += [
                        "",
                        "This label was auto-generated as the target for a branch-likely",
                        "instruction (e.g. beql); you can also try to turn that into a non-likely",
                        "branch if that's equivalent in this context, i.e., if it's okay to",
                        "execute its delay slot unconditionally.",
                    ]
                raise DecompFailure("\n".join(msg))

        if next_item.is_delay_slot_instruction():
            raise DecompFailure(
                f"Two delay slot instructions in a row is not supported:\n{item}\n{next_item}"
            )

        if item.is_branch_likely_instruction():
            target = item.get_branch_target()
            branch_likely_counts[target.target] += 1
            index = branch_likely_counts[target.target]
            mn_inverted = invert_branch_mnemonic(item.mnemonic[:-1])
            temp_label = JumpTarget(f"{target.target}_branchlikelyskip_{index}")
            branch_not = Instruction.derived(
                mn_inverted, item.args[:-1] + [temp_label], item
            )
            nop = Instruction.derived("nop", [], item)
            block_builder.add_instruction(branch_not)
            block_builder.add_instruction(nop)
            block_builder.new_block()
            block_builder.add_instruction(next_item)
            block_builder.add_instruction(Instruction.derived("b", [target], item))
            block_builder.add_instruction(nop)
            block_builder.new_block()
            block_builder.set_label(Label(temp_label.target))
            block_builder.add_instruction(nop)

        elif item.mnemonic in ["jal", "jalr"]:
            # Move the delay slot instruction to before the call so it
            # passes correct arguments.
            if next_item.args and next_item.args[0] == item.args[0]:
                raise DecompFailure(
                    f"Instruction after {item.mnemonic} clobbers its source\n"
                    "register, which is currently not supported.\n\n"
                    "Try rewriting the assembly to avoid that."
                )
            block_builder.add_instruction(next_item)
            block_builder.add_instruction(item)
        else:
            block_builder.add_instruction(item)
            block_builder.add_instruction(next_item)

        if item.is_jump_instruction():
            # Split blocks at jumps, after the next instruction.
            block_builder.new_block()

        for item in process_after:
            process(item)

    for item in body_iter:
        process(item)

    if block_builder.curr_label:
        # As an easy-to-implement safeguard, check that the current block is
        # anonymous ("jr" instructions create new anonymous blocks, so if it's
        # not we must be missing a "jr $ra").
        label = block_builder.curr_label.name
        print(f'Warning: missing "jr $ra" in last block (.{label}).\n')
        meta = InstructionMeta.missing()
        block_builder.add_instruction(Instruction("jr", [Register("ra")], meta))
        block_builder.add_instruction(Instruction("nop", [], meta))
        block_builder.new_block()

    # Throw away whatever is past the last "jr $ra" and return what we have.
    return block_builder.get_blocks()


# Split out dataclass from abc due to a mypy limitation:
# https://github.com/python/mypy/issues/5374
@dataclass(eq=False)
class _BaseNode:
    block: Block
    emit_goto: bool
    parents: List["Node"] = field(init=False, default_factory=list)
    dominators: Set["Node"] = field(init=False, default_factory=set)
    immediate_dominator: Optional["Node"] = field(init=False, default=None)
    immediately_dominates: List["Node"] = field(init=False, default_factory=list)
    postdominators: Set["Node"] = field(init=False, default_factory=set)
    immediate_postdominator: Optional["Node"] = field(init=False, default=None)
    immediately_postdominates: List["Node"] = field(init=False, default_factory=list)
    # This is only populated on the head node of the loop,
    # i.e. there is an invariant `(node.loop is None) or (node.loop.head is node)`
    loop: Optional["NaturalLoop"] = field(init=False, default=None)


class BaseNode(_BaseNode, abc.ABC):
    def name(self) -> str:
        return str(self.block.index)

    @abc.abstractmethod
    def children(self) -> List["Node"]:
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.block.index}>"


@dataclass(eq=False, repr=False)
class BasicNode(BaseNode):
    successor: "Node"
    # Optional link to the TerminalNode, used to make infinite loops reducible.
    # Used by `compute_relations()`, but ignored when emitting code.
    fake_successor: Optional["TerminalNode"] = None

    def children(self) -> List["Node"]:
        # TODO: Should we also include the fallthrough node if `emit_goto` is True?
        if self.fake_successor is not None:
            return [self.successor, self.fake_successor]
        return [self.successor]

    def __str__(self) -> str:
        return "".join(
            [
                f"{self.block}\n",
                f"# {self.block.index} -> {self.successor.block.index}",
                " (loop)" if self.loop else "",
                " (goto)" if self.emit_goto else "",
            ]
        )


@dataclass(eq=False, repr=False)
class ConditionalNode(BaseNode):
    conditional_edge: "Node"
    fallthrough_edge: "Node"

    def children(self) -> List["Node"]:
        if self.conditional_edge == self.fallthrough_edge:
            return [self.conditional_edge]
        return [self.conditional_edge, self.fallthrough_edge]

    def __str__(self) -> str:
        return "".join(
            [
                f"{self.block}\n",
                f"# {self.block.index} -> ",
                f"cond: {self.conditional_edge.block.index}",
                " (loop)" if self.loop else "",
                ", ",
                f"def: {self.fallthrough_edge.block.index}",
                " (goto)" if self.emit_goto else "",
            ]
        )


@dataclass(eq=False, repr=False)
class ReturnNode(BaseNode):
    index: int
    terminal: "TerminalNode"

    def children(self) -> List["Node"]:
        return [self.terminal]

    def name(self) -> str:
        name = super().name()
        return name if self.is_real() else f"{name}.{self.index}"

    def is_real(self) -> bool:
        return self.index == 0

    def __str__(self) -> str:
        return "".join(
            [
                f"{self.block}\n",
                f"# {self.block.index} -> ret",
                " (goto)" if self.emit_goto else "",
            ]
        )


@dataclass(eq=False, repr=False)
class SwitchNode(BaseNode):
    cases: List["Node"]

    def children(self) -> List["Node"]:
        # Deduplicate nodes in `self.cases`
        seen = set()
        children = []
        for node in self.cases:
            if node not in seen:
                seen.add(node)
                children.append(node)
        return children

    def __str__(self) -> str:
        targets = ", ".join(str(c.block.index) for c in self.cases)
        return f"{self.block}\n# {self.block.index} -> {targets}"


@dataclass(eq=False, repr=False)
class TerminalNode(BaseNode):
    """
    This is a fictive node that acts as a common postdominator for every node
    in the flowgraph. There should be exactly one TerminalNode per flowgraph.
    This is sometimes referred to as the "exit node", and having it makes it
    easier to perform interval/structural analysis on the flowgraph.
    This node has no `block_info` or contents: it has no corresponding assembly
    and emits no C code.
    """

    @staticmethod
    def terminal() -> "TerminalNode":
        return TerminalNode(Block(-1, None, "", []), False)

    def children(self) -> List["Node"]:
        return []

    def __str__(self) -> str:
        return "return"


Node = Union[BasicNode, ConditionalNode, ReturnNode, SwitchNode, TerminalNode]


@dataclass(eq=False)
class NaturalLoop:
    """
    Loops are defined by "backedges" `tail->head` where `head`
    dominates `tail` in the control flow graph.

    The loop body is the set of nodes dominated by `head` where there
    is a path from the node to `tail` that does not contain `head`.

    A loop can contain multiple backedges that all point to `head`
    Each backedge is represented by the source node.
    """

    head: Node
    nodes: Set[Node] = field(default_factory=set)
    backedges: Set[Node] = field(default_factory=set)


def build_graph_from_block(
    block: Block, blocks: List[Block], nodes: List[Node], asm_data: AsmData
) -> Node:
    # Don't reanalyze blocks.
    for node in nodes:
        if node.block == block:
            return node

    new_node: Node
    dummy_node: Any = None
    terminal_node = nodes[0]
    assert isinstance(
        terminal_node, TerminalNode
    ), "expected first node to be TerminalNode"

    def find_block_by_label(label: str) -> Optional[Block]:
        for block in blocks:
            if block.label and block.label.name == label:
                return block
        return None

    # Extract branching instructions from this block.
    jumps: List[Instruction] = [
        inst for inst in block.instructions if inst.is_jump_instruction()
    ]
    assert len(jumps) in [0, 1], "too many jump instructions in one block"

    if len(jumps) == 0:
        # No jumps, i.e. the next block is this node's successor block.
        new_node = BasicNode(block, False, dummy_node)
        nodes.append(new_node)

        # Recursively analyze.
        next_block = blocks[block.index + 1]
        new_node.successor = build_graph_from_block(next_block, blocks, nodes, asm_data)
    elif len(jumps) == 1:
        # There is a jump. This is either:
        # - a ReturnNode, if it's "jr $ra",
        # - a SwitchNode, if it's "jr $something_else",
        # - a BasicNode, if it's an unconditional branch, or
        # - a ConditionalNode.
        jump = jumps[0]

        if jump.mnemonic == "jr" and jump.args[0] == Register("ra"):
            new_node = ReturnNode(block, False, index=0, terminal=terminal_node)
            nodes.append(new_node)
            return new_node

        if jump.mnemonic == "jr":
            new_node = SwitchNode(block, False, [])
            nodes.append(new_node)

            jtbl_names = []
            for ins in block.instructions:
                for arg in ins.args:
                    if (
                        isinstance(arg, AsmAddressMode)
                        and isinstance(arg.lhs, Macro)
                        and arg.lhs.macro_name == "lo"
                        and isinstance(arg.lhs.argument, AsmGlobalSymbol)
                        and any(
                            arg.lhs.argument.symbol_name.startswith(prefix)
                            for prefix in ("jtbl", "jpt_")
                        )
                    ):
                        jtbl_names.append(arg.lhs.argument.symbol_name)
            if len(jtbl_names) != 1:
                raise DecompFailure(
                    f"Unable to determine jump table for jr instruction {jump.meta.loc_str()}.\n\n"
                    "There must be a read of a variable in the same block as\n"
                    'the instruction, which has a name starting with "jtbl"/"jpt_".'
                )

            jtbl_name = jtbl_names[0]
            if jtbl_name not in asm_data.values:
                raise DecompFailure(
                    f"Found jr instruction {jump.meta.loc_str()}, but the "
                    "corresponding jump table is not provided.\n"
                    "\n"
                    "Please include it in the input .s file(s), or in an additional file.\n"
                    'It needs to be within ".section .rodata" or ".section .late_rodata".\n'
                    "\n"
                    "(You might need to pass --goto and --no-andor flags as well, "
                    "to get correct control flow for non-jtbl switch jumps.)"
                )

            jtbl_value = asm_data.values[jtbl_name]
            jtbl_value.is_jtbl = True
            for entry in jtbl_value.data:
                if isinstance(entry, bytes):
                    # We have entered padding, stop reading.
                    break
                entry = entry.lstrip(".")
                case_block = find_block_by_label(entry)
                if case_block is None:
                    raise DecompFailure(f"Cannot find jtbl target {entry}")
                case_node = build_graph_from_block(case_block, blocks, nodes, asm_data)
                new_node.cases.append(case_node)
            return new_node

        assert jump.is_branch_instruction()

        # Get the block associated with the jump target.
        branch_label = jump.get_branch_target()
        branch_block = find_block_by_label(branch_label.target)
        if branch_block is None:
            target = branch_label.target
            raise DecompFailure(f"Cannot find branch target {target}")

        is_constant_branch = jump.mnemonic in ["b", "j"]
        emit_goto = jump.meta.emit_goto
        if is_constant_branch:
            # A constant branch becomes a basic edge to our branch target.
            new_node = BasicNode(block, emit_goto, dummy_node)
            nodes.append(new_node)
            # Recursively analyze.
            new_node.successor = build_graph_from_block(
                branch_block, blocks, nodes, asm_data
            )
        else:
            # A conditional branch means the fallthrough block is the next
            # block if the branch isn't.
            new_node = ConditionalNode(block, emit_goto, dummy_node, dummy_node)
            nodes.append(new_node)
            # Recursively analyze this too.
            next_block = blocks[block.index + 1]
            new_node.conditional_edge = build_graph_from_block(
                branch_block, blocks, nodes, asm_data
            )
            new_node.fallthrough_edge = build_graph_from_block(
                next_block, blocks, nodes, asm_data
            )
    return new_node


def is_trivial_return_block(block: Block) -> bool:
    # A heuristic for when a block is a simple "early-return" block.
    # This could be improved.
    stores = ["sb", "sh", "sw", "swc1", "sdc1", "swr", "swl", "jal"]
    return_regs = [Register("v0"), Register("f0")]
    for instr in block.instructions:
        if instr.mnemonic in stores:
            return False
        if any(reg in instr.args for reg in return_regs):
            return False
    return True


def reachable_without(start: Node, end: Node, without: Node) -> bool:
    """Return whether `end` is reachable from `start` with `without` removed."""
    reachable: Set[Node] = set()
    stack: List[Node] = [start]

    while stack:
        node = stack.pop()
        if node == without or node in reachable:
            continue
        reachable.add(node)
        stack.extend(node.children())

    return end in reachable


def build_nodes(
    function: Function, blocks: List[Block], asm_data: AsmData
) -> List[Node]:
    terminal_node = TerminalNode.terminal()
    graph: List[Node] = [terminal_node]

    if not blocks:
        raise DecompFailure(
            f"Function {function.name} contains no instructions. Maybe it is rodata?"
        )

    # Traverse through the block tree.
    entry_block = blocks[0]
    build_graph_from_block(entry_block, blocks, graph, asm_data)

    # Give the TerminalNode a new index so that it sorts to the end of the list
    assert [n for n in graph if isinstance(n, TerminalNode)] == [terminal_node]
    terminal_node.block.index = max(n.block.index for n in graph) + 1

    # Sort the nodes by index.
    graph.sort(key=lambda node: node.block.index)
    return graph


def is_premature_return(node: Node, edge: Node, nodes: List[Node]) -> bool:
    """Check whether a given edge in the flow graph is an early return."""
    if len(nodes) < 3:
        return False
    antepenultimate_node, final_return_node, terminal_node = nodes[-3:]
    assert isinstance(terminal_node, TerminalNode)
    if not isinstance(edge, ReturnNode) or edge != final_return_node:
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
    # case -- some functions have a dead antepenultimate block with a
    # superfluous unreachable return.)
    return node != antepenultimate_node


def duplicate_premature_returns(nodes: List[Node]) -> List[Node]:
    """For each jump to an early return node, create a duplicate return node
    for it to jump to. This ensures nice nesting for the if_statements code,
    and avoids a phi node for the return value."""
    extra_nodes: List[Node] = []
    index = 0
    for node in nodes:
        if (
            isinstance(node, BasicNode)
            and not node.emit_goto
            and is_premature_return(node, node.successor, nodes)
        ):
            assert isinstance(node.successor, ReturnNode)
            index += 1
            n = ReturnNode(
                node.successor.block.clone(),
                False,
                index=index,
                terminal=node.successor.terminal,
            )
            node.successor = n
            extra_nodes.append(n)

    nodes += extra_nodes
    nodes.sort(key=lambda node: node.block.index)

    # Filter nodes to only include ones reachable from the entry node
    queue = {nodes[0]}
    # Always include the TerminalNode (even if it isn't reachable right now)
    reachable_nodes: Set[Node] = {n for n in nodes if isinstance(n, TerminalNode)}
    while queue:
        node = queue.pop()
        reachable_nodes.add(node)
        queue.update(set(node.children()) - reachable_nodes)
    return [n for n in nodes if n in reachable_nodes]


def compute_relations(nodes: List[Node]) -> None:
    """
    Compute dominators, postdominators, and backedges for the set of nodes.
    These are properties of control-flow graphs, for example see:
    https://en.wikipedia.org/wiki/Control-flow_graph
    https://www.cs.princeton.edu/courses/archive/spr04/cos598C/lectures/02-ControlFlow.pdf
    """
    # Calculate parents
    for node in nodes:
        node.parents = []
    for node in nodes:
        for child in node.children():
            child.parents.append(node)

    def compute_dominators(
        entry: Node,
        parents: Callable[[Node], List[Node]],
        dominators: Callable[[Node], Set[Node]],
        immediately_dominates: Callable[[Node], List[Node]],
        set_immediate_dominator: Callable[[Node, Optional[Node]], None],
    ) -> None:
        # See https://en.wikipedia.org/wiki/Dominator_(graph_theory)#Algorithms
        # Note: if `n` is unreachable from `entry`, then *every* node will
        # vacuously belong to `n`'s dominator set.
        for n in nodes:
            dominators(n).clear()
            if n == entry:
                dominators(n).add(n)
            else:
                dominators(n).update(nodes)

        changes = True
        while changes:
            changes = False
            for node in nodes:
                if node == entry:
                    continue
                nset = dominators(node)
                for parent in parents(node):
                    nset = nset.intersection(dominators(parent))
                nset.add(node)
                if len(nset) < len(dominators(node)):
                    assert nset.issubset(dominators(node))
                    dominators(node).intersection_update(nset)
                    changes = True

        # Compute immediate dominator, and the inverse relation
        for node in nodes:
            immediately_dominates(node).clear()
        for node in nodes:
            doms = dominators(node).difference({node})
            # If `node == entry` or the flow graph is not reducible, `doms` may be empty.
            # TODO: Infinite loops could be made reducible by introducing
            # branches like `if (false) { return; }` without breaking semantics
            if doms:
                # There should be a unique max `len(dominators(d))` if the flowgraph
                # is reducible. Fall back to largest index for irreducible graphs.
                imdom = max(doms, key=lambda d: (len(dominators(d)), d.block.index))
                immediately_dominates(imdom).append(node)
                set_immediate_dominator(node, imdom)
            else:
                set_immediate_dominator(node, None)
        for node in nodes:
            immediately_dominates(node).sort(key=lambda x: x.block.index)

    def _set_immediate_dominator(node: Node, imdom: Optional[Node]) -> None:
        node.immediate_dominator = imdom

    def _set_immediate_postdominator(node: Node, impdom: Optional[Node]) -> None:
        node.immediate_postdominator = impdom

    entry = nodes[0]
    terminal = nodes[-1]
    assert isinstance(terminal, TerminalNode)

    # Compute dominators & immediate dominators
    compute_dominators(
        entry=entry,
        parents=lambda n: n.parents,
        dominators=lambda n: n.dominators,
        immediately_dominates=lambda n: n.immediately_dominates,
        set_immediate_dominator=_set_immediate_dominator,
    )

    # Compute postdominators & immediate postdominators
    # This uses the same algorithm as above, but with edges reversed
    compute_dominators(
        entry=terminal,
        parents=lambda n: n.children(),
        dominators=lambda n: n.postdominators,
        immediately_dominates=lambda n: n.immediately_postdominates,
        set_immediate_dominator=_set_immediate_postdominator,
    )

    # Iterate over all edges n -> c and check for backedges, which define natural loops
    for node in nodes:
        for child in node.children():
            if child not in node.dominators:
                continue
            # Found a backedge node -> child where child dominates node; child is the "head" of the loop
            if child.loop is None:
                child.loop = NaturalLoop(child)
            child.loop.nodes |= {child, node}
            child.loop.backedges.add(node)
            for parent in nodes:
                if reachable_without(parent, node, child):
                    child.loop.nodes.add(parent)


def terminate_infinite_loops(nodes: List[Node]) -> None:
    terminal_node = nodes[-1]
    assert isinstance(terminal_node, TerminalNode)

    # Fixed-point iteration
    while True:
        # Find the set of nodes which are rechable from the TerminalNode, backwards
        seen = set()
        queue: Set[Node] = {terminal_node}
        while queue:
            n = queue.pop()
            seen.add(n)
            queue.update(set(n.parents) - seen)

        # Find the BasicNodes which cannot reach the TerminalNode
        # Although it's possible to design asm which has an infinite loop without
        # BasicNodes, it doesn't seem to happen in practice. Keeping these fixups
        # contained keeps the `build_flowgraph_between(...)` logic simpler.
        unterminated_nodes = [
            node for node in nodes if node not in seen and isinstance(node, BasicNode)
        ]
        if not unterminated_nodes:
            # Nothing left to do!
            return

        # Add a "fake" edge from the final BasicNode to the TerminalNode,
        # which is used by `compute_relations()` but is ignored in codegen.
        unterminated_nodes[-1].fake_successor = terminal_node
        compute_relations(nodes)


@dataclass(frozen=True)
class FlowGraph:
    nodes: List[Node]

    def entry_node(self) -> Node:
        return self.nodes[0]

    def terminal_node(self) -> TerminalNode:
        node = self.nodes[-1]
        assert isinstance(node, TerminalNode)
        return node

    def is_reducible(self) -> bool:
        """
        A control flow graph is considered reducible or "well-structured" if:
        1. The edges can be partitioned into two disjoint sets: forward edges
          and back edges
        2. The forward edges form a DAG, with all nodes reachable from the entry
        3. For all back edges A -> B, B dominates A

        We already calculated back edges in `compute_relations`, so we need to
        check that the non-back edges satisfy #2.

        Most C code produces reducible graphs: to produce non-reducible graphs,
        the code either needs a `goto` or an infinite loop, which is rare in
        practice.

        Having a reducible control flow graphs allows us to perform interval analysis:
        we can examine a section of the graph bounded by entry & exit nodes, where
        the exit node postdominates the entry node. In a reducible graph, the backedges
        in loops are unambiguous.

        https://en.wikipedia.org/wiki/Control-flow_graph#Reducibility
        https://www.cs.princeton.edu/courses/archive/spr04/cos598C/lectures/02-ControlFlow.pdf
        https://www.cs.columbia.edu/~suman/secure_sw_devel/Basic_Program_Analysis_CF.pdf
        """

        # Kahn's Algorithm, with backedges excluded
        # https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
        seen = set()
        queue = {self.entry_node()}
        incoming_edges: Dict[Node, Set[Node]] = {
            n: set(n.parents) - (n.loop.backedges if n.loop else set())
            for n in self.nodes
        }

        while queue:
            n = queue.pop()
            seen.add(n)
            for s in n.children():
                if s.loop is not None and n in s.loop.backedges:
                    continue
                incoming_edges[s].remove(n)
                if not incoming_edges[s]:
                    queue.add(s)

        if any(v for v in incoming_edges.values()):
            # There are remaining edges: the graph has at least one cycle
            return False
        if len(seen) != len(self.nodes):
            # Not all nodes are reachable from the entry node
            return False

        # Traverse the graph again, but starting at the terminal and following
        # all edges backwards, to ensure that it is possible to reach the exit
        # from every node. We don't need to look for loops this time.
        seen = set()
        queue = {self.terminal_node()}
        while queue:
            n = queue.pop()
            seen.add(n)
            queue.update(set(n.parents) - seen)

        if len(seen) != len(self.nodes):
            # Not all nodes can reach the terminal node
            return False
        return True


def build_flowgraph(function: Function, asm_data: AsmData) -> FlowGraph:
    blocks = build_blocks(function, asm_data)
    nodes = build_nodes(function, blocks, asm_data)
    nodes = duplicate_premature_returns(nodes)
    compute_relations(nodes)
    terminate_infinite_loops(nodes)
    return FlowGraph(nodes)


def visualize_flowgraph(flow_graph: FlowGraph) -> str:
    import graphviz as g

    fmt = Formatter(debug=True)
    dot = g.Digraph(
        node_attr={
            "shape": "rect",
            "fontname": "Monospace",
        },
        edge_attr={
            "fontname": "Monospace",
        },
    )
    for node in flow_graph.nodes:
        block_info: Optional[Any] = node.block.block_info
        # In Graphviz, "\l" makes the preceeding text left-aligned, and inserts a newline
        label = f"// Node {node.name()}\l"
        if block_info:
            label += "".join(
                w.format(fmt) + "\l" for w in block_info.to_write if w.should_write()
            )
        dot.node(node.name())
        if isinstance(node, BasicNode):
            dot.edge(node.name(), node.successor.name(), color="black")
        elif isinstance(node, ConditionalNode):
            if block_info:
                label += f"if ({block_info.branch_condition.format(fmt)})\l"
            dot.edge(node.name(), node.fallthrough_edge.name(), label="F", color="blue")
            dot.edge(node.name(), node.conditional_edge.name(), label="T", color="red")
        elif isinstance(node, ReturnNode):
            if block_info and block_info.return_value:
                label += f"return ({block_info.return_value.format(fmt)});\l"
            else:
                label += "return;\l"
            dot.edge(node.name(), node.terminal.name())
        elif isinstance(node, SwitchNode):
            assert block_info is not None
            switch_control = block_info.switch_control
            label += f"switch ({switch_control.control_expr.format(fmt)})\l"
            for i, case in enumerate(node.cases):
                dot.edge(
                    node.name(),
                    case.name(),
                    label=str(i + switch_control.offset),
                    color="green",
                )
        else:
            assert isinstance(node, TerminalNode)
            label += "// exit\l"
        dot.node(node.name(), label=label)
    svg_bytes: bytes = dot.pipe("svg")
    return svg_bytes.decode("utf-8", "replace")
