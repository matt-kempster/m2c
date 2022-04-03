import abc
from collections import defaultdict
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Type, TypeVar

from .error import static_assert_unreachable
from .flow_graph import (
    ArchFlowGraph,
    BaseNode,
    FlowGraph,
    InstrRef,
    Instruction,
    LocationRefSetDict,
    RefSet,
    Reference,
    TerminalNode,
    build_flowgraph,
)
from .parse_file import AsmData, Function
from .parse_instruction import (
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmInstruction,
    AsmLiteral,
    BinOp,
    Instruction,
    InstructionMeta,
    JumpTarget,
    Location,
    Macro,
    RegFormatter,
    Register,
    StackLocation,
    parse_instruction,
)


@dataclass(eq=False, frozen=True)
class IrPattern(abc.ABC):
    """
    Template for defining "IR" patterns that can match against input asm.
    The matching process uses the FlowGraph and register analysis to compute
    inter-instruction dependencies, so these patterns can match even when
    they have been interleaved/reordered by the compiler in the input asm.

    IrPattern subclasses *must* define `parts` and `replacement`, and can
    optionally implement `check()`.

    For now, the pattern cannot contain any branches, and the replacement
    must be a single instruction (though, it can be fictive).
    """

    parts: ClassVar[List[str]]
    replacement: ClassVar[str]

    flow_graph: FlowGraph
    replacement_instr: Instruction

    def check(self, m: "IrMatch") -> bool:
        """Override to perform additional checks/calculations before replacement."""
        return True

    @classmethod
    def compile(cls, arch: ArchFlowGraph) -> "IrPattern":
        missing_meta = InstructionMeta.missing()
        regf = RegFormatter()
        replacement_instr = parse_instruction(cls.replacement, missing_meta, arch, regf)

        name = f"__pattern_{cls.__name__}"
        func = Function(name=name, arguments=[])
        # Add a fictive nop instruction for each input to the replacement_instr
        # This acts as a placeholder Reference to represent where the input was set
        for inp in replacement_instr.inputs:
            func.new_instruction(
                Instruction(
                    "in.fictive",
                    [],
                    meta=missing_meta,
                    inputs=[],
                    clobbers=[],
                    outputs=[inp],
                    reads_memory=False,
                    writes_memory=False,
                )
            )
        for part in cls.parts:
            func.new_instruction(parse_instruction(part, missing_meta, arch, regf))
        # Add a fictive nop instruction for each output from the replacement_instr
        for out in replacement_instr.outputs:
            func.new_instruction(
                Instruction(
                    "out.fictive",
                    [],
                    meta=missing_meta,
                    inputs=[out],
                    clobbers=[],
                    outputs=[],
                    reads_memory=False,
                    writes_memory=False,
                )
            )

        asm_data = AsmData()
        flow_graph = build_flowgraph(func, asm_data, arch, fragment=True)
        return cls(
            flow_graph=flow_graph,
            replacement_instr=replacement_instr,
        )


@dataclass
class IrMatch:
    """
    IrMatch represents the matched state of an IrPattern.
    This object is considered read-only; none of its methods modify its state.
    Its `map_*` methods take a pattern part and return the matched instruction part.
    """

    arch: ArchFlowGraph
    symbolic_registers: Dict[str, Register] = field(default_factory=dict)
    symbolic_labels: Dict[str, str] = field(default_factory=dict)
    symbolic_args: Dict[str, Argument] = field(default_factory=dict)
    ref_map: Dict[Reference, Reference] = field(default_factory=dict)

    def eval_math(self, pat: Argument) -> int:
        # This function can only evaluate math in *patterns*, not candidate
        # instructions. It does not need to support arbitrary math, only
        # math used by IR patterns.
        if isinstance(pat, AsmLiteral):
            return pat.value
        if isinstance(pat, BinOp):
            if pat.op == "+":
                return self.eval_math(pat.lhs) + self.eval_math(pat.rhs)
            if pat.op == "-":
                return self.eval_math(pat.lhs) - self.eval_math(pat.rhs)
            if pat.op == "<<":
                return self.eval_math(pat.lhs) << self.eval_math(pat.rhs)
            assert False, f"bad pattern binop: {pat}"
        elif isinstance(pat, AsmGlobalSymbol):
            assert (
                pat.symbol_name in self.symbolic_args
            ), f"undefined variable in math pattern: {pat.symbol_name}"
            lit = self.symbolic_args[pat.symbol_name]
            assert isinstance(lit, AsmLiteral)
            return lit.value
        else:
            assert False, f"bad pattern expr: {pat}"

    def map_reg(self, key: Register) -> Register:
        if len(key.register_name) <= 1:
            return self.symbolic_registers[key.register_name]
        return key

    def map_arg(self, key: Argument) -> Argument:
        if isinstance(key, AsmLiteral):
            return key
        if isinstance(key, Register):
            return self.map_reg(key)
        if isinstance(key, AsmGlobalSymbol):
            if key.symbol_name.isupper():
                return self.symbolic_args[key.symbol_name]
            return key
        if isinstance(key, AsmAddressMode):
            rhs = self.map_arg(key.rhs)
            assert isinstance(rhs, Register)
            return AsmAddressMode(lhs=self.map_arg(key.lhs), rhs=rhs)
        if isinstance(key, JumpTarget):
            return JumpTarget(self.symbolic_labels[key.target])
        if isinstance(key, BinOp):
            return AsmLiteral(self.eval_math(key))
        assert False, f"bad pattern part: {key}"

    def map_ref(self, key: InstrRef) -> InstrRef:
        value = self.ref_map[key]
        assert isinstance(value, InstrRef)
        return value

    def map_location(self, key: Location) -> Location:
        if isinstance(key, Register):
            return self.map_reg(key)
        if isinstance(key, StackLocation):
            loc = StackLocation.from_offset(self.map_arg(key.offset_as_arg()), key.size)
            assert loc is not None
            return loc
        static_assert_unreachable(key)


class TryIrMatch(IrMatch):
    """
    TryIrMatch represents the partial (in-progress) match state of an IrPattern.
    Unlike IrMatch, all of its `match_*` methods may modify its internal state.
    These all take a pair of arguments: pattern part, and candidate part.
    """

    K = TypeVar("K")
    V = TypeVar("V")

    def copy(self) -> "TryIrMatch":
        return TryIrMatch(
            arch=self.arch,
            symbolic_registers=self.symbolic_registers.copy(),
            symbolic_labels=self.symbolic_labels.copy(),
            symbolic_args=self.symbolic_args.copy(),
            ref_map=self.ref_map.copy(),
        )

    def _match_var(self, var_map: Dict[K, V], key: K, value: V) -> bool:
        if key in var_map:
            if var_map[key] != value:
                return False
        else:
            var_map[key] = value
        return True

    def match_reg(self, pat: Register, cand: Register) -> bool:
        # Single-letter registers are symbolic, and not matched exactly
        if len(pat.register_name) > 1:
            return pat == cand
        return self._match_var(self.symbolic_registers, pat.register_name, cand)

    def match_arg(self, pat: Argument, cand: Argument) -> bool:
        if isinstance(pat, AsmLiteral):
            return pat == cand
        if isinstance(pat, Register):
            return isinstance(cand, Register) and self.match_reg(pat, cand)
        if isinstance(pat, AsmGlobalSymbol):
            if pat.symbol_name.isupper():
                return self._match_var(self.symbolic_args, pat.symbol_name, cand)
            else:
                return pat == cand
        if isinstance(pat, AsmAddressMode):
            return (
                isinstance(cand, AsmAddressMode)
                and self.match_arg(pat.lhs, cand.lhs)
                and self.match_reg(pat.rhs, cand.rhs)
            )
        if isinstance(pat, JumpTarget):
            return isinstance(cand, JumpTarget) and self._match_var(
                self.symbolic_labels, pat.target, cand.target
            )
        if isinstance(pat, BinOp):
            return isinstance(cand, AsmLiteral) and self.eval_math(pat) == cand.value
        assert False, f"bad pattern arg: {pat}"

    def match_location(self, pat: Location, cand: Location) -> bool:
        if isinstance(pat, Register):
            return isinstance(cand, Register) and self.match_reg(pat, cand)
        if isinstance(pat, StackLocation):
            return (
                isinstance(cand, StackLocation)
                and pat.size == cand.size
                and self.match_arg(pat.offset_as_arg(), cand.offset_as_arg())
            )
        static_assert_unreachable(pat)

    def match_instr(self, pat: Instruction, cand: Instruction) -> bool:
        if pat.mnemonic != cand.mnemonic or len(pat.args) != len(cand.args):
            return False
        if not all(self.match_arg(*args) for args in zip(pat.args, cand.args)):
            return False
        return True

    def match_ref(self, pat: Reference, cand: Reference) -> bool:
        if isinstance(pat, str) and isinstance(cand, str):
            return pat == cand
        return self._match_var(self.ref_map, pat, cand)

    def match_refset(self, pat: RefSet, cand: RefSet) -> bool:
        """Match every ref in `pat` to a unique ref in `cand`"""
        if len(pat) > len(cand):
            return False
        # TODO: This may need backtracking?
        cand_refs = cand.copy()
        for p in pat:
            assert len(cand_refs) >= 1
            for c in cand_refs:
                if self.match_ref(p, c):
                    cand_refs.remove(c)
                    break
            else:
                return False
        return True

    def match_inputrefs(
        self, pat: LocationRefSetDict, cand: LocationRefSetDict
    ) -> bool:
        for pat_reg, pat_refs in pat.items():
            cand_reg = self.map_location(pat_reg)
            cand_refs = cand.get(cand_reg)
            if not self.match_refset(pat_refs, cand_refs):
                return False
        return True


def simplify_ir_patterns(
    arch: ArchFlowGraph, flow_graph: FlowGraph, pattern_classes: List[Type[IrPattern]]
) -> None:
    # Precompute a RefSet for each mnemonic
    refs_by_mnemonic = defaultdict(list)
    for node in flow_graph.nodes:
        for ref in node.block.instruction_refs:
            refs_by_mnemonic[ref.instruction.mnemonic].append(ref)

    for pattern_class in pattern_classes:
        pattern = pattern_class.compile(arch)

        # For now, patterns can't have branches: they should only have 2 Nodes,
        # a BaseNode and an (empty) TerminalNode
        assert (
            len(pattern.flow_graph.nodes) == 2
        ), "branching patterns not yet supported"
        assert isinstance(pattern.flow_graph.nodes[0], BaseNode)
        assert isinstance(pattern.flow_graph.nodes[1], TerminalNode)
        pattern_node = pattern.flow_graph.nodes[0]

        # Perform a brute force-ish graph search to find candidate sets of instructions
        # that match the pattern with the correct dependencies & arguments. This will
        # have poor performance with large patterns that use common mnemonics.
        partial_matches = [TryIrMatch(arch=arch)]
        for pat_ref in pattern_node.block.instruction_refs:
            if pat_ref.instruction.mnemonic in ("in.fictive", "out.fictive"):
                continue

            pat_inputs = pattern.flow_graph.instr_inputs[pat_ref]

            next_partial_matches = []
            for prev_state in partial_matches:
                for cand_ref in refs_by_mnemonic.get(pat_ref.instruction.mnemonic, []):
                    cand_instr = cand_ref.instruction
                    state = prev_state.copy()
                    if not state.match_ref(pat_ref, cand_ref):
                        continue
                    if not state.match_instr(pat_ref.instruction, cand_instr):
                        continue
                    if not state.match_inputrefs(
                        pat_inputs, flow_graph.instr_inputs[cand_ref]
                    ):
                        continue
                    next_partial_matches.append(state)

            partial_matches = next_partial_matches

        for n, state in enumerate(partial_matches):
            # Perform any additional pattern-specific validation or compuation
            if not pattern.check(state):
                continue

            pattern_inputs = {
                state.map_location(p) for p in pattern.replacement_instr.inputs
            }
            pattern_outputs = LocationRefSetDict()
            for i, out in enumerate(
                reversed(pattern.replacement_instr.outputs), start=1
            ):
                out_ref = pattern_node.block.instruction_refs[-i]
                out_instr = out_ref.instruction
                assert out_instr.mnemonic == "out.fictive" and out_instr.inputs == [out]
                pattern_outputs.extend(
                    state.map_location(out),
                    pattern.flow_graph.instr_inputs[out_ref].get(out),
                )

            refs_to_replace: List[InstrRef] = []
            for pat_ref in pattern_node.block.instruction_refs[::-1]:
                if pat_ref.instruction.mnemonic in ("in.fictive", "out.fictive"):
                    continue
                cand_ref = state.map_ref(pat_ref)
                instr = cand_ref.instruction

                # Only add this instruction to refs_to_replace if its outputs are one of
                # the replacement instruction's outputs, or if they are not used by by
                # any instruction outside this pattern.
                is_unreferenced = True
                for reg, refs in flow_graph.instr_uses[cand_ref].items():
                    if pat_ref in pattern_outputs.get(reg):
                        continue
                    if not all(r in refs_to_replace for r in refs):
                        is_unreferenced = False
                        break

                if is_unreferenced:
                    refs_to_replace.append(cand_ref)
                elif any(r in pattern_inputs for r in instr.clobbers + instr.outputs):
                    # If this instruction can't be replaced, but it clobbers a needed
                    # input register, then we can't perform the rewrite
                    # TODO: It may be possible to do the rewrite by introducing
                    # additional temporary registers?
                    refs_to_replace = []
                    break
            if not refs_to_replace:
                continue

            for i, ref in enumerate(refs_to_replace):
                # Remove ref from all instr_uses
                instr = ref.instruction
                for _, rs in flow_graph.instr_inputs[ref].items():
                    for r in rs:
                        if isinstance(r, InstrRef):
                            flow_graph.instr_uses[r].remove_ref(ref)

                # The last instruction in the source is rewritten with the replacement instruction
                # (refs_to_replace is in reverse order), and the others are replaced with a nop.
                if i == 0:
                    new_asm = AsmInstruction(
                        pattern.replacement_instr.mnemonic,
                        [state.map_arg(a) for a in pattern.replacement_instr.args],
                    )
                else:
                    new_asm = AsmInstruction("nop", [])

                # Parse the asm & set the clobbers
                new_instr = arch.parse(
                    new_asm.mnemonic, new_asm.args, instr.meta.derived()
                )
                for loc in new_instr.outputs + new_instr.clobbers:
                    if loc not in new_instr.clobbers:
                        new_instr.clobbers.append(loc)

                # Replace the instruction in the block
                ref.instruction = new_instr
