import abc
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import permutations
from typing import ClassVar, Dict, List, Optional, TypeVar

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

    def check(self, m: "IrMatch") -> bool:
        """Override to perform additional checks/calculations before replacement."""
        return True

    def compile(self, arch: ArchFlowGraph) -> "CompiledIrPattern":
        missing_meta = InstructionMeta.missing()
        regf = RegFormatter()
        replacement_instr = parse_instruction(
            self.replacement, missing_meta, arch, regf
        )

        name = f"__pattern_{self.__class__.__name__}"
        func = Function(name=name)
        # Add a fictive nop instruction for each input to the replacement_instr
        # This acts as a placeholder Reference to represent where the input was set
        assert len(replacement_instr.inputs) == len(
            set(replacement_instr.inputs)
        ), "pattern inputs must be unique"
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
        for part in self.parts:
            func.new_instruction(parse_instruction(part, missing_meta, arch, regf))

        asm_data = AsmData()
        flow_graph = build_flowgraph(func, asm_data, arch, fragment=True)
        return CompiledIrPattern(
            source=self,
            flow_graph=flow_graph,
            replacement_instr=replacement_instr,
        )


@dataclass(eq=False, frozen=True)
class CompiledIrPattern:
    source: IrPattern

    flow_graph: FlowGraph
    replacement_instr: Instruction


@dataclass
class IrMatch:
    """
    IrMatch represents the matched state of an IrPattern.
    This object is considered read-only; none of its methods modify its state.
    Its `map_*` methods take a pattern part and return the matched instruction part.
    """

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

    def map_ref(self, key: Reference) -> InstrRef:
        value = self.ref_map[key]
        assert isinstance(value, InstrRef)
        return value

    def try_map_ref(self, key: Reference) -> Optional[Reference]:
        return self.ref_map.get(key)

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

    def match_arg(self, pat: Argument, cand: Argument) -> bool:
        if isinstance(pat, AsmLiteral):
            return pat == cand
        if isinstance(pat, Register):
            # Single-letter registers are symbolic
            if len(pat.register_name) > 1:
                return pat == cand
            if not isinstance(cand, Register):
                return False
            return self._match_var(self.symbolic_registers, pat.register_name, cand)
        if isinstance(pat, AsmGlobalSymbol):
            # Uppercase AsmGlobalSymbols are symbolic
            if pat.symbol_name.isupper():
                return self._match_var(self.symbolic_args, pat.symbol_name, cand)
            else:
                return pat == cand
        if isinstance(pat, AsmAddressMode):
            return (
                isinstance(cand, AsmAddressMode)
                and self.match_arg(pat.lhs, cand.lhs)
                and self.match_arg(pat.rhs, cand.rhs)
            )
        if isinstance(pat, JumpTarget):
            return isinstance(cand, JumpTarget) and self._match_var(
                self.symbolic_labels, pat.target, cand.target
            )
        if isinstance(pat, BinOp):
            return isinstance(cand, AsmLiteral) and self.eval_math(pat) == cand.value
        assert False, f"bad pattern arg: {pat}"

    def match_instr(self, pat: Instruction, cand: Instruction) -> bool:
        if pat.mnemonic != cand.mnemonic or len(pat.args) != len(cand.args):
            return False
        return all(self.match_arg(*args) for args in zip(pat.args, cand.args))

    def match_ref(self, pat: Reference, cand: Reference) -> bool:
        if isinstance(pat, str) and isinstance(cand, str):
            return pat == cand
        return self._match_var(self.ref_map, pat, cand)

    def permute_and_match_refset(self, pat: RefSet, cand: RefSet) -> List["TryIrMatch"]:
        """
        Return a list of all possible TryIrMatch states, where every ref in `pat` is
        matched to a unique ref in `cand`. This may modify the original TryIrMatch object.
        Although this function is technically exponential in the size of `pat`, it
        usually only contains a single ref.
        """
        if len(pat) > len(cand):
            # Pigeonhole principle: there can't be a unique ref in `cand` for each ref in `pat`
            return []
        if len(pat) == 0:
            # Vacuous special case: nothing to match
            return [self]
        pat_unique = pat.get_unique()
        cand_unique = cand.get_unique()
        if pat_unique is not None and cand_unique is not None:
            # Optimization for the most common case to avoid a copy
            if self.match_ref(pat_unique, cand_unique):
                return [self]
            return []

        matches = []
        for cand_perm in permutations(cand, len(pat)):
            state = self.copy()
            for p, c in zip(pat, cand_perm):
                if not state.match_ref(p, c):
                    break
            else:
                matches.append(state)
        return matches

    def permute_and_match_inputrefs(
        self, pat: LocationRefSetDict, cand: LocationRefSetDict
    ) -> List["TryIrMatch"]:
        matches = [self]
        for pat_loc, pat_refs in pat.items():
            cand_loc = self.map_location(pat_loc)
            cand_refs = cand.get(cand_loc)
            new_matches = []
            for state in matches:
                new_matches.extend(state.permute_and_match_refset(pat_refs, cand_refs))
            matches = new_matches
        return matches

    def rename_reg(self, pat: Register, new_reg: Register) -> None:
        assert pat.register_name in self.symbolic_registers
        self.symbolic_registers[pat.register_name] = new_reg


def simplify_ir_patterns(
    arch: ArchFlowGraph, flow_graph: FlowGraph, pattern_classes: List[IrPattern]
) -> None:
    # Precompute a RefSet for each mnemonic
    refs_by_mnemonic = defaultdict(list)
    for node in flow_graph.nodes:
        for ref in node.block.instruction_refs:
            refs_by_mnemonic[ref.instruction.mnemonic].append(ref)

    # Counter used to name temporary registers
    fictive_reg_index = 0

    for pattern_class in pattern_classes:
        pattern = pattern_class.compile(arch)

        # For now, patterns can't have branches: they should only have 2 Nodes,
        # a BaseNode and an (empty) TerminalNode.
        assert (
            len(pattern.flow_graph.nodes) == 2
        ), "branching patterns not yet supported"
        assert isinstance(pattern.flow_graph.nodes[0], BaseNode)
        assert isinstance(pattern.flow_graph.nodes[1], TerminalNode)
        pattern_node = pattern.flow_graph.nodes[0]
        pattern_refs = pattern_node.block.instruction_refs

        # Split the pattern asm into 3 disjoint sets of instructions:
        # input_refs ("in.fictive"s), body_refs, and tail_ref (the last instruction)
        n_inputs = len(pattern.replacement_instr.inputs)
        input_refs, body_refs, tail_ref = (
            pattern_refs[:n_inputs],
            pattern_refs[n_inputs:-1],
            pattern_refs[-1],
        )
        assert all(r.instruction.mnemonic == "in.fictive" for r in input_refs)
        assert all(r.instruction.mnemonic != "in.fictive" for r in body_refs)

        # For now, pattern inputs must be Registers, not StackLocations. It's not always
        # trivial to create temporary StackLocations in the same way we create temporary
        # Registers during replacement.
        assert all(
            isinstance(inp, Register) for inp in pattern.replacement_instr.inputs
        )

        # For now, patterns can only have 1 output register (which must be set by the
        # final instruction in the pattern). This simplifies the replacement step because
        # we can replace the final instruction and know that all the pattern inputs have
        # been assigned by there, and the output has not yet been used.
        assert len(pattern.replacement_instr.outputs) == 1
        assert pattern.replacement_instr.outputs == tail_ref.instruction.outputs

        # Start the matching by finding all possible matches for the last instruction
        try_matches = []
        tail_inputs = pattern.flow_graph.instr_inputs[tail_ref]
        for cand_ref in refs_by_mnemonic.get(tail_ref.instruction.mnemonic, []):
            state = TryIrMatch()
            if state.match_ref(tail_ref, cand_ref):
                try_matches.append(state)

        # Continue matching by working backwards through the pattern
        for pat_ref in [tail_ref] + body_refs[::-1]:
            pat_inputs = pattern.flow_graph.instr_inputs[pat_ref]

            next_try_matches = []
            for state in try_matches:
                # By pattern construction, pat_ref should be in the state's ref_map
                # (It would be missing for "disjoint" or irrelevant instructions in the
                # pattern, like random nops: these aren't allowed)
                cand = state.try_map_ref(pat_ref)
                if not isinstance(cand, InstrRef):
                    continue
                if not state.match_instr(pat_ref.instruction, cand.instruction):
                    continue
                states = state.permute_and_match_inputrefs(
                    pat_inputs,
                    flow_graph.instr_inputs[cand],
                )
                next_try_matches.extend(states)
            try_matches = next_try_matches

        for n, state in enumerate(try_matches):
            # Perform any additional pattern-specific validation or computation
            if not pattern.source.check(state):
                continue

            # Determine which instructions we can replace with the replacement_instr or nops
            refs_to_replace: List[InstrRef] = [state.map_ref(tail_ref)]
            for pat_ref in body_refs[::-1]:
                cand_ref = state.map_ref(pat_ref)
                # The candidate instruction can be replaced if all of its downstream
                # uses are also being replaced
                if all(
                    all(r in refs_to_replace for r in refs)
                    for refs in flow_graph.instr_uses[cand_ref].values()
                ):
                    refs_to_replace.append(cand_ref)

            # Create temporary registers for the inputs to the replacement_instr
            temp_reg_refs = {}
            for pat_ref in input_refs:
                assert len(pat_ref.instruction.outputs) == 1
                input_reg = pat_ref.instruction.outputs[0]
                assert isinstance(input_reg, Register)

                # It doesn't matter which instruction we pick; the new move instruction to save
                # the register can come before any of the instructions that use it.
                input_uses = pattern.flow_graph.instr_uses[pat_ref].get(input_reg)
                input_use_ref = state.map_ref(next(iter(input_uses)))

                # Create a unique fictive register to act as a temporary
                original_reg = state.map_reg(input_reg)
                temp_reg = Register(
                    f"{original_reg.register_name}_fictive_{fictive_reg_index}"
                )
                fictive_reg_index += 1
                state.rename_reg(input_reg, temp_reg)

                move_instr = arch.parse(
                    "move.fictive",
                    [temp_reg, original_reg],
                    input_use_ref.instruction.meta,
                )
                move_ref = input_use_ref.add_instruction_before(move_instr)
                temp_reg_refs[temp_reg] = move_ref

                # Update the instr_inputs/instr_uses graph
                flow_graph.instr_inputs[move_ref] = LocationRefSetDict()
                flow_graph.instr_uses[move_ref] = LocationRefSetDict()
                for src_ref in flow_graph.instr_inputs[input_use_ref].get(original_reg):
                    if isinstance(src_ref, InstrRef):
                        flow_graph.instr_inputs[move_ref].add(original_reg, src_ref)
                        flow_graph.instr_uses[src_ref].add(original_reg, move_ref)

            for i, ref in enumerate(refs_to_replace):
                # Remove ref from all instr_uses/instr_inputs
                for input_loc in ref.instruction.inputs:
                    for input_ref in flow_graph.instr_inputs[ref].get(input_loc):
                        flow_graph.instr_inputs[ref].remove(input_loc)
                        if isinstance(input_ref, InstrRef):
                            flow_graph.instr_uses[input_ref].remove_ref(ref)
                assert flow_graph.instr_inputs[ref].is_empty()

                # The last instruction in the source is rewritten with the replacement instruction
                # (refs_to_replace is in reverse order), and the others are replaced with a nop.
                if i == 0:
                    new_asm = AsmInstruction(
                        pattern.replacement_instr.mnemonic,
                        [state.map_arg(a) for a in pattern.replacement_instr.args],
                    )

                    # Update instr_inputs/instr_uses to mark that the replacement instruction
                    # will depend on all of the temporary registers added earlier
                    for temp_reg, temp_ref in temp_reg_refs.items():
                        flow_graph.instr_inputs[ref].add(temp_reg, temp_ref)
                        flow_graph.instr_uses[temp_ref].add(temp_reg, ref)
                else:
                    new_asm = AsmInstruction("nop", [])
                    # Assert that there are no uses of the outputs from this instruction
                    assert flow_graph.instr_uses[ref].is_empty()

                # Parse the asm & set the clobbers
                instr = ref.instruction
                new_instr = arch.parse(
                    new_asm.mnemonic, new_asm.args, instr.meta.derived()
                )
                for loc in instr.outputs + instr.clobbers:
                    if loc not in new_instr.clobbers:
                        new_instr.clobbers.append(loc)

                # Replace the instruction in the block
                ref.instruction = new_instr

    # After all of the rewrites above, verify that the instr_inputs & instr_uses
    # dicts are still consistent by recomputing instr_uses.
    new_uses = {r: LocationRefSetDict() for r in flow_graph.instr_inputs}
    for ref, inputs in flow_graph.instr_inputs.items():
        for reg, deps in inputs.items():
            for dep in deps:
                if isinstance(dep, InstrRef):
                    new_uses[dep].add(reg, ref)

    # Assert that new_uses is equivalent to flow_graph.instr_uses
    for ref, uses in new_uses.items():
        for reg, deps in uses.items():
            assert deps == flow_graph.instr_uses[ref].get(reg)
