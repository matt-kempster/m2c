import abc
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Callable,
    ClassVar,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from .error import static_assert_unreachable
from .flow_graph import (
    AccessRefs,
    ArchFlowGraph,
    BaseNode,
    FlowGraph,
    Instruction,
    InstrRef,
    RefSet,
    Reference,
    Node,
    TerminalNode,
    build_flowgraph,
)
from .parse_file import AsmData, Function
from .parse_instruction import (
    Access,
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmInstruction,
    AsmLiteral,
    BinOp,
    Instruction,
    InstructionMeta,
    JumpTarget,
    Macro,
    MemoryAccess,
    Register,
    parse_instruction,
)


@dataclass(eq=False, frozen=True)
class IrPattern(abc.ABC):
    parts: ClassVar[List[str]]
    replacement: ClassVar[str]

    flow_graph: FlowGraph
    replacement_instr: Instruction

    @classmethod
    def compile(cls, arch: ArchFlowGraph) -> "IrPattern":
        missing_meta = InstructionMeta.missing()
        replacement_instr = parse_instruction(cls.replacement, missing_meta, arch)

        name = f"__pattern_{cls.__name__}"
        func = Function(name=name, arguments=[])
        # Add a fictive nop instruction for each input to the replacement_instr
        # This acts as a placeholder Reference to represent where the input was set
        for inp in replacement_instr.inputs:
            func.new_instruction(
                Instruction(
                    "nop",
                    [],
                    meta=missing_meta,
                    inputs=[],
                    clobbers=[],
                    outputs=[inp],
                )
            )
        for part in cls.parts:
            func.new_instruction(parse_instruction(part, missing_meta, arch))
        # Add a fictive nop instruction for each output from the replacement_instr
        for out in replacement_instr.outputs:
            func.new_instruction(
                Instruction(
                    "nop",
                    [],
                    meta=missing_meta,
                    outputs=[out],
                    clobbers=[],
                    inputs=[],
                )
            )

        asm_data = AsmData()
        flow_graph = build_flowgraph(func, asm_data, arch, fragment=True)
        return cls(
            flow_graph=flow_graph,
            replacement_instr=replacement_instr,
        )

    def check(self, m: "TryMatchState") -> bool:
        """Override to perform additional checks/calculations before replacement."""
        return True


@dataclass
class TryMatchState:
    arch: ArchFlowGraph
    symbolic_registers: Dict[str, Register] = field(default_factory=dict)
    symbolic_labels: Dict[str, str] = field(default_factory=dict)
    symbolic_args: Dict[str, Argument] = field(default_factory=dict)
    ref_map: Dict[Reference, Reference] = field(default_factory=dict)

    K = TypeVar("K")
    V = TypeVar("V")

    def copy(self) -> "TryMatchState":
        return TryMatchState(
            arch=self.arch,
            symbolic_registers=self.symbolic_registers.copy(),
            symbolic_labels=self.symbolic_labels.copy(),
            symbolic_args=self.symbolic_args.copy(),
            ref_map=self.ref_map.copy(),
        )

    def match_var(self, var_map: Dict[K, V], key: K, value: V) -> bool:
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
        return self.match_var(self.symbolic_registers, pat.register_name, cand)

    def eval_math(self, pat: Argument) -> int:
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

    def match_arg(self, pat: Argument, cand: Argument) -> bool:
        if isinstance(pat, AsmLiteral):
            return pat == cand
        if isinstance(pat, Register):
            return isinstance(cand, Register) and self.match_reg(pat, cand)
        if isinstance(pat, AsmGlobalSymbol):
            if pat.symbol_name.isupper():
                return self.match_var(self.symbolic_args, pat.symbol_name, cand)
            else:
                return pat == cand
        if isinstance(pat, AsmAddressMode):
            return (
                isinstance(cand, AsmAddressMode)
                and self.match_arg(pat.lhs, cand.lhs)
                and self.match_reg(pat.rhs, cand.rhs)
            )
        if isinstance(pat, JumpTarget):
            return isinstance(cand, JumpTarget) and self.match_var(
                self.symbolic_labels, pat.target, cand.target
            )
        if isinstance(pat, BinOp):
            return isinstance(cand, AsmLiteral) and self.eval_math(pat) == cand.value
        assert False, f"bad pattern arg: {pat}"

    def match_access(self, pat: Access, cand: Access) -> bool:
        if isinstance(pat, Register):
            return isinstance(cand, Register) and self.match_reg(pat, cand)
        if isinstance(pat, MemoryAccess):
            return (
                isinstance(cand, MemoryAccess)
                and pat.size == cand.size
                and self.match_reg(pat.base_reg, cand.base_reg)
                and self.match_arg(pat.offset, cand.offset)
            )
        assert False, f"bad pattern access: {pat}"

    def match_instr(self, pat: Instruction, cand: Instruction) -> bool:
        if (
            pat.mnemonic != cand.mnemonic
            or len(pat.args) != len(cand.args)
            or len(pat.inputs) != len(cand.inputs)
            or len(pat.outputs) != len(cand.outputs)
        ):
            return False
        for (p_arg, c_arg) in zip(pat.args, cand.args):
            if not self.match_arg(p_arg, c_arg):
                return False
        for (p_acc, c_acc) in zip(pat.inputs, cand.inputs):
            if not self.match_access(p_acc, c_acc):
                return False
        for (p_acc, c_acc) in zip(pat.outputs, cand.outputs):
            if not self.match_access(p_acc, c_acc):
                return False
        # TODO: Do clobbers also need to be matched?
        return True

    def match_ref(self, pat: Reference, cand: Reference) -> bool:
        if isinstance(pat, str) and isinstance(cand, str):
            return pat == cand
        return self.match_var(self.ref_map, pat, cand)

    def match_refset(self, pat: RefSet, cand: RefSet) -> bool:
        if len(pat) > len(cand):
            return False
        # TODO: This may need backtracking?
        cand = cand.copy()
        for e in pat:
            assert len(cand) >= 1
            for a in cand:
                if self.match_ref(e, a):
                    cand.remove(a)
                    break
            else:
                return False
        return True

    def match_accessrefs(self, pat: AccessRefs, cand: AccessRefs) -> bool:
        for pat_reg, pat_refs in pat.items():
            # For now, skip mapping any memory accesses outside of the stack.
            # Usually, these accesses are not intended to be part of the matched pattern,
            # but in the future the pattern syntax could be extended to explicitly mark
            # which accesses must be matched.
            if (
                isinstance(pat_reg, MemoryAccess)
                and pat_reg.base_reg != self.arch.stack_pointer_reg
            ):
                continue
            cand_reg = self.map_access(pat_reg)
            cand_refs = cand.get(cand_reg)
            if not self.match_refset(pat_refs, cand_refs):
                return False
        return True

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

    def map_access(self, key: Access) -> Access:
        if isinstance(key, Register):
            return self.map_reg(key)
        elif isinstance(key, MemoryAccess):
            return MemoryAccess(
                base_reg=self.map_reg(key.base_reg),
                offset=self.map_arg(key.offset),
                size=key.size,
            )
        assert False, f"bad access: {key}"


def simplify_ir_patterns(
    arch: ArchFlowGraph, flow_graph: FlowGraph, pattern_classes: List[Type[IrPattern]]
) -> None:
    debug = False
    # Precompute a RefSet for each mnemonic
    refs_by_mnemonic = defaultdict(list)
    for node in flow_graph.nodes:
        for i, instr in enumerate(node.block.instructions):
            ref = InstrRef(node, i)
            refs_by_mnemonic[instr.mnemonic].append(ref)

    def replace_instr(ref: InstrRef, new_asm: AsmInstruction) -> None:
        # Remove ref from all instr_references
        # TODO: should the data structures be changed to better accommodate this?
        instr = ref.instruction()
        for rs in flow_graph.instr_inputs[ref].values():
            for r in rs:
                if isinstance(r, InstrRef):
                    flow_graph.instr_references[r].remove_ref(ref)

        # Parse the asm & set the clobbers
        new_instr = arch.parse(new_asm.mnemonic, new_asm.args, instr.meta.derived())
        new_instr.clobbers.extend(
            acc for acc in instr.outputs if acc not in new_instr.outputs
        )
        new_instr.clobbers.extend(
            acc for acc in instr.clobbers if acc not in new_instr.clobbers
        )

        # Replace the instruction in the block
        ref.node.block.instructions[ref.index] = new_instr

    for pattern_class in pattern_classes:
        pattern = pattern_class.compile(arch)
        assert (
            len(pattern.flow_graph.nodes) == 2
        ), "branching patterns not yet supported"
        assert isinstance(pattern.flow_graph.nodes[0], BaseNode)
        assert isinstance(pattern.flow_graph.nodes[1], TerminalNode)
        pattern_node = pattern.flow_graph.nodes[0]

        partial_matches = [TryMatchState(arch=arch)]
        for i, pat in enumerate(pattern_node.block.instructions):
            if pat.mnemonic == "nop":
                continue
            if pat.mnemonic not in refs_by_mnemonic:
                partial_matches = []
                break

            pat_ref = InstrRef(pattern_node, i)
            pat_inputs = pattern.flow_graph.instr_inputs[pat_ref]
            candidate_refs = refs_by_mnemonic[pat.mnemonic]
            # if debug: print(f"pat {pat_ref} {pat} inputs {pat_inputs}")

            next_partial_matches = []
            for prev_state in partial_matches:
                for ref in candidate_refs:
                    instr = ref.instruction()
                    assert instr is not None
                    state = prev_state.copy()
                    if not state.match_ref(pat_ref, ref):
                        continue
                    if not state.match_instr(pat, instr):
                        continue
                    if not state.match_accessrefs(
                        pat_inputs, flow_graph.instr_inputs[ref]
                    ):
                        continue
                    next_partial_matches.append(state)
            partial_matches = next_partial_matches
        last = True
        for n, state in enumerate(partial_matches):
            if not pattern.check(state):
                continue
            new_instr = AsmInstruction(
                pattern.replacement_instr.mnemonic,
                [state.map_arg(a) for a in pattern.replacement_instr.args],
            )
            if debug:
                print(f">>> Match #{n}  --> {new_instr}")
            pat_in = InstrRef(pattern.flow_graph.nodes[0], 0)
            pat_out = InstrRef(
                pattern.flow_graph.nodes[0],
                len(pattern.flow_graph.nodes[0].block.instructions) - 1,
            )
            refs_to_replace = []
            last = True
            invalid = False
            matched_inputs = [
                state.map_access(p) for p in pattern.replacement_instr.inputs
            ]
            for i, pat in reversed(list(enumerate(pattern_node.block.instructions))):
                if pat.mnemonic == "nop":
                    continue
                pat_ref = InstrRef(pattern_node, i)
                ins_ref = state.map_ref(pat_ref)
                instr = ins_ref.instruction()

                clobbers_inputs = any(r in matched_inputs for r in instr.clobbers)
                clobbers_inputs |= any(r in matched_inputs for r in instr.outputs)

                deps = flow_graph.instr_references[ins_ref].values()
                is_unrefd = all(r in refs_to_replace for rs in deps for r in rs)
                if last or is_unrefd:
                    refs_to_replace.append(ins_ref)
                elif clobbers_inputs:
                    invalid = True
                    break
                last = False
            if invalid:
                continue

            for i, pat in enumerate(pattern_node.block.instructions):
                if pat.mnemonic == "nop":
                    continue
                pat_ref = InstrRef(pattern_node, i)
                pat_instr = pat_ref.instruction()
                ins_ref = state.map_ref(pat_ref)
                rfs = flow_graph.instr_references[ins_ref]
                if debug:
                    print(
                        f"> map {str(ins_ref):16} {str(pat_ref.instruction()):>20}  <>  {str(ins_ref.instruction()):30} {' ' if ins_ref in refs_to_replace else '*'} refs: {rfs}"
                    )
                if ins_ref not in refs_to_replace:
                    continue
                nop_instr = AsmInstruction("nop", [])
                repl_instr = new_instr if ins_ref == refs_to_replace[0] else nop_instr
                replace_instr(ins_ref, repl_instr)
