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
    ref_map: Dict[Union[InstrRef, str], Union[InstrRef, str]] = field(
        default_factory=dict
    )

    T = TypeVar("T")

    def copy(self) -> "TryMatchState":
        return TryMatchState(
            arch=self.arch,
            symbolic_registers=self.symbolic_registers.copy(),
            symbolic_labels=self.symbolic_labels.copy(),
            symbolic_args=self.symbolic_args.copy(),
            ref_map=self.ref_map.copy(),
        )

    def match_var(self, var_map: Dict[str, T], key: str, value: T) -> bool:
        if key in var_map:
            if var_map[key] != value:
                return False
        else:
            var_map[key] = value
        return True

    def match_reg(self, actual: Register, exp: Register) -> bool:
        # Single-letter registers are symbolic, and not matched exactly
        if len(exp.register_name) <= 1:
            return self.match_var(self.symbolic_registers, exp.register_name, actual)
        else:
            return exp.register_name == actual.register_name

    def eval_math(self, e: Argument) -> int:
        if isinstance(e, AsmLiteral):
            return e.value
        if isinstance(e, BinOp):
            if e.op == "+":
                return self.eval_math(e.lhs) + self.eval_math(e.rhs)
            if e.op == "-":
                return self.eval_math(e.lhs) - self.eval_math(e.rhs)
            if e.op == "<<":
                return self.eval_math(e.lhs) << self.eval_math(e.rhs)
            assert False, f"bad binop in math pattern: {e}"
        elif isinstance(e, AsmGlobalSymbol):
            assert (
                e.symbol_name in self.symbolic_args
            ), f"undefined variable in math pattern: {e.symbol_name}"
            lit = self.symbolic_args[e.symbol_name]
            assert isinstance(lit, AsmLiteral)
            return lit.value
        else:
            assert False, f"bad pattern part in math pattern: {e}"

    def match_arg(self, a: Argument, e: Argument) -> bool:
        if isinstance(e, AsmLiteral):
            return isinstance(a, AsmLiteral) and a.value == e.value
        if isinstance(e, Register):
            return isinstance(a, Register) and self.match_reg(a, e)
        if isinstance(e, AsmGlobalSymbol):
            if e.symbol_name.isupper():
                return self.match_var(self.symbolic_args, e.symbol_name, a)
            else:
                return isinstance(a, AsmGlobalSymbol) and a.symbol_name == e.symbol_name
        if isinstance(e, AsmAddressMode):
            return (
                isinstance(a, AsmAddressMode)
                and self.match_arg(a.lhs, e.lhs)
                and self.match_reg(a.rhs, e.rhs)
            )
        if isinstance(e, JumpTarget):
            return isinstance(a, JumpTarget) and self.match_var(
                self.symbolic_labels, e.target, a.target
            )
        if isinstance(e, BinOp):
            return isinstance(a, AsmLiteral) and a.value == self.eval_math(e)
        assert False, f"bad pattern part: {e}"


    def match_access(self, a: Access, e: Access) -> bool:
        if isinstance(e, Register):
            return isinstance(a, Register) and self.match_reg(a, e)
        if isinstance(e, MemoryAccess):
            return (
                isinstance(a, MemoryAccess)
                and a.size == e.size
                and self.match_reg(a.base_reg, e.base_reg)
                and self.match_arg(a.offset, e.offset)
            )
        assert False, f"bad access: {e}"

    def match_instr(self, ins: Instruction, exp: Instruction) -> bool:
        if (
            ins.mnemonic != exp.mnemonic
            or len(ins.args) != len(ins.args)
            or len(ins.inputs) != len(ins.inputs)
            or len(ins.outputs) != len(ins.outputs)
        ):
            return False
        for (a_arg, e_arg) in zip(ins.args, exp.args):
            if not self.match_arg(a_arg, e_arg):
                return False
        for (a_acc, e_acc) in zip(ins.inputs, exp.inputs):
            if not self.match_access(a_acc, e_acc):
                return False
        for (a_acc, e_acc) in zip(ins.outputs, exp.outputs):
            if not self.match_access(a_acc, e_acc):
                return False
        # TODO: What about clobbers?
        return True

    def match_ref(self, key: Reference, value: Reference) -> bool:
        # TODO: This is backwards
        if isinstance(key, str) and isinstance(value, str):
            return key == value
        existing_value = self.ref_map.get(key)
        if existing_value is not None:
            return existing_value == value
        self.ref_map[key] = value
        return True

    def match_refset(self, exp: RefSet, act: RefSet) -> bool:
        # TODO: This is backwards
        if len(exp) > len(act):
            return False
        # TODO: This may need backtracking?
        act = act.copy()
        for e in exp:
            assert len(act) >= 1
            for a in act:
                if self.match_ref(e, a):
                    act.remove(a)
                    break
            else:
                return False
        return True

    def match_accessrefs(self, exp: AccessRefs, act: AccessRefs) -> bool:
        # TODO: This is backwards
        for exp_reg, exp_refs in exp.items():
            if (
                isinstance(exp_reg, MemoryAccess)
                and exp_reg.base_reg != self.arch.stack_pointer_reg
            ):
                continue
            mapped_reg = self.map_access(exp_reg)
            act_refs = act.get(mapped_reg)
            if not self.match_refset(exp_refs, act_refs):
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
                    if not state.match_instr(instr, pat):
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
