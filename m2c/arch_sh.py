from __future__ import annotations
from typing import Callable, Dict, List, Optional

from .error import DecompFailure
from .options import Target
from .asm_instruction import (
    Argument,
    AsmAddressMode,
    AsmInstruction,
    AsmLiteral,
    AsmState,
    Register,
    Writeback,
)
from .instruction import (
    Instruction,
    InstructionMeta,
    Location,
)
from .translate import (
    Abi,
    AbiArgSlot,
    Arch,
    ArgLoc,
    Cast,
    Expression,
    InstrArgs,
    Literal,
    NodeState,
)

from .types import FunctionSignature, Type


class Sh2Arch(Arch):
    arch = Target.ArchEnum.SH2

    re_comment = r"!.*"
    supports_dollar_regs = False
    supports_at_addressing = True
    has_delay_slots = True

    home_space_size = 0

    stack_pointer_reg = Register("r15")
    frame_pointer_regs = [Register("r14")]
    return_address_reg = Register("pr")

    base_return_regs = [(Register("r0"), False)]
    all_return_regs = [Register("r0"), Register("r1")]

    argument_regs = [Register(r) for r in ["r4", "r5", "r6", "r7"]]
    simple_temp_regs = [Register(r) for r in ["r0", "r1", "r2", "r3"]]
    temp_regs = argument_regs + simple_temp_regs

    saved_regs = [
        Register(r) for r in ["r8", "r9", "r10", "r11", "r12", "r13", "r14", "pr"]
    ]

    all_regs = (
        saved_regs
        + temp_regs
        + [stack_pointer_reg]
        + [
            Register(r)
            for r in [
                "mach",
                "macl",
            ]
        ]
    )

    aliased_regs: Dict[str, Register] = {}

    @classmethod
    def missing_return(cls) -> List[Instruction]:
        meta = InstructionMeta.missing()
        return [
            cls.parse("rts", [], meta),
            cls.parse("nop", [], meta),
        ]

    @classmethod
    def normalize_instruction(
        cls, instr: AsmInstruction, asm_state: AsmState
    ) -> AsmInstruction:
        return instr

    @classmethod
    def parse(
        cls, mnemonic: str, args: List[Argument], meta: InstructionMeta
    ) -> Instruction:
        inputs: List[Location] = []
        clobbers: List[Location] = []
        outputs: List[Location] = []
        is_return = False
        is_load = False
        is_store = False
        has_delay_slot = False
        eval_fn: Optional[Callable[[NodeState, InstrArgs], object]] = None

        if mnemonic == "rts":
            assert len(args) == 0
            inputs = [Register("pr")]
            is_return = True
            has_delay_slot = True
        elif mnemonic == "nop":
            assert len(args) == 0
        elif mnemonic == "mov":
            assert len(args) == 2 and isinstance(args[1], Register)
            outputs = [args[1]]
            if isinstance(args[0], Register):
                inputs = [args[0]]
                eval_fn = lambda s, a: s.set_reg(a.reg_ref(1), a.reg(0))
            else:
                assert isinstance(args[0], AsmLiteral)
                eval_fn = lambda s, a: s.set_reg(a.reg_ref(1), Literal(a.imm_value(0)))
        elif mnemonic == "mov.l":
            assert len(args) == 2
            if isinstance(args[0], Register):
                assert isinstance(args[1], AsmAddressMode)
                assert args[1].base == cls.stack_pointer_reg
                assert args[1].writeback == Writeback.PRE
                inputs = [args[0], args[1].base]
                is_store = True
            else:
                assert isinstance(args[0], AsmAddressMode)
                assert isinstance(args[1], Register)
                assert args[0].base == cls.stack_pointer_reg
                assert args[0].writeback == Writeback.POST
                inputs = [args[0].base]
                outputs = [args[1]]
                is_load = True
        else:
            raise DecompFailure(f"Unable to parse instruction: {mnemonic}")

        return Instruction(
            mnemonic=mnemonic,
            args=args,
            meta=meta,
            inputs=inputs,
            clobbers=clobbers,
            outputs=outputs,
            eval_fn=eval_fn,
            is_return=is_return,
            is_load=is_load,
            is_store=is_store,
            has_delay_slot=has_delay_slot,
        )

    def arg_name(self, loc: ArgLoc) -> str:
        if loc.offset is not None:
            return f"arg{loc.offset // 4 + 4}"
        assert loc.reg is not None
        reg_num = int(loc.reg.register_name[1:])
        return f"arg{reg_num - 4}"

    def function_abi(
        self,
        fn_sig: FunctionSignature,
        likely_regs: Dict[Register, bool],
        *,
        for_call: bool,
    ) -> Abi:
        known_slots: List[AbiArgSlot] = []
        candidate_slots: List[AbiArgSlot] = []
        possible_slots: List[AbiArgSlot] = []

        if fn_sig.params_known:
            for i, param in enumerate(fn_sig.params):
                param_type = param.type.decay()
                reg = Register(f"r{i + 4}") if i < 4 else None
                stack_offset = None if reg is not None else (i - 4) * 4
                known_slots.append(
                    AbiArgSlot(
                        ArgLoc(stack_offset, i, reg),
                        param_type,
                        name=param.name,
                    )
                )
        else:
            candidate_slots = [
                AbiArgSlot(ArgLoc(None, i, Register(f"r{i + 4}")), Type.intptr())
                for i in range(4)
            ]

        # argument registers are filled in order, so using a higher register implies
        # that the preceding registers may also contain arguments
        highest_used = -1
        for i, reg in enumerate(self.argument_regs):
            if likely_regs.get(reg, False):
                highest_used = i
        for i, slot in enumerate(candidate_slots):
            if i <= highest_used:
                possible_slots.append(slot)

        return Abi(
            arg_slots=known_slots,
            possible_slots=possible_slots,
        )

    @staticmethod
    def function_return(expr: Expression) -> Dict[Register, Expression]:
        return {
            Register("r0"): Cast(
                expr, reinterpret=True, silent=True, type=Type.intptr()
            ),
        }
