from __future__ import annotations
from typing import Callable, Dict, List, Optional

from .error import DecompFailure
from .options import Target
from .asm_instruction import (
    Argument,
    AsmInstruction,
    AsmState,
    Register,
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
    NodeState,
)

from .types import FunctionSignature, Type


class Sh2Arch(Arch):
    arch = Target.ArchEnum.SH2

    re_comment = r"!.*"
    supports_dollar_regs = False

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
        has_delay_slot = False
        eval_fn: Optional[Callable[[NodeState, object], object]] = None

        if mnemonic == "rts":
            assert len(args) == 0
            inputs = [Register("pr")]
            is_return = True
            has_delay_slot = True
        elif mnemonic == "nop":
            assert len(args) == 0
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
        possible_slots: List[AbiArgSlot] = []

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
