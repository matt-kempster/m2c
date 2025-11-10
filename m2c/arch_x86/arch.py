from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..asm_instruction import (
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmInstruction,
    AsmLiteral,
    AsmState,
    Register,
    get_jump_target,
    traverse_arg,
)
from ..error import DecompFailure
from ..instruction import Instruction, InstructionMeta, Location, StackLocation
from ..options import Target
from ..translate import (
    Abi,
    AbiArgSlot,
    ArgLoc,
    Arch,
    BinaryOp,
    Cast,
    Expression,
    InstrArgs,
    Literal,
    NodeState,
    Type,
    as_u32,
)
from ..types import FunctionSignature


def _no_op_eval(_: NodeState, __: InstrArgs) -> None:
    """Placeholder evaluation function for instructions we model structurally."""
    return None


class X86Arch(Arch):
    arch = Target.ArchEnum.X86

    re_comment = r"[;#].*"
    supports_dollar_regs = False

    home_space_size = 0

    stack_pointer_reg = Register("esp")
    frame_pointer_regs = [Register("ebp")]
    return_address_reg = Register("eip")

    base_return_regs = [(Register("eax"), False)]
    all_return_regs = [Register("eax"), Register("edx")]
    argument_regs: List[Register] = []
    simple_temp_regs = [Register("ecx"), Register("edx")]
    temp_regs = simple_temp_regs + [
        Register("eax"),
        Register("ebx"),
        Register("esi"),
        Register("edi"),
    ]
    saved_regs = [
        Register("ebp"),
        Register("ebx"),
        Register("esi"),
        Register("edi"),
    ]
    flag_regs = [
        Register("eflags"),
        Register("zf"),
        Register("cf"),
        Register("sf"),
        Register("of"),
    ]
    all_regs = (
        saved_regs
        + temp_regs
        + [stack_pointer_reg, Register("eip")]
        + flag_regs
    )
    aliased_regs: Dict[str, Register] = {
        "sp": stack_pointer_reg,
        "bp": Register("ebp"),
        "ip": Register("eip"),
    }

    size_prefixes = {
        "byte",
        "word",
        "dword",
        "qword",
        "ptr",
        "short",
        "near",
        "far",
        "long",
    }

    _instr_parsers: Dict[str, Callable[[List[Argument], InstructionMeta], Instruction]] = {}

    @classmethod
    def should_ignore_symbol(cls, symbol: str) -> bool:
        return symbol in cls.size_prefixes

    @classmethod
    def missing_return(cls) -> List[Instruction]:
        return [cls.parse("ret", [], InstructionMeta.missing())]

    @classmethod
    def normalize_instruction(
        cls, instr: AsmInstruction, asm_state: AsmState
    ) -> AsmInstruction:
        return instr

    @classmethod
    def _unsupported_message(cls, mnemonic: str, args: List[Argument]) -> str:
        rendered_args = ", ".join(str(arg) for arg in args)
        suffix = f" {rendered_args}" if rendered_args else ""
        return f"x86 unsupported instruction: {mnemonic}{suffix}"

    @classmethod
    def parse(
        cls, mnemonic: str, args: List[Argument], meta: InstructionMeta
    ) -> Instruction:
        handler = cls._instr_parsers.get(mnemonic)
        if handler is None:
            raise DecompFailure(cls._unsupported_message(mnemonic, args))
        return handler(args, meta)

    @classmethod
    def _make_instruction(
        cls,
        mnemonic: str,
        args: List[Argument],
        meta: InstructionMeta,
        *,
        inputs: Optional[List[Register]] = None,
        outputs: Optional[List[Register]] = None,
        clobbers: Optional[List[Register]] = None,
        is_return: bool = False,
        eval_fn: Callable[[NodeState, InstrArgs], object] = _no_op_eval,
    ) -> Instruction:
        return Instruction(
            mnemonic=mnemonic,
            args=args,
            meta=meta,
            inputs=list(inputs or []),
            clobbers=list(clobbers or []),
            outputs=list(outputs or []),
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=is_return,
            is_store=False,
            is_load=False,
            eval_fn=eval_fn,
        )

    @classmethod
    def _parse_ret(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert not args, "ret does not take operands"
        return cls._make_instruction(
            "ret",
            args,
            meta,
            inputs=[cls.stack_pointer_reg],
            is_return=True,
        )

    @classmethod
    def _parse_nop(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert not args, "nop does not take operands"
        return cls._make_instruction("nop", args, meta)

    @classmethod
    def _stack_location_from_address(cls, addr: AsmAddressMode) -> Optional[StackLocation]:
        if addr.base == cls.stack_pointer_reg and isinstance(addr.addend, AsmLiteral):
            return StackLocation.from_offset(addr.addend)
        if addr.base == cls.frame_pointer_regs[0] and isinstance(addr.addend, AsmLiteral):
            loc = StackLocation.from_offset(addr.addend)
            if loc is not None:
                return StackLocation(offset=loc.offset, symbolic_offset=None)
        return None

    @classmethod
    def _parse_mov(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 2, "mov expects two operands"
        dst, src = args
        outputs: List[Location] = []
        inputs: List[Location] = []

        def eval_mov(state: NodeState, a: InstrArgs) -> None:
            if isinstance(dst, Register):
                state.set_reg(dst, a.arg(1))

        src_is_mem = isinstance(src, AsmAddressMode)
        dst_is_mem = isinstance(dst, AsmAddressMode)

        if src_is_mem and dst_is_mem:
            raise DecompFailure(cls._unsupported_message("mov", args))

        if isinstance(dst, Register):
            outputs.append(dst)
        elif isinstance(dst, AsmAddressMode):
            loc = cls._stack_location_from_address(dst)
            if loc is not None:
                outputs.append(loc)
            inputs.append(dst.base)
            for sub in traverse_arg(dst.addend):
                if isinstance(sub, Register) and sub not in inputs:
                    inputs.append(sub)
        else:
            raise DecompFailure(cls._unsupported_message("mov", args))

        if isinstance(src, AsmAddressMode):
            loc = cls._stack_location_from_address(src)
            if loc is not None:
                inputs.append(loc)
            inputs.append(src.base)
            for sub in traverse_arg(src.addend):
                if isinstance(sub, Register) and sub not in inputs:
                    inputs.append(sub)
        elif isinstance(src, Register):
            inputs.append(src)

        return Instruction(
            mnemonic="mov",
            args=args,
            meta=meta,
            inputs=inputs,
            clobbers=[],
            outputs=outputs,
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=False,
            is_store=dst_is_mem,
            is_load=src_is_mem,
            eval_fn=eval_mov,
        )

    @classmethod
    def _parse_push(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 1, "push expects one operand"
        src = args[0]
        inputs: List[Register] = [cls.stack_pointer_reg]
        if isinstance(src, Register):
            inputs.append(src)

        def eval_push(state: NodeState, a: InstrArgs) -> None:
            esp = cls.stack_pointer_reg
            current = state.regs[esp]
            new_sp = BinaryOp.intptr(current, "-", Literal(4))
            state.set_reg(esp, new_sp)

        return Instruction(
            mnemonic="push",
            args=args,
            meta=meta,
            inputs=inputs,
            clobbers=[],
            outputs=[cls.stack_pointer_reg],
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=False,
            is_store=True,
            is_load=False,
            eval_fn=eval_push,
        )

    @classmethod
    def _parse_pop(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 1, "pop expects one operand"
        dst = args[0]
        if not isinstance(dst, Register):
            raise DecompFailure(cls._unsupported_message("pop", args))

        def eval_pop(state: NodeState, a: InstrArgs) -> None:
            esp = cls.stack_pointer_reg
            current = state.regs[esp]
            new_sp = BinaryOp.intptr(current, "+", Literal(4))
            state.set_reg(dst, a.arg(0))
            state.set_reg(esp, new_sp)

        return Instruction(
            mnemonic="pop",
            args=args,
            meta=meta,
            inputs=[dst, cls.stack_pointer_reg],
            clobbers=[],
            outputs=[dst, cls.stack_pointer_reg],
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=False,
            is_store=False,
            is_load=True,
            eval_fn=eval_pop,
        )

    @classmethod
    def _parse_sub(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 2, "sub expects two operands"
        dst, src = args
        if dst != cls.stack_pointer_reg or not isinstance(src, AsmLiteral):
            raise DecompFailure(cls._unsupported_message("sub", args))

        def eval_sub(state: NodeState, a: InstrArgs) -> None:
            esp = cls.stack_pointer_reg
            current = state.regs[esp]
            new_sp = BinaryOp.intptr(current, "-", Literal(src.value))
            state.set_reg(esp, new_sp)

        return Instruction(
            mnemonic="sub",
            args=args,
            meta=meta,
            inputs=[cls.stack_pointer_reg],
            clobbers=[],
            outputs=[cls.stack_pointer_reg],
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=False,
            is_store=False,
            is_load=False,
            eval_fn=eval_sub,
        )

    @classmethod
    def _parse_add(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 2, "add expects two operands"
        dst, src = args
        if not isinstance(dst, Register) or not isinstance(src, AsmLiteral):
            raise DecompFailure(cls._unsupported_message("add", args))

        def eval_add(state: NodeState, a: InstrArgs) -> None:
            current = state.regs[dst]
            add_expr = BinaryOp.intptr(current, "+", Literal(src.value))
            state.set_reg(dst, add_expr)

        return Instruction(
            mnemonic="add",
            args=args,
            meta=meta,
            inputs=[dst],
            clobbers=[],
            outputs=[dst],
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=False,
            is_store=False,
            is_load=False,
            eval_fn=eval_add,
        )

    @classmethod
    def _parse_test(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 2, "test expects two operands"
        lhs, rhs = args
        if not isinstance(lhs, Register) or not isinstance(rhs, Register):
            raise DecompFailure(cls._unsupported_message("test", args))

        def eval_test(state: NodeState, a: InstrArgs) -> None:
            value = BinaryOp.int(a.reg(0), "&", a.reg(1))
            zero = BinaryOp.icmp(value, "==", Literal(0))
            state.set_reg(Register("zf"), zero)

        return Instruction(
            mnemonic="test",
            args=args,
            meta=meta,
            inputs=[lhs, rhs],
            clobbers=[],
            outputs=[Register("zf")],
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=False,
            is_store=False,
            is_load=False,
            eval_fn=eval_test,
        )

    @classmethod
    def _parse_xor(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 2, "xor expects two operands"
        dst, src = args
        if not isinstance(dst, Register) or not isinstance(src, Register):
            raise DecompFailure(cls._unsupported_message("xor", args))
        if dst != src:
            raise DecompFailure("xor currently only supports zeroing form (dst == src)")

        def eval_xor(state: NodeState, a: InstrArgs) -> None:
            zero = Literal(0, type=Type.intptr())
            state.set_reg(dst, zero)
            state.set_reg(Register("zf"), Literal(1))

        return Instruction(
            mnemonic="xor",
            args=args,
            meta=meta,
            inputs=[dst],
            clobbers=[],
            outputs=[dst],
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=False,
            is_store=False,
            is_load=False,
            eval_fn=eval_xor,
        )

    @classmethod
    def _parse_call(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 1, "call expects one operand"
        target = args[0]
        outputs = list(cls.all_return_regs)
        clobbers = list(cls.temp_regs)
        inputs: List[Register] = list(cls.argument_regs)

        if isinstance(target, Register):
            inputs.append(target)
            eval_fn = lambda s, a: s.make_function_call(a.reg(0), outputs)
        elif isinstance(target, (AsmGlobalSymbol, AsmLiteral)):
            eval_fn = lambda s, a: s.make_function_call(a.sym_imm(0), outputs)
        else:
            raise DecompFailure(cls._unsupported_message("call", args))

        return Instruction(
            mnemonic="call",
            args=args,
            meta=meta,
            inputs=inputs,
            clobbers=clobbers,
            outputs=outputs,
            jump_target=None,
            function_target=target,
            is_conditional=False,
            is_return=False,
            is_store=False,
            is_load=False,
            eval_fn=eval_fn,
        )

    @classmethod
    def _parse_lea(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 2, "lea expects two operands"
        dst, src = args
        if not isinstance(dst, Register) or not isinstance(src, AsmAddressMode):
            raise DecompFailure(cls._unsupported_message("lea", args))

        inputs: List[Location] = [src.base]
        for sub in traverse_arg(src.addend):
            if isinstance(sub, Register) and sub not in inputs:
                inputs.append(sub)

        return Instruction(
            mnemonic="lea",
            args=args,
            meta=meta,
            inputs=inputs,
            clobbers=[],
            outputs=[dst],
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=False,
            is_store=False,
            is_load=False,
            eval_fn=_no_op_eval,
        )

    @classmethod
    def _parse_dec(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 1, "dec expects one operand"
        dst = args[0]
        if not isinstance(dst, Register):
            raise DecompFailure(cls._unsupported_message("dec", args))

        def eval_dec(state: NodeState, a: InstrArgs) -> None:
            reg_val = state.regs[dst]
            result = BinaryOp.intptr(reg_val, "-", Literal(1))
            state.set_reg(dst, result)
            state.set_reg(Register("zf"), BinaryOp.icmp(result, "==", Literal(0)))

        return Instruction(
            mnemonic="dec",
            args=args,
            meta=meta,
            inputs=[dst],
            clobbers=[],
            outputs=[dst],
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=False,
            is_store=False,
            is_load=False,
            eval_fn=eval_dec,
        )

    @classmethod
    def _parse_cmp(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 2, "cmp expects two operands"
        lhs, rhs = args
        if not isinstance(lhs, Register):
            raise DecompFailure(cls._unsupported_message("cmp", args))

        if isinstance(rhs, Register):
            inputs = [lhs, rhs]

            def eval_cmp(state: NodeState, a: InstrArgs) -> None:
                diff = BinaryOp.intptr(a.reg(0), "-", a.reg(1))
                zero = BinaryOp.icmp(diff, "==", Literal(0))
                state.set_reg(Register("zf"), zero)

        elif isinstance(rhs, AsmLiteral):
            inputs = [lhs]

            def eval_cmp(state: NodeState, a: InstrArgs) -> None:
                diff = BinaryOp.intptr(a.reg(0), "-", Literal(rhs.value))
                zero = BinaryOp.icmp(diff, "==", Literal(0))
                state.set_reg(Register("zf"), zero)

        else:
            raise DecompFailure(cls._unsupported_message("cmp", args))

        return Instruction(
            mnemonic="cmp",
            args=args,
            meta=meta,
            inputs=inputs,
            clobbers=[],
            outputs=[Register("zf")],
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=False,
            is_store=False,
            is_load=False,
            eval_fn=eval_cmp,
        )

    @classmethod
    def _parse_jz(cls, args: List[Argument], meta: InstructionMeta, *, mnemonic: str) -> Instruction:
        assert len(args) == 1, "jump expects one operand"
        target = get_jump_target(args[0])

        def eval_jump(state: NodeState, a: InstrArgs) -> None:
            zero = state.regs[Register("zf")]
            cond = BinaryOp.icmp(zero, "!=", Literal(0))
            if mnemonic == "jnz":
                cond = cond.negated()
            state.set_branch_condition(cond)

        return Instruction(
            mnemonic=mnemonic,
            args=args,
            meta=meta,
            inputs=[Register("zf")],
            clobbers=[],
            outputs=[],
            jump_target=target,
            function_target=None,
            is_conditional=True,
            is_return=False,
            is_store=False,
            is_load=False,
            eval_fn=eval_jump,
        )

    def arg_name(self, loc: ArgLoc) -> str:
        if loc.offset is not None:
            return f"arg_sp{loc.offset:#x}"
        assert loc.reg is not None
        return loc.reg.register_name

    def function_abi(
        self,
        fn_sig: FunctionSignature,
        likely_regs: Dict[Register, bool],
        *,
        for_call: bool,
    ) -> Abi:
        del likely_regs, for_call
        known_slots: List[AbiArgSlot] = []
        stack_offset = 0
        if fn_sig.params_known:
            for idx, param in enumerate(fn_sig.params):
                param_type = param.type.decay()
                known_slots.append(
                    AbiArgSlot(
                        ArgLoc(offset=stack_offset, sort_index=idx, reg=None),
                        param_type,
                        name=param.name,
                    )
                )
                stack_offset += max(param_type.get_size_bytes(), 4)
        candidate_slots: List[AbiArgSlot] = []
        start_offset = stack_offset if fn_sig.params_known else 0
        for i in range(8):
            offset = start_offset + i * 4
            candidate_slots.append(
                AbiArgSlot(ArgLoc(offset=offset, sort_index=len(known_slots) + i, reg=None), Type.intptr())
            )
        return Abi(arg_slots=known_slots, possible_slots=candidate_slots)

    @staticmethod
    def function_return(expr: Expression) -> Dict[Register, Expression]:
        return {
            Register("eax"): Cast(
                expr, reinterpret=True, silent=True, type=Type.intptr()
            ),
            Register("edx"): as_u32(
                Cast(expr, reinterpret=True, silent=False, type=Type.u64())
            ),
        }


X86Arch._instr_parsers = {
    "nop": X86Arch._parse_nop,
    "ret": X86Arch._parse_ret,
    "mov": X86Arch._parse_mov,
    "push": X86Arch._parse_push,
    "pop": X86Arch._parse_pop,
    "sub": X86Arch._parse_sub,
    "add": X86Arch._parse_add,
    "test": X86Arch._parse_test,
    "xor": X86Arch._parse_xor,
    "call": X86Arch._parse_call,
    "jz": lambda args, meta: X86Arch._parse_jz(args, meta, mnemonic="jz"),
    "jnz": lambda args, meta: X86Arch._parse_jz(args, meta, mnemonic="jnz"),
    "cmp": X86Arch._parse_cmp,
    "lea": X86Arch._parse_lea,
    "dec": X86Arch._parse_dec,
}
