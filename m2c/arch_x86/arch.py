from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

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
    AddressMode,
    ArgLoc,
    Arch,
    BinaryOp,
    Cast,
    Expression,
    InstrArgs,
    Literal,
    NodeState,
    RawSymbolRef,
    Type,
    as_u32,
)
from ..evaluate import deref
from ..types import FunctionSignature

CondBuilder = Callable[["X86Arch", NodeState], Expression]


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
        "ax": Register("eax"),
        "bx": Register("ebx"),
        "cx": Register("ecx"),
        "dx": Register("edx"),
        "si": Register("esi"),
        "di": Register("edi"),
        "cl": Register("ecx"),
        "ch": Register("ecx"),
        "al": Register("eax"),
        "ah": Register("eax"),
        "bl": Register("ebx"),
        "bh": Register("ebx"),
        "dl": Register("edx"),
        "dh": Register("edx"),
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
    def _address_inputs(cls, addr: AsmAddressMode) -> List[Location]:
        inputs: List[Location] = [addr.base]
        for sub in traverse_arg(addr.addend):
            if isinstance(sub, Register) and sub not in inputs:
                inputs.append(sub)
        stack_loc = cls._stack_location_from_address(addr)
        if stack_loc is not None:
            inputs.append(stack_loc)
        return inputs

    @classmethod
    def _address_mode_reference(cls, addr: AsmAddressMode) -> Optional[Union[AddressMode, RawSymbolRef]]:
        if isinstance(addr.addend, AsmLiteral):
            return AddressMode(addr.addend.value, addr.base)
        if isinstance(addr.addend, AsmGlobalSymbol):
            return RawSymbolRef(0, addr.addend)
        return None

    @classmethod
    def _value_from_operand(
        cls,
        mnemonic: str,
        full_args: List[Argument],
        arg: Argument,
        *,
        arg_index: int,
        allow_literal: bool,
        allow_memory: bool,
        load_size: int = 4,
    ) -> Tuple[List[Location], Callable[[InstrArgs], Expression], bool]:
        if isinstance(arg, Register):
            return [arg], (lambda a, idx=arg_index: a.reg(idx)), False
        if allow_memory and isinstance(arg, AsmAddressMode):
            ref = cls._address_mode_reference(arg)
            if ref is None:
                raise DecompFailure(cls._unsupported_message(mnemonic, full_args))

            def addr_value(a: InstrArgs, ref=ref) -> Expression:
                return deref(ref, a.regs, a.stack_info, size=load_size)

            return (
                cls._address_inputs(arg),
                addr_value,
                True,
            )
        if allow_literal and isinstance(arg, AsmLiteral):
            return [], lambda _a, value=arg.value: Literal(value), False
        if allow_literal and isinstance(arg, AsmGlobalSymbol):
            return (
                [],
                lambda a, sym=arg.symbol_name: a.stack_info.global_info.address_of_gsym(sym),
                False,
            )
        raise DecompFailure(cls._unsupported_message(mnemonic, full_args))

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
    def _parse_movsx(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 2, "movsx expects two operands"
        dst, src = args
        if not isinstance(dst, Register):
            raise DecompFailure(cls._unsupported_message("movsx", args))

        inputs: List[Location] = []
        src_inputs, src_value, _ = cls._value_from_operand(
            "movsx",
            args,
            src,
            arg_index=1,
            allow_literal=False,
            allow_memory=True,
            load_size=2,
        )
        inputs.extend(src_inputs)

        def eval_movsx(state: NodeState, a: InstrArgs) -> None:
            value = src_value(a)
            state.set_reg(dst, value)

        return Instruction(
            mnemonic="movsx",
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
            is_load=isinstance(src, AsmAddressMode),
            eval_fn=eval_movsx,
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
        if dst == cls.stack_pointer_reg and isinstance(src, AsmLiteral):

            def eval_sp_sub(state: NodeState, a: InstrArgs) -> None:
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
                eval_fn=eval_sp_sub,
            )

        if not isinstance(dst, Register):
            raise DecompFailure(cls._unsupported_message("sub", args))

        if isinstance(src, Register):
            inputs = [dst, src]

            def eval_sub(state: NodeState, a: InstrArgs) -> None:
                result = BinaryOp.intptr(a.reg(0), "-", a.reg(1))
                state.set_reg(dst, result)

        elif isinstance(src, AsmLiteral):
            inputs = [dst]

            def eval_sub(state: NodeState, a: InstrArgs) -> None:
                result = BinaryOp.intptr(a.reg(0), "-", Literal(src.value))
                state.set_reg(dst, result)

        else:
            raise DecompFailure(cls._unsupported_message("sub", args))

        return Instruction(
            mnemonic="sub",
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
            eval_fn=eval_sub,
        )

    @classmethod
    def _parse_add(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 2, "add expects two operands"
        dst, src = args
        if not isinstance(dst, Register):
            raise DecompFailure(cls._unsupported_message("add", args))

        if isinstance(src, Register):
            inputs = [dst, src]

            def eval_add(state: NodeState, a: InstrArgs) -> None:
                add_expr = BinaryOp.intptr(a.reg(0), "+", a.reg(1))
                state.set_reg(dst, add_expr)

        elif isinstance(src, AsmLiteral):
            inputs = [dst]

            def eval_add(state: NodeState, a: InstrArgs) -> None:
                add_expr = BinaryOp.intptr(a.reg(0), "+", Literal(src.value))
                state.set_reg(dst, add_expr)

        else:
            raise DecompFailure(cls._unsupported_message("add", args))

        return Instruction(
            mnemonic="add",
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
            eval_fn=eval_add,
        )

    @classmethod
    def _parse_sbb(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 2, "sbb expects two operands"
        dst, src = args
        if not isinstance(dst, Register):
            raise DecompFailure(cls._unsupported_message("sbb", args))

        cf = Register("cf")

        if isinstance(src, Register):
            inputs = [dst]
            if src != dst:
                inputs.append(src)
            inputs.append(cf)

            def rhs_value(a: InstrArgs) -> Expression:
                return a.reg(1)

        elif isinstance(src, AsmLiteral):
            inputs = [dst, cf]

            def rhs_value(_: InstrArgs, value: int = src.value) -> Expression:
                return Literal(value)

        else:
            raise DecompFailure(cls._unsupported_message("sbb", args))

        def eval_sbb(state: NodeState, a: InstrArgs) -> None:
            cf_value = state.regs[cf]
            borrow = BinaryOp.int(cf_value, "&", Literal(1))
            result = BinaryOp.intptr(BinaryOp.intptr(a.reg(0), "-", rhs_value(a)), "-", borrow)
            state.set_reg(dst, result)
            state.set_reg(cf, borrow)

        return Instruction(
            mnemonic="sbb",
            args=args,
            meta=meta,
            inputs=inputs,
            clobbers=[],
            outputs=[dst, cf],
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=False,
            is_store=False,
            is_load=False,
            eval_fn=eval_sbb,
        )

    @classmethod
    def _parse_shift(cls, args: List[Argument], meta: InstructionMeta, *, mnemonic: str) -> Instruction:
        assert len(args) == 2, "shift expects two operands"
        dst, src = args
        if not isinstance(dst, Register):
            raise DecompFailure(cls._unsupported_message(mnemonic, args))

        if mnemonic in ("shl", "sal"):
            op_builder = lambda value, amount: BinaryOp.int(value, "<<", amount)
        elif mnemonic == "shr":
            op_builder = lambda value, amount: BinaryOp.uint(value, ">>", amount)
        elif mnemonic == "sar":
            op_builder = lambda value, amount: BinaryOp.sint(value, ">>", amount)
        else:
            raise DecompFailure(cls._unsupported_message(mnemonic, args))

        if isinstance(src, Register):
            inputs: List[Register] = [dst, src]

            def shift_amount(a: InstrArgs) -> Expression:
                return a.reg(1)

        elif isinstance(src, AsmLiteral):
            inputs = [dst]

            def shift_amount(_: InstrArgs, value: int = src.value) -> Literal:
                return Literal(value)

        else:
            raise DecompFailure(cls._unsupported_message(mnemonic, args))

        def eval_shift(state: NodeState, a: InstrArgs) -> None:
            amount = shift_amount(a)
            result = op_builder(a.reg(0), amount)
            state.set_reg(dst, result)

        return Instruction(
            mnemonic=mnemonic,
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
            eval_fn=eval_shift,
        )

    @classmethod
    def _parse_neg(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 1, "neg expects one operand"
        dst = args[0]
        if not isinstance(dst, Register):
            raise DecompFailure(cls._unsupported_message("neg", args))

        def eval_neg(state: NodeState, a: InstrArgs) -> None:
            result = BinaryOp.intptr(Literal(0), "-", a.reg(0))
            state.set_reg(dst, result)

        return Instruction(
            mnemonic="neg",
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
            eval_fn=eval_neg,
        )

    @classmethod
    def _parse_not(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 1, "not expects one operand"
        dst = args[0]
        if not isinstance(dst, Register):
            raise DecompFailure(cls._unsupported_message("not", args))

        def eval_not(state: NodeState, a: InstrArgs) -> None:
            result = BinaryOp.int(a.reg(0), "^", Literal(-1))
            state.set_reg(dst, result)

        return Instruction(
            mnemonic="not",
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
            eval_fn=eval_not,
        )

    @classmethod
    def _parse_jmp(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 1, "jmp expects one operand"
        target = args[0]
        inputs: List[Location] = []
        jump_target: Optional[Union[JumpTarget, Register]]

        if isinstance(target, Register):
            inputs = [target]
            jump_target = target
        else:
            jump_target = get_jump_target(target)

        return Instruction(
            mnemonic="jmp",
            args=args,
            meta=meta,
            inputs=inputs,
            clobbers=[],
            outputs=[],
            jump_target=jump_target,
            function_target=None,
            is_conditional=isinstance(target, Register),
            is_return=False,
            is_store=False,
            is_load=False,
            eval_fn=_no_op_eval,
        )

    @classmethod
    def _parse_and(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 2, "and expects two operands"
        dst, src = args
        if not isinstance(dst, Register):
            raise DecompFailure(cls._unsupported_message("and", args))

        if isinstance(src, Register):
            inputs = [dst, src]

            def eval_and(state: NodeState, a: InstrArgs) -> None:
                result = BinaryOp.int(a.reg(0), "&", a.reg(1))
                state.set_reg(dst, result)
                state.set_reg(Register("zf"), BinaryOp.icmp(result, "==", Literal(0)))

        elif isinstance(src, AsmLiteral):
            inputs = [dst]

            def eval_and(state: NodeState, a: InstrArgs) -> None:
                result = BinaryOp.int(a.reg(0), "&", Literal(src.value))
                state.set_reg(dst, result)
                state.set_reg(Register("zf"), BinaryOp.icmp(result, "==", Literal(0)))

        else:
            raise DecompFailure(cls._unsupported_message("and", args))

        return Instruction(
            mnemonic="and",
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
            eval_fn=eval_and,
        )

    @classmethod
    def _parse_or(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 2, "or expects two operands"
        dst, src = args
        if not isinstance(dst, Register):
            raise DecompFailure(cls._unsupported_message("or", args))

        if isinstance(src, Register):
            inputs = [dst, src]

            def eval_or(state: NodeState, a: InstrArgs) -> None:
                result = BinaryOp.int(a.reg(0), "|", a.reg(1))
                state.set_reg(dst, result)
                state.set_reg(Register("zf"), BinaryOp.icmp(result, "==", Literal(0)))

        elif isinstance(src, AsmLiteral):
            inputs = [dst]

            def eval_or(state: NodeState, a: InstrArgs) -> None:
                result = BinaryOp.int(a.reg(0), "|", Literal(src.value))
                state.set_reg(dst, result)
                state.set_reg(Register("zf"), BinaryOp.icmp(result, "==", Literal(0)))

        else:
            raise DecompFailure(cls._unsupported_message("or", args))

        return Instruction(
            mnemonic="or",
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
            eval_fn=eval_or,
        )

    @classmethod
    def _parse_test(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 2, "test expects two operands"
        lhs, rhs = args
        inputs: List[Location] = []
        lhs_inputs, lhs_value, lhs_is_load = cls._value_from_operand(
            "test",
            args,
            lhs,
            arg_index=0,
            allow_literal=False,
            allow_memory=True,
        )

        rhs_inputs, rhs_value, _ = cls._value_from_operand(
            "test",
            args,
            rhs,
            arg_index=1,
            allow_literal=True,
            allow_memory=False,
        )

        for loc in lhs_inputs + rhs_inputs:
            if loc not in inputs:
                inputs.append(loc)

        return Instruction(
            mnemonic="test",
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
            is_load=lhs_is_load,
            eval_fn=lambda state, a: cls._eval_test(state, lhs_value, rhs_value, a),
        )

    @staticmethod
    def _eval_test(
        state: NodeState,
        lhs_value: Callable[[InstrArgs], Expression],
        rhs_value: Callable[[InstrArgs], Expression],
        args: InstrArgs,
    ) -> None:
        value = BinaryOp.int(lhs_value(args), "&", rhs_value(args))
        zero = BinaryOp.icmp(value, "==", Literal(0))
        state.set_reg(Register("zf"), zero)

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
        inputs: List[Location] = list(cls.argument_regs)
        is_load = False

        if isinstance(target, Register):
            inputs.append(target)
            eval_fn = lambda s, a: s.make_function_call(a.reg(0), outputs)
        elif isinstance(target, (AsmGlobalSymbol, AsmLiteral)):
            eval_fn = lambda s, a: s.make_function_call(a.sym_imm(0), outputs)
        elif isinstance(target, AsmAddressMode):
            addr_inputs, addr_value, _ = cls._value_from_operand(
                "call",
                args,
                target,
                arg_index=0,
                allow_literal=False,
                allow_memory=True,
            )
            for loc in addr_inputs:
                if loc not in inputs:
                    inputs.append(loc)
            is_load = True

            def eval_fn(s: NodeState, a: InstrArgs, fn=addr_value) -> None:
                s.make_function_call(fn(a), outputs)
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
            is_load=is_load,
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
    def _parse_inc(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 1, "inc expects one operand"
        dst = args[0]
        if not isinstance(dst, Register):
            raise DecompFailure(cls._unsupported_message("inc", args))

        def eval_inc(state: NodeState, a: InstrArgs) -> None:
            reg_val = state.regs[dst]
            result = BinaryOp.intptr(reg_val, "+", Literal(1))
            state.set_reg(dst, result)
            state.set_reg(Register("zf"), BinaryOp.icmp(result, "==", Literal(0)))

        return Instruction(
            mnemonic="inc",
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
            eval_fn=eval_inc,
        )

    @classmethod
    def _parse_cmp(cls, args: List[Argument], meta: InstructionMeta) -> Instruction:
        assert len(args) == 2, "cmp expects two operands"
        lhs, rhs = args
        inputs: List[Location] = []

        lhs_inputs, lhs_value, _ = cls._value_from_operand(
            "cmp",
            args,
            lhs,
            arg_index=0,
            allow_literal=False,
            allow_memory=True,
        )
        if isinstance(lhs, AsmLiteral):
            raise DecompFailure(cls._unsupported_message("cmp", args))

        rhs_inputs, rhs_value, _ = cls._value_from_operand(
            "cmp",
            args,
            rhs,
            arg_index=1,
            allow_literal=True,
            allow_memory=True,
        )

        for loc in lhs_inputs + rhs_inputs:
            if loc not in inputs:
                inputs.append(loc)

        def eval_cmp(state: NodeState, a: InstrArgs) -> None:
            diff = BinaryOp.intptr(lhs_value(a), "-", rhs_value(a))
            zero = BinaryOp.icmp(diff, "==", Literal(0))
            state.set_reg(Register("zf"), zero)

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
    @staticmethod
    def _logical_and(lhs: Expression, rhs: Expression) -> BinaryOp:
        return BinaryOp(left=lhs, op="&&", right=rhs, type=Type.bool())

    @staticmethod
    def _logical_or(lhs: Expression, rhs: Expression) -> BinaryOp:
        return BinaryOp(left=lhs, op="||", right=rhs, type=Type.bool())

    @classmethod
    def _flag_value(cls, state: NodeState, flag: str) -> Expression:
        return state.regs[Register(flag)]

    @classmethod
    def _flag_is_set(cls, state: NodeState, flag: str) -> BinaryOp:
        return BinaryOp.icmp(cls._flag_value(state, flag), "!=", Literal(0))

    @classmethod
    def _flag_is_clear(cls, state: NodeState, flag: str) -> BinaryOp:
        return BinaryOp.icmp(cls._flag_value(state, flag), "==", Literal(0))

    @classmethod
    def _flags_compare(cls, state: NodeState, left: str, op: str, right: str) -> BinaryOp:
        return BinaryOp.icmp(cls._flag_value(state, left), op, cls._flag_value(state, right))

    @classmethod
    def _parse_conditional_jump(
        cls,
        args: List[Argument],
        meta: InstructionMeta,
        *,
        mnemonic: str,
        flag_inputs: List[Register],
        condition_builder: Callable[[NodeState], Expression],
    ) -> Instruction:
        assert len(args) == 1, "jump expects one operand"
        target = get_jump_target(args[0])

        def eval_jump(state: NodeState, _: InstrArgs) -> None:
            state.set_branch_condition(condition_builder(state))

        return Instruction(
            mnemonic=mnemonic,
            args=args,
            meta=meta,
            inputs=list(flag_inputs),
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

    @classmethod
    def _condition_spec(
        cls, mnemonic: str, *, allow_alias: bool = True
    ) -> Tuple[List[Register], CondBuilder]:
        spec = cls._conditional_jump_specs.get(mnemonic)
        if spec is not None:
            return spec
        if allow_alias:
            alias = cls._condition_aliases.get(mnemonic)
            if alias is not None:
                return cls._condition_spec(alias, allow_alias=False)
        raise DecompFailure(cls._unsupported_message(mnemonic, []))

    @classmethod
    def _parse_conditional_jump_mnemonic(
        cls, args: List[Argument], meta: InstructionMeta, *, mnemonic: str
    ) -> Instruction:
        flag_inputs, builder = cls._condition_spec(mnemonic)
        return cls._parse_conditional_jump(
            args,
            meta,
            mnemonic=mnemonic,
            flag_inputs=list(flag_inputs),
            condition_builder=lambda state, b=builder: b(cls, state),
        )

    @classmethod
    def _parse_setcc(
        cls, args: List[Argument], meta: InstructionMeta, *, mnemonic: str
    ) -> Instruction:
        assert len(args) == 1, "setcc expects one operand"
        dst = args[0]
        if not isinstance(dst, Register):
            raise DecompFailure(cls._unsupported_message(mnemonic, args))
        flag_inputs, builder = cls._condition_spec(mnemonic)

        def eval_set(state: NodeState, _: InstrArgs) -> None:
            cond = builder(cls, state)
            value = Cast(cond, reinterpret=False, silent=True, type=Type.intish())
            state.set_reg(dst, value)

        return Instruction(
            mnemonic=mnemonic,
            args=args,
            meta=meta,
            inputs=list(flag_inputs),
            clobbers=[],
            outputs=[dst],
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=False,
            is_store=False,
            is_load=False,
            eval_fn=eval_set,
        )

    @classmethod
    def _parse_rep_string(
        cls, prefix: str, args: List[Argument], meta: InstructionMeta
    ) -> Instruction:
        assert len(args) == 1 and isinstance(args[0], AsmGlobalSymbol), "rep expects mnemonic"
        op = args[0].symbol_name.lower()
        spec = cls._rep_string_specs.get((prefix, op))
        if spec is None:
            raise DecompFailure(cls._unsupported_message(prefix, args))

        return Instruction(
            mnemonic=f"{prefix}_{op}",
            args=args,
            meta=meta,
            inputs=list(spec["inputs"]),
            clobbers=[],
            outputs=list(spec["outputs"]),
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=False,
            is_store=spec["is_store"],
            is_load=spec["is_load"],
            eval_fn=_no_op_eval,
        )

    _conditional_jump_specs: Dict[str, Tuple[List[Register], CondBuilder]] = {
        "jz": (
            [Register("zf")],
            lambda cls, state: cls._flag_is_set(state, "zf"),
        ),
        "jnz": (
            [Register("zf")],
            lambda cls, state: cls._flag_is_clear(state, "zf"),
        ),
        "ja": (
            [Register("cf"), Register("zf")],
            lambda cls, state: cls._logical_and(
                cls._flag_is_clear(state, "cf"),
                cls._flag_is_clear(state, "zf"),
            ),
        ),
        "jl": (
            [Register("sf"), Register("of")],
            lambda cls, state: cls._flags_compare(state, "sf", "!=", "of"),
        ),
        "jle": (
            [Register("zf"), Register("sf"), Register("of")],
            lambda cls, state: cls._logical_or(
                cls._flag_is_set(state, "zf"),
                cls._flags_compare(state, "sf", "!=", "of"),
            ),
        ),
        "jg": (
            [Register("zf"), Register("sf"), Register("of")],
            lambda cls, state: cls._logical_and(
                cls._flag_is_clear(state, "zf"),
                cls._flags_compare(state, "sf", "==", "of"),
            ),
        ),
        "jge": (
            [Register("sf"), Register("of")],
            lambda cls, state: cls._flags_compare(state, "sf", "==", "of"),
        ),
        "jns": (
            [Register("sf")],
            lambda cls, state: cls._flag_is_clear(state, "sf"),
        ),
        "jc": (
            [Register("cf")],
            lambda cls, state: cls._flag_is_set(state, "cf"),
        ),
        "jnc": (
            [Register("cf")],
            lambda cls, state: cls._flag_is_clear(state, "cf"),
        ),
        "jbe": (
            [Register("cf"), Register("zf")],
            lambda cls, state: cls._logical_or(
                cls._flag_is_set(state, "cf"),
                cls._flag_is_set(state, "zf"),
            ),
        ),
    }

    _condition_aliases: Dict[str, str] = {
        "setz": "jz",
        "setnz": "jnz",
        "setg": "jg",
        "setge": "jge",
        "setl": "jl",
    }

    _rep_string_specs: Dict[Tuple[str, str], Dict[str, object]] = {
        (
            "rep",
            "movsd",
        ): {
            "inputs": [Register("esi"), Register("edi"), Register("ecx")],
            "outputs": [Register("esi"), Register("edi"), Register("ecx")],
            "is_load": True,
            "is_store": True,
        },
        (
            "rep",
            "movsb",
        ): {
            "inputs": [Register("esi"), Register("edi"), Register("ecx")],
            "outputs": [Register("esi"), Register("edi"), Register("ecx")],
            "is_load": True,
            "is_store": True,
        },
        (
            "rep",
            "stosd",
        ): {
            "inputs": [Register("edi"), Register("eax"), Register("ecx")],
            "outputs": [Register("edi"), Register("ecx")],
            "is_load": False,
            "is_store": True,
        },
        (
            "repne",
            "scasb",
        ): {
            "inputs": [Register("edi"), Register("ecx"), Register("eax")],
            "outputs": [Register("edi"), Register("ecx"), Register("zf")],
            "is_load": True,
            "is_store": False,
        },
    }

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
    "movsx": X86Arch._parse_movsx,
    "push": X86Arch._parse_push,
    "pop": X86Arch._parse_pop,
    "sub": X86Arch._parse_sub,
    "add": X86Arch._parse_add,
    "sbb": X86Arch._parse_sbb,
    "shl": lambda args, meta: X86Arch._parse_shift(args, meta, mnemonic="shl"),
    "sal": lambda args, meta: X86Arch._parse_shift(args, meta, mnemonic="sal"),
    "shr": lambda args, meta: X86Arch._parse_shift(args, meta, mnemonic="shr"),
    "sar": lambda args, meta: X86Arch._parse_shift(args, meta, mnemonic="sar"),
    "jmp": X86Arch._parse_jmp,
    "rep": lambda args, meta: X86Arch._parse_rep_string("rep", args, meta),
    "repne": lambda args, meta: X86Arch._parse_rep_string("repne", args, meta),
    "neg": X86Arch._parse_neg,
    "not": X86Arch._parse_not,
    "and": X86Arch._parse_and,
    "or": X86Arch._parse_or,
    "test": X86Arch._parse_test,
    "xor": X86Arch._parse_xor,
    "call": X86Arch._parse_call,
    "jz": lambda args, meta: X86Arch._parse_conditional_jump_mnemonic(args, meta, mnemonic="jz"),
    "jnz": lambda args, meta: X86Arch._parse_conditional_jump_mnemonic(args, meta, mnemonic="jnz"),
    "ja": lambda args, meta: X86Arch._parse_conditional_jump_mnemonic(args, meta, mnemonic="ja"),
    "jl": lambda args, meta: X86Arch._parse_conditional_jump_mnemonic(args, meta, mnemonic="jl"),
    "jle": lambda args, meta: X86Arch._parse_conditional_jump_mnemonic(args, meta, mnemonic="jle"),
    "jg": lambda args, meta: X86Arch._parse_conditional_jump_mnemonic(args, meta, mnemonic="jg"),
    "jge": lambda args, meta: X86Arch._parse_conditional_jump_mnemonic(args, meta, mnemonic="jge"),
    "jns": lambda args, meta: X86Arch._parse_conditional_jump_mnemonic(args, meta, mnemonic="jns"),
    "jc": lambda args, meta: X86Arch._parse_conditional_jump_mnemonic(args, meta, mnemonic="jc"),
    "jnc": lambda args, meta: X86Arch._parse_conditional_jump_mnemonic(args, meta, mnemonic="jnc"),
    "jbe": lambda args, meta: X86Arch._parse_conditional_jump_mnemonic(args, meta, mnemonic="jbe"),
    "setz": lambda args, meta: X86Arch._parse_setcc(args, meta, mnemonic="setz"),
    "setnz": lambda args, meta: X86Arch._parse_setcc(args, meta, mnemonic="setnz"),
    "setg": lambda args, meta: X86Arch._parse_setcc(args, meta, mnemonic="setg"),
    "setge": lambda args, meta: X86Arch._parse_setcc(args, meta, mnemonic="setge"),
    "setl": lambda args, meta: X86Arch._parse_setcc(args, meta, mnemonic="setl"),
    "cmp": X86Arch._parse_cmp,
    "lea": X86Arch._parse_lea,
    "dec": X86Arch._parse_dec,
    "inc": X86Arch._parse_inc,
}
