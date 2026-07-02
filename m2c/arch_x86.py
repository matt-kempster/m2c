"""i386 (x86) architecture support, for Ghidra-exported Intel-syntax asm.

Phase 1: registration, parsing, and structural instruction information
(inputs/outputs/jump targets). Instruction *semantics* (eval_fns) are not
implemented yet; every instruction's eval_fn raises DecompFailure, so
translation of x86 functions fails with a clear error while file parsing
and flow-graph-level analyses work.

Design notes:

- Operand widths: x86 encodes operand sizes both in memory operand prefixes
  ("byte ptr"/"word ptr"/"dword ptr"/"qword ptr") and in sub-register names
  (al/ah/ax/...). Both are canonicalized into a mnemonic suffix during
  parsing/normalization, ARM-style: `mov.b`, `mov.w`, `mov.q`. The default
  32-bit width has no suffix ("dword ptr" and plain 32-bit registers map to
  a bare mnemonic). This happens in two places:
    * `preprocess_instruction` strips "<size> ptr" from the argument string
      (before the generic argument parser runs) and appends the suffix;
    * `normalize_instruction` rewrites sub-register operands (al -> eax etc.)
      and appends the suffix derived from the narrowest sub-register, if the
      mnemonic doesn't already carry one.
  This keeps width information available in `X86Arch.parse` (and phase 2's
  eval functions) without extending the shared Argument types.

- Memory operands: Intel bracket expressions are parsed by the shared parser
  (gated by the `supports_intel_addressing` capability) into AsmAddressMode,
  with an Optional base register: `[esp + 0xc]` -> base=esp, addend=0xc;
  `[symbol]` -> base=None, addend=symbol; scaled indices stay in the addend
  as BinOp trees: `[esi + ebx*8 + 0x30]` -> base=esi,
  addend=(ebx * 8) + 0x30.

- Flags: mirrors ARM's condition flag scheme (z, n, c, v plus the composite
  hi/ge/gt pseudo-registers) so that phase 2 can reuse the eval_arm_cmp-style
  machinery. Note that after `cmp a, b`, x86's "ja" (CF=0 && ZF=0) has the
  same meaning as ARM's "hi" (unsigned greater-than), "jl" matches !ge, etc.
"""

from __future__ import annotations
import re
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from .error import DecompFailure
from .options import Target
from .asm_instruction import (
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmInstruction,
    AsmLiteral,
    AsmState,
    BinOp,
    JumpTarget,
    Register,
    get_jump_target,
    traverse_arg,
)
from .instruction import (
    Instruction,
    InstructionMeta,
    Location,
    StackLocation,
)
from .translate import (
    Abi,
    AbiArgSlot,
    ArgLoc,
    Arch,
    Expression,
    InstrArgs,
    NodeState,
    Type,
    as_type,
)
from .evaluate import fn_op
from .types import FunctionSignature


EAX = Register("eax")
ECX = Register("ecx")
EDX = Register("edx")
EBX = Register("ebx")
ESP = Register("esp")
EBP = Register("ebp")
ESI = Register("esi")
EDI = Register("edi")
EIP = Register("eip")

# Sub-register name -> (full register, width in bytes)
SUB_REGS: Dict[Register, Tuple[Register, int]] = {
    Register("al"): (EAX, 1),
    Register("ah"): (EAX, 1),
    Register("ax"): (EAX, 2),
    Register("bl"): (EBX, 1),
    Register("bh"): (EBX, 1),
    Register("bx"): (EBX, 2),
    Register("cl"): (ECX, 1),
    Register("ch"): (ECX, 1),
    Register("cx"): (ECX, 2),
    Register("dl"): (EDX, 1),
    Register("dh"): (EDX, 1),
    Register("dx"): (EDX, 2),
    Register("si"): (ESI, 2),
    Register("di"): (EDI, 2),
    Register("bp"): (EBP, 2),
    Register("sp"): (ESP, 2),
}

WIDTH_SUFFIXES: Dict[int, str] = {1: ".b", 2: ".w", 4: "", 8: ".q"}
PTR_WIDTHS: Dict[str, int] = {"byte": 1, "word": 2, "dword": 4, "qword": 8}

RE_PTR = re.compile(r"\b(byte|word|dword|qword)\s+ptr\s+", re.IGNORECASE)
RE_OFFSET = re.compile(r"\boffset\s+", re.IGNORECASE)
RE_ST_REG = re.compile(r"\bst\((\d)\)", re.IGNORECASE)
RE_SEGMENT = re.compile(r"\b([cdefgs]s):", re.IGNORECASE)


def split_width_suffix(mnemonic: str) -> Tuple[str, int]:
    """Split e.g. "mov.b" into ("mov", 1). No suffix means 4 bytes."""
    for width, suffix in WIDTH_SUFFIXES.items():
        if suffix and mnemonic.endswith(suffix):
            return mnemonic[: -len(suffix)], width
    return mnemonic, 4


class X86Arch(Arch):
    arch = Target.ArchEnum.X86

    re_comment = r"[#;].*"
    supports_dollar_regs = False
    supports_intel_addressing = True

    home_space_size = 0
    base_struct_align = 4

    stack_pointer_reg = ESP
    frame_pointer_regs = [EBP]
    return_address_reg = EIP

    base_return_regs = [(EAX, False)]
    all_return_regs = [EAX, EDX]
    argument_regs: List[Register] = []
    simple_temp_regs = [ECX, EDX]
    flag_regs = [Register(r) for r in ["n", "z", "c", "v", "hi", "ge", "gt"]]
    temp_regs = [EAX] + simple_temp_regs + flag_regs
    saved_regs = [EBX, ESI, EDI, EBP, EIP]
    # x87 FPU stack registers (untranslated in phase 1, but must parse)
    fpu_regs = [Register(f"st{i}") for i in range(8)]
    # Sub-registers are parsed as their own Register instances so that operand
    # widths survive until normalize_instruction, which rewrites them into
    # full registers plus a width-suffixed mnemonic.
    all_regs = (
        saved_regs + temp_regs + [stack_pointer_reg] + fpu_regs + list(SUB_REGS.keys())
    )

    aliased_regs: Dict[str, Register] = {}

    def missing_return(self) -> List[Instruction]:
        return [self.parse("ret", [], InstructionMeta.missing())]

    def preprocess_instruction(self, mnemonic: str, args: str) -> Tuple[str, str]:
        # Fold "<size> ptr" memory operand prefixes into the mnemonic as a
        # width suffix, and strip syntactic sugar the generic argument parser
        # should not see ("offset symbol" just means the symbol's address,
        # which is how bare symbols are treated anyway).
        widths = [PTR_WIDTHS[m.lower()] for m in RE_PTR.findall(args)]
        args = RE_PTR.sub("", args)
        args = RE_OFFSET.sub("", args)
        # Rewrite st(N) FPU registers into parseable names.
        args = RE_ST_REG.sub(lambda m: f"st{m.group(1)}", args)
        # Segment override prefixes (e.g. the fs:[0] accesses in SEH
        # prologues): move the segment into the mnemonic. The resulting
        # mnemonic (e.g. "mov.fs") is treated as an unknown instruction, which
        # parses fine structurally but fails translation with a clear error.
        segments = [m.lower() for m in RE_SEGMENT.findall(args)]
        args = RE_SEGMENT.sub("", args)
        for seg in segments:
            mnemonic += f".{seg}"
        if widths:
            # x86 has no instructions with two memory operands of different
            # widths, so all prefixes agree.
            mnemonic += WIDTH_SUFFIXES[min(widths)]
        return mnemonic, args

    def normalize_instruction(
        self, instr: AsmInstruction, asm_state: AsmState
    ) -> AsmInstruction:
        # rep/repne/repe prefixes: fold the string instruction into the
        # mnemonic ("rep movsd" -> "rep.movsd").
        if instr.mnemonic in ("rep", "repe", "repne", "repz", "repnz"):
            assert len(instr.args) == 1 and isinstance(instr.args[0], AsmGlobalSymbol)
            op = instr.args[0].symbol_name.lower()
            return AsmInstruction(f"{instr.mnemonic}.{op}", [])

        # Rewrite sub-register operands into full registers, deriving a width
        # suffix from the narrowest sub-register if the mnemonic does not
        # already carry one from a "<size> ptr" prefix.
        sub_width: Optional[int] = None

        def rewrite(arg: Argument) -> Argument:
            nonlocal sub_width
            if isinstance(arg, Register) and arg in SUB_REGS:
                full, width = SUB_REGS[arg]
                if sub_width is None or width < sub_width:
                    sub_width = width
                return full
            if isinstance(arg, AsmAddressMode):
                # (Sub-registers cannot appear in 32-bit address modes;
                # this is just defensive.)
                base = arg.base
                if base is not None and base in SUB_REGS:
                    base = SUB_REGS[base][0]
                return AsmAddressMode(base, arg.addend, arg.writeback)
            return arg

        new_args = [rewrite(arg) for arg in instr.args]
        mnemonic = instr.mnemonic
        base, width = split_width_suffix(mnemonic)
        if sub_width is not None and width == 4 and sub_width != 4:
            mnemonic = base + WIDTH_SUFFIXES[sub_width]
        if new_args != instr.args or mnemonic != instr.mnemonic:
            instr = AsmInstruction(mnemonic, new_args)
        return instr

    # Condition code -> (flag registers read, negated). The flags mirror
    # ARM's scheme; see the module docstring.
    _flag_z = Register("z")
    _flag_n = Register("n")
    _flag_c = Register("c")
    _flag_v = Register("v")
    _flag_hi = Register("hi")
    _flag_ge = Register("ge")
    _flag_gt = Register("gt")

    condition_flags: Dict[str, Tuple[Register, bool]] = {
        "z": (_flag_z, False),
        "e": (_flag_z, False),
        "nz": (_flag_z, True),
        "ne": (_flag_z, True),
        "s": (_flag_n, False),
        "ns": (_flag_n, True),
        # After `cmp a, b`, x86's carry flag means "borrow" (a < b unsigned),
        # which is the negation of ARM's carry ("no borrow"). The c pseudo-reg
        # follows ARM semantics, so jc/jb are its negation.
        "c": (_flag_c, True),
        "b": (_flag_c, True),
        "nae": (_flag_c, True),
        "nc": (_flag_c, False),
        "ae": (_flag_c, False),
        "nb": (_flag_c, False),
        "a": (_flag_hi, False),
        "nbe": (_flag_hi, False),
        "be": (_flag_hi, True),
        "na": (_flag_hi, True),
        "ge": (_flag_ge, False),
        "nl": (_flag_ge, False),
        "l": (_flag_ge, True),
        "nge": (_flag_ge, True),
        "g": (_flag_gt, False),
        "nle": (_flag_gt, False),
        "le": (_flag_gt, True),
        "ng": (_flag_gt, True),
        "o": (_flag_v, False),
        "no": (_flag_v, True),
    }

    # Structural instruction classification tables. Grouped by operand
    # behavior; parse() below interprets these.

    # dst is read and written (register or memory), src is read; sets flags.
    instrs_alu_rmw: Set[str] = {
        "add",
        "sub",
        "adc",
        "sbb",
        "and",
        "or",
        "xor",
        "shl",
        "sal",
        "shr",
        "sar",
        "rol",
        "ror",
        "shrd",
        "shld",
    }
    # dst is written only (not read), src is read; no flags.
    instrs_dst_write: Set[str] = {"mov", "movsx", "movzx", "lea"}
    # single operand, read and written; sets flags.
    instrs_unary_rmw: Set[str] = {"inc", "dec", "neg", "not", "bswap"}
    # two operands, both read; only flags written.
    instrs_cmp: Set[str] = {"cmp", "test"}
    # instructions with no operands and no structural effects.
    instrs_ignore: Set[str] = {"nop", "int3"}
    # rep-prefixed string instructions: mnemonic -> (inputs, outputs, load, store)
    instrs_string: Dict[str, Tuple[List[Register], List[Register], bool, bool]] = {
        "rep.movsd": ([ESI, EDI, ECX], [ESI, EDI, ECX], True, True),
        "rep.movsw": ([ESI, EDI, ECX], [ESI, EDI, ECX], True, True),
        "rep.movsb": ([ESI, EDI, ECX], [ESI, EDI, ECX], True, True),
        "rep.stosd": ([EDI, EAX, ECX], [EDI, ECX], False, True),
        "rep.stosw": ([EDI, EAX, ECX], [EDI, ECX], False, True),
        "rep.stosb": ([EDI, EAX, ECX], [EDI, ECX], False, True),
        "repne.scasb": ([EDI, ECX, EAX], [EDI, ECX, Register("z")], True, False),
        "repe.cmpsb": ([ESI, EDI, ECX], [ESI, EDI, ECX, Register("z")], True, False),
    }

    @classmethod
    def _unsupported_eval(
        cls, instr_str: str
    ) -> Callable[[NodeState, InstrArgs], object]:
        def eval_fn(s: NodeState, a: InstrArgs) -> None:
            raise DecompFailure(
                f"x86 instruction evaluation is not implemented yet: {instr_str}"
            )

        return eval_fn

    @classmethod
    def _stack_location(cls, addr: AsmAddressMode) -> Optional[StackLocation]:
        if addr.base == cls.stack_pointer_reg:
            return StackLocation.from_offset(addr.addend)
        return None

    @classmethod
    def _operand_inputs(cls, arg: Argument) -> List[Location]:
        """Locations read in order to evaluate `arg` (for memory operands:
        the registers making up the address, plus the stack location for
        esp-relative addresses, since the operand value is also read)."""
        inputs: List[Location] = []
        for sub in traverse_arg(arg):
            if isinstance(sub, Register) and sub not in inputs:
                inputs.append(sub)
        if isinstance(arg, AsmAddressMode):
            stack_loc = cls._stack_location(arg)
            if stack_loc is not None:
                inputs.append(stack_loc)
        return inputs

    @classmethod
    def _address_regs(cls, arg: AsmAddressMode) -> List[Location]:
        """Registers making up a memory operand's address (the operand's
        value itself is not read)."""
        return [sub for sub in traverse_arg(arg) if isinstance(sub, Register)]

    def parse(
        self, mnemonic: str, args: List[Argument], meta: InstructionMeta
    ) -> Instruction:
        cls = type(self)
        inputs: List[Location] = []
        clobbers: List[Location] = []
        outputs: List[Location] = []
        jump_target: Optional[Union[JumpTarget, Register, List[JumpTarget]]] = None
        function_target: Optional[Argument] = None
        is_conditional = False
        is_return = False
        is_load = False
        is_store = False
        is_effectful = True

        instr_str = str(AsmInstruction(mnemonic, args))
        eval_fn: Optional[Callable[[NodeState, InstrArgs], object]] = (
            cls._unsupported_eval(instr_str)
        )

        base, width = split_width_suffix(mnemonic)

        def add_inputs(arg: Argument) -> None:
            for loc in cls._operand_inputs(arg):
                if loc not in inputs:
                    inputs.append(loc)

        def dest_operand(arg: Argument, *, also_read: bool) -> None:
            """Handle a destination operand (register or memory)."""
            nonlocal is_store
            if isinstance(arg, Register):
                outputs.append(arg)
                if also_read and arg not in inputs:
                    inputs.append(arg)
            elif isinstance(arg, AsmAddressMode):
                is_store = True
                for loc in cls._address_regs(arg):
                    if loc not in inputs:
                        inputs.append(loc)
                stack_loc = cls._stack_location(arg)
                if stack_loc is not None:
                    outputs.append(stack_loc)
                    if also_read and stack_loc not in inputs:
                        inputs.append(stack_loc)
            else:
                raise DecompFailure(f"Invalid x86 destination operand in `{instr_str}`")

        def src_operand(arg: Argument) -> None:
            nonlocal is_load
            if isinstance(arg, AsmAddressMode):
                is_load = True
            add_inputs(arg)

        if base == "ret":
            assert len(args) <= 1, "ret takes at most one (immediate) operand"
            inputs = [cls.stack_pointer_reg]
            is_return = True
        elif base == "jmp":
            assert len(args) == 1
            target = args[0]
            if isinstance(target, Register):
                # Indirect jump (jump table); phase 2 will resolve targets.
                inputs = [target]
                jump_target = target
                is_conditional = True
            elif isinstance(target, AsmAddressMode):
                src_operand(target)
                regs = [loc for loc in inputs if isinstance(loc, Register)]
                if regs:
                    # Jump through memory, e.g. `jmp [eax*4 + switchdata]`.
                    # Treat like an indirect jump through the index register.
                    jump_target = regs[0]
                    is_conditional = True
                else:
                    # Register-less jump through an absolute address, e.g.
                    # `jmp [__imp__GetTickCount]`: a tail call through an
                    # import thunk.
                    outputs = list(cls.all_return_regs)
                    function_target = target
                    is_return = True
            else:
                jump_target = get_jump_target(target)
        elif base.startswith("j") and base[1:] in cls.condition_flags:
            assert len(args) == 1
            flag, _negated = cls.condition_flags[base[1:]]
            inputs = [flag]
            jump_target = get_jump_target(args[0])
            is_conditional = True
        elif base == "loop":
            assert len(args) == 1
            inputs = [ECX]
            outputs = [ECX]
            jump_target = get_jump_target(args[0])
            is_conditional = True
        elif base == "call":
            assert len(args) == 1
            target = args[0]
            inputs = list(cls.argument_regs)
            outputs = list(cls.all_return_regs)
            clobbers = list(cls.temp_regs)
            function_target = target
            if isinstance(target, Register):
                inputs.append(target)
            elif isinstance(target, AsmAddressMode):
                src_operand(target)
            elif not isinstance(target, (AsmGlobalSymbol, AsmLiteral)):
                raise DecompFailure(f"Invalid x86 call target in `{instr_str}`")
        elif base == "push":
            assert len(args) == 1
            inputs = [cls.stack_pointer_reg]
            src_operand(args[0])
            outputs = [cls.stack_pointer_reg]
            is_store = True
        elif base == "pop":
            assert len(args) == 1
            inputs = [cls.stack_pointer_reg]
            dest_operand(args[0], also_read=False)
            outputs.append(cls.stack_pointer_reg)
            is_load = True
        elif base == "pushad":
            inputs = [cls.stack_pointer_reg, EAX, ECX, EDX, EBX, EBP, ESI, EDI]
            outputs = [cls.stack_pointer_reg]
            is_store = True
        elif base == "popad":
            inputs = [cls.stack_pointer_reg]
            outputs = [EAX, ECX, EDX, EBX, EBP, ESI, EDI, cls.stack_pointer_reg]
            is_load = True
        elif base in cls.instrs_dst_write:
            assert len(args) == 2
            dst, src = args
            src_operand(src)
            if base == "lea":
                # lea only computes the address; it does not load from it.
                is_load = False
            dest_operand(dst, also_read=False)
            is_effectful = is_store
        elif base in cls.instrs_alu_rmw:
            assert len(args) in (2, 3)  # shrd/shld take three operands
            dst = args[0]
            for src in args[1:]:
                src_operand(src)
            dest_operand(dst, also_read=True)
            outputs.extend(cls.flag_regs)
            is_effectful = is_store
        elif base in cls.instrs_unary_rmw:
            assert len(args) == 1
            dest_operand(args[0], also_read=True)
            if isinstance(args[0], AsmAddressMode):
                is_load = True
            elif isinstance(args[0], Register) and args[0] not in inputs:
                inputs.append(args[0])
            if base not in ("not", "bswap"):
                outputs.extend(cls.flag_regs)
            is_effectful = is_store
        elif base in cls.instrs_cmp:
            assert len(args) == 2
            for arg in args:
                src_operand(arg)
            outputs = list(cls.flag_regs)
            is_effectful = False
        elif base.startswith("set") and base[3:] in cls.condition_flags:
            assert len(args) == 1
            flag, _negated = cls.condition_flags[base[3:]]
            inputs = [flag]
            dest_operand(args[0], also_read=False)
            is_effectful = is_store
        elif base == "cdq":
            assert not args
            inputs = [EAX]
            outputs = [EDX]
            is_effectful = False
        elif base in ("mul", "imul", "div", "idiv") and len(args) <= 1:
            # One-operand forms operate on edx:eax.
            inputs = [EAX] if base in ("mul", "imul") else [EAX, EDX]
            if args:
                src_operand(args[0])
            outputs = [EAX, EDX, *cls.flag_regs]
            is_effectful = False
        elif base == "imul":
            # Two/three-operand forms only write the destination register.
            assert len(args) in (2, 3) and isinstance(args[0], Register)
            inputs = [args[0]] if len(args) == 2 else []
            for src in args[1:]:
                src_operand(src)
            outputs = [args[0], *cls.flag_regs]
            is_effectful = False
        elif base == "xchg":
            assert len(args) == 2
            for arg in args:
                src_operand(arg)
            for arg in args:
                dest_operand(arg, also_read=True)
            is_effectful = is_store
        elif mnemonic in cls.instrs_string:
            str_inputs, str_outputs, is_load, is_store = cls.instrs_string[mnemonic]
            inputs = list(str_inputs)
            outputs = list(str_outputs)
        elif base in cls.instrs_ignore:
            is_effectful = False
            eval_fn = None
        else:
            # Unknown instruction (x87 FPU, etc.). Guess a structural shape
            # so that file parsing and flow graph construction still work;
            # evaluation will fail with a clear error.
            for arg in args:
                add_inputs(arg)
            if args and isinstance(args[0], Register):
                inputs = [loc for loc in inputs if loc != args[0]]
                outputs = [args[0]]

        return Instruction(
            mnemonic=mnemonic,
            args=args,
            meta=meta,
            inputs=inputs,
            clobbers=clobbers,
            outputs=outputs,
            jump_target=jump_target,
            function_target=function_target,
            is_conditional=is_conditional,
            is_return=is_return,
            is_store=is_store,
            is_load=is_load,
            is_effectful=is_effectful,
            eval_fn=eval_fn,
        )

    def default_function_abi_candidate_slots(self) -> List[AbiArgSlot]:
        return []

    def arg_name(self, loc: ArgLoc) -> str:
        if loc.offset is not None:
            return f"arg_{loc.offset:x}"
        assert loc.reg is not None
        return loc.reg.register_name

    def function_abi(
        self,
        fn_sig: FunctionSignature,
        likely_regs: Dict[Register, bool],
        *,
        for_call: bool,
    ) -> Abi:
        """cdecl: all arguments are passed on the stack, pushed right to left."""
        known_slots: List[AbiArgSlot] = []
        offset = 0
        if fn_sig.params_known:
            for i, param in enumerate(fn_sig.params):
                param_type = param.type.decay()
                size, align = param_type.get_parameter_size_align_bytes()
                size = (size + 3) & ~3
                offset = (offset + align - 1) & -align
                known_slots.append(
                    AbiArgSlot(
                        ArgLoc(offset, i, None),
                        param_type,
                        name=param.name,
                    )
                )
                offset += size
        candidate_slots: List[AbiArgSlot] = []
        if not fn_sig.params_known or fn_sig.is_variadic:
            for i in range(8):
                candidate_slots.append(
                    AbiArgSlot(
                        ArgLoc(offset + 4 * i, len(known_slots) + i, None),
                        Type.any_reg(),
                    )
                )
        return Abi(arg_slots=known_slots, possible_slots=candidate_slots)

    def function_return(self, expr: Expression) -> Dict[Register, Expression]:
        # Return values are in eax, with edx holding the high half of u64's.
        return {
            EAX: as_type(expr, Type.intptr(), silent=True, unify=False),
            EDX: fn_op("SECOND_REG", [expr], Type.reg32(likely_float=False)),
        }
