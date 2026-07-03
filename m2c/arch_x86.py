"""i386 (x86) architecture support, for Ghidra-exported Intel-syntax asm.

Phase 1 established registration, parsing, and structural instruction
information (inputs/outputs/jump targets). Phase 2a adds real semantics
(eval_fns) for data operations, flags, and conditionals: mov/movsx/movzx/lea/
xchg, the ALU (incl. mul/imul/div/idiv/cdq and shifts), cmp/test, all jcc and
setcc, and loads/stores through all addressing modes. Phase 2b adds the
moving stack: an ESP-delta prepass (X86StackRewritePattern / rewrite_stack_ops)
computes esp's offset from function entry at every instruction and rewrites
push/pop/call-argument/ebp-frame accesses into fixed frame offsets, so the
rest of m2c (which assumes a constant post-prologue stack pointer) works
unchanged. This recovers call arguments (cdecl and stdcall), tail calls, and
jump-table switches, plus rep string ops, loop, and rdtsc. Phase 3 adds x87
FPU support via a second whole-body prepass (X86FpuRewritePattern in
m2c/x86_fpu.py) that eliminates the FPU register stack into flat virtual
registers f0..f7, with the per-instruction semantics in X86Arch._parse_fpu
(float arithmetic/compares/conversions, the fnstsw/test-ah compare idiom, and
the float call ABI: returns, per-callee stack deltas, and float arguments).

The ESP-delta design: a linear dataflow pass over the flow graph tracks
(esp_delta, ebp_delta) per instruction (push/pop = ∓4, sub/add esp = ∓N,
call = +callee_cleanup, `mov ebp, esp` binds ebp to the frame). The frame is a
fixed region sized to the maximum stack depth; a synthetic `sub esp, frame`
at entry makes get_stack_info see it. `[esp+k]`/`[ebp+k]` at delta d become
`[esp + k + frame + d]`; saved-register push/pop become plain stores/loads
(callee-saved); other pushes become `storearg.fictive` feeding the next call;
each call is annotated with its argument-region base and byte count so
translation splits pending stack arguments across nested calls. See
rewrite_stack_ops for the full rewrite.

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
  This keeps width information available in `X86Arch.parse` (and the eval
  functions) without extending the shared Argument types.

- Sub-register writes: an instruction that writes a sub-register (`mov.b` to
  cl, setcc, a byte load, ...) is deliberately modeled as writing the *full*
  storage register with a partial-width-typed value. This is correct for the
  overwhelmingly common patterns (`xor reg, reg` + byte load, setcc after
  clearing the register, byte loads that feed byte stores) but loses the
  upper bits of the old register value in the rare cases where they are
  live across the sub-register write.

- Memory operands: Intel bracket expressions are parsed by the shared parser
  (gated by the `supports_intel_addressing` capability) into AsmAddressMode,
  with an Optional base register: `[esp + 0xc]` -> base=esp, addend=0xc;
  `[symbol]` -> base=None, addend=symbol; scaled indices stay in the addend
  as BinOp trees: `[esi + ebx*8 + 0x30]` -> base=esi,
  addend=(ebx * 8) + 0x30. During evaluation these become either an
  AddressMode (base + literal offset; handles stack accesses), a RawSymbolRef
  (absolute [symbol + off]), or a generic address Expression whose shape
  (base + index * scale + offset) `deref`/`array_access_from_add` recognize
  and turn into array indexing.

- Flags: mirrors ARM's condition flag scheme (z, n, c, v plus the composite
  hi/ge/gt pseudo-registers): flag-setting instructions store complete
  symbolic conditions into the flag registers, and consumers (jcc/setcc/
  adc/sbb) read them back, negating via Condition.negated() where needed.
  One crucial difference from ARM: after `cmp a, b` (or sub/neg), x86's
  carry flag is a *borrow*: c = (u32)a < (u32)b, the inverse of ARM's carry.
  After additions, c is the carry-out (same as ARM). See eval_x86_cmp in
  evaluate.py and `condition_flags` below.

- Arguments: cdecl passes all arguments on the stack. At function entry
  [esp + 0] holds the return address, so with an unmoved stack pointer the
  arguments live at [esp + 4], [esp + 8], ... A literal `sub esp, N` in the
  prologue is folded into StackInfo.allocated_stack_size (see
  get_stack_info), moving the argument region to [esp + N + 4]. Arbitrary
  mid-function esp adjustment (push/pop in particular) is not modeled yet.
"""

from __future__ import annotations
import re
import struct
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from .error import DecompFailure
from .options import Target
from .asm_file import AsmData, AsmDataEntry, AsmSymbolicData, Label
from .c_types import CType, TypeMap
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
from .asm_pattern import AsmMatcher, AsmPattern, BodyPart, Replacement
from .instruction import (
    ArchAsm,
    Instruction,
    InstructionMeta,
    Location,
    StackLocation,
)
from .translate import (
    Abi,
    AbiArgSlot,
    AddressMode,
    ArgLoc,
    Arch,
    BinaryOp,
    Cast,
    Condition,
    Expression,
    FuncCall,
    GlobalSymbol,
    InstrArgs,
    Literal,
    NodeState,
    RawSymbolRef,
    RegExpression,
    StoreStmt,
    Type,
    UnaryOp,
    as_intish,
    as_type,
    early_unwrap,
    parse_symbol_ref,
)
from .evaluate import (
    condition_from_expr,
    deref,
    eval_x86_cmp,
    fn_op,
    fold_divmod,
    fold_mul_chains,
    handle_add_real,
    handle_addi_real,
    handle_bitinv,
    handle_convert,
    handle_or,
    handle_sub,
    make_store_real,
    replace_bitand,
    set_x86_flags_from_add,
    set_x86_flags_from_result,
    shift_right_expr,
    void_fn_op,
)
from .types import FunctionSignature
from .x86_fpu import X86FpuRewritePattern


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


def width_type(width: int) -> Type:
    """The Type used for loads/stores/casts of a given operand width.
    Sub-32-bit widths use sign-ambiguous types so that unification with
    surrounding code can decide signedness."""
    if width == 4:
        return Type.reg32(likely_float=False)
    return Type.int_of_size(width * 8)


def sign_extended_imm(value: int, width: int) -> Literal:
    """Interpret an immediate as a signed value of the given operand width
    (x86 immediates for arithmetic/compares are sign-extended)."""
    bits = width * 8
    value &= (1 << bits) - 1
    if value >= 1 << (bits - 1):
        value -= 1 << bits
    return Literal(value)


def address_expr(arg: Argument, a: InstrArgs) -> Expression:
    """Convert (part of) an Intel address-mode addend into an Expression."""
    if isinstance(arg, Register):
        return a.regs[arg]
    if isinstance(arg, AsmLiteral):
        return Literal(arg.value)
    if isinstance(arg, AsmGlobalSymbol):
        return a.stack_info.global_info.address_of_gsym(arg.symbol_name)
    if isinstance(arg, BinOp) and arg.op in ("+", "-", "*", "<<"):
        lhs = address_expr(arg.lhs, a)
        rhs = address_expr(arg.rhs, a)
        if arg.op in ("+", "-"):
            return BinaryOp.intptr(lhs, arg.op, rhs)
        return BinaryOp.int(lhs, arg.op, rhs)
    raise DecompFailure(f"Unsupported x86 address expression: {arg}")


def mem_target(a: InstrArgs, index: int) -> Union[AddressMode, RawSymbolRef, Expression]:
    """Compute the target of a memory operand, as either an AddressMode
    (base register + literal offset, which also handles esp-relative stack
    accesses), a RawSymbolRef (absolute [symbol + offset]), or a generic
    address Expression (scaled-index modes, which deref turns into array
    accesses)."""
    arg = a.raw_arg(index)
    assert isinstance(arg, AsmAddressMode), f"expected a memory operand, found {arg}"
    if arg.base is not None and isinstance(arg.addend, AsmLiteral):
        return AddressMode(offset=arg.addend.value, base=arg.base)
    if arg.base is None:
        ref = parse_symbol_ref(arg.addend)
        if ref is not None:
            return ref
    addend = address_expr(arg.addend, a)
    if arg.base is None:
        return addend
    return BinaryOp.intptr(a.regs[arg.base], "+", addend)


def mem_load(a: InstrArgs, index: int, type: Type) -> Expression:
    size = type.get_size_bytes()
    assert size is not None
    target = mem_target(a, index)
    expr = deref(target, a.regs, a.stack_info, size=size)
    return as_type(expr, type, silent=True)


def mem_store(
    a: InstrArgs,
    index: int,
    value: Expression,
    value_reg: Optional[Register],
    type: Type,
) -> Optional[StoreStmt]:
    size = type.get_size_bytes()
    assert size is not None
    target = mem_target(a, index)
    source_raw: Optional[RegExpression] = None
    if value_reg is not None:
        source_raw = a.regs.get_raw(value_reg)
    if isinstance(target, (AddressMode, RawSymbolRef)):
        return make_store_real(value, source_raw, target, a.regs, a.stack_info, type)
    dest = deref(target, a.regs, a.stack_info, size=size, store=True)
    dest.type.unify(type)
    return StoreStmt(source=as_type(value, type, silent=False), dest=dest)


def op_value(
    a: InstrArgs,
    index: int,
    width: int,
    *,
    type: Optional[Type] = None,
    sign_extend_imm: bool = True,
) -> Expression:
    """Read an operand's value (register, memory, or immediate/symbol)."""
    arg = a.raw_arg(index)
    if isinstance(arg, Register):
        val = a.regs[arg]
        if width < 4:
            # Reading a sub-register: reinterpret the low bits.
            val = as_type(val, type or width_type(width), silent=True, unify=False)
        return val
    if isinstance(arg, AsmAddressMode):
        return mem_load(a, index, type or width_type(width))
    imm = a.full_imm(index)
    if isinstance(imm, Literal) and sign_extend_imm:
        return sign_extended_imm(imm.value, width)
    return imm


def sub_expr(lhs: Expression, rhs: Expression) -> Expression:
    val = handle_sub(lhs, rhs)
    if isinstance(val, BinaryOp):
        val = fold_divmod(val)
    return fold_mul_chains(val)


def carry_in(a: InstrArgs) -> Expression:
    """The x86 carry flag as a 0/1 integer expression (for adc/sbb)."""
    return condition_from_expr(a.regs[Register("c")])


def adc_expr(a: InstrArgs, lhs: Expression, srcs: List[Expression]) -> Expression:
    return handle_add_real(handle_add_real(lhs, srcs[0], a), carry_in(a), a)


def sbb_expr(a: InstrArgs, lhs: Expression, srcs: List[Expression]) -> Expression:
    if a.raw_arg(0) == a.raw_arg(1):
        # sbb r, r: idiom for materializing the carry (borrow) flag as
        # 0 / -1 without branching. After `cmp a, b` the carry is the unsigned
        # borrow (a < b), so this is -(a < b); `neg`/`inc` of it (below)
        # recover the plain (in)equality.
        return UnaryOp("-", carry_in(a), type=Type.intish())
    return BinaryOp.intptr(handle_sub(lhs, srcs[0]), "-", carry_in(a))


def neg_expr(v: Expression) -> Expression:
    """`neg`: arithmetic negation, collapsing a double negation. Applied to the
    `sbb r,r` idiom (which materializes a compare's borrow as -(cond)), this
    recovers the plain boolean, so `cmp a,b; sbb r,r; neg r` reads as `a < b`
    rather than `--(a < b)`. -(-x) == x holds for every two's-complement
    integer, so the fold is always valid, not only for the 0/-1 boolean."""
    uw = early_unwrap(v)
    if isinstance(uw, UnaryOp) and uw.op == "-" and not uw.expr.type.is_float():
        return uw.expr
    return UnaryOp.sint("-", v)


def inc_expr(a: InstrArgs, v: Expression) -> Expression:
    """`inc`. Applied to the negated boolean from `sbb r,r` (-(cond)), inc
    computes 1 + -(cond) = !cond (cond is 0 or 1), so `cmp a,b; sbb r,r; inc r`
    reads as the negated comparison (e.g. `a >= b`) instead of `-(a < b) + 1`.
    Only fires for a negated comparison, where 1 - cond == !cond holds."""
    uw = early_unwrap(v)
    if isinstance(uw, UnaryOp) and uw.op == "-":
        inner = early_unwrap(uw.expr)
        if isinstance(inner, BinaryOp) and inner.is_comparison():
            return inner.negated()
    return handle_add_real(v, Literal(1), a)


def shrd_expr(a: InstrArgs, lhs: Expression, srcs: List[Expression]) -> Expression:
    src, amt = srcs
    if isinstance(amt, Literal) and 0 < amt.value < 32:
        return BinaryOp.int(
            BinaryOp.uint(lhs, ">>", amt),
            "|",
            fold_mul_chains(BinaryOp.int(src, "<<", Literal(32 - amt.value))),
        )
    return fn_op("M2C_SHRD", [lhs, src, amt], Type.u32())


def shld_expr(a: InstrArgs, lhs: Expression, srcs: List[Expression]) -> Expression:
    src, amt = srcs
    if isinstance(amt, Literal) and 0 < amt.value < 32:
        return BinaryOp.int(
            fold_mul_chains(BinaryOp.int(lhs, "<<", amt)),
            "|",
            BinaryOp.uint(src, ">>", Literal(32 - amt.value)),
        )
    return fn_op("M2C_SHLD", [lhs, src, amt], Type.u32())


# Ghidra encodes stdcall decoration in symbol names: `_name@8` becomes
# `__imp__name_8` for import thunks. The trailing number is the number of
# argument bytes the callee pops on return.
RE_STDCALL_IMPORT = re.compile(r"^__imp__.*_(\d+)$")


# Undecorated stdcall imports appear as `_<LIB>_DLL_<Func>` or
# `_<LIB>_DRV_<Func>` (e.g. `_USER32_DLL_ShowWindow`), Ghidra's naming for
# imports whose thunks it resolved to a library.
RE_DLL_IMPORT = re.compile(r"^_[A-Za-z0-9]+_(?:DLL|DRV)_(\w+)$")


# Argument byte counts for well-known stdcall Win32/multimedia/DirectX APIs.
# MSVC calls these through undecorated thunk symbols (`call _MessageBoxA`),
# so the `@N` suffix that normally identifies callee cleanup is missing; this
# table restores it. The values are the documented argument counts * 4 (all
# arguments of these APIs are 4-byte). Names that are cdecl despite living in
# system DLLs (e.g. the variadic wsprintfA) must not be listed here.
#
# This built-in table is a Win32 convenience layer, and deliberately the
# lowest-priority *name-based* source of callee cleanup. callee_cleanup_bytes()
# resolves a call's cleanup with the precedence:
#   1. explicit context/prototype (__attribute__((stdcall)) in the user
#      context, i.e. arch.context_stdcall_arg_bytes),
#   2. a decorated stdcall suffix (`__imp__X_N`, or a Ghidra `.set name@N`
#      recorded in asm_data.stdcall_arg_bytes),
#   3. this configured platform table,
#   4. conservative structural inference (compute_call_cleanup, when the name
#      gives no answer).
# So a user context declaration always overrides a table entry for the same
# API, and the table only fills in APIs the context did not describe.
STDCALL_API_ARG_BYTES: Dict[str, int] = {
    "AddFontResourceA": 4,
    "AdjustWindowRect": 12,
    "ClientToScreen": 8,
    "CloseHandle": 4,
    "CoCreateInstance": 20,
    "CoInitialize": 4,
    "CoUninitialize": 0,
    "CreateCompatibleDC": 4,
    "CreateDCA": 16,
    "CreateDIBitmap": 24,
    "CreateEventA": 16,
    "CreateFileA": 28,
    "CreateFontIndirectA": 4,
    "CreateMutexA": 12,
    "CreatePen": 12,
    "CreateRectRgn": 16,
    "CreateRectRgnIndirect": 4,
    "CreateSolidBrush": 4,
    "CreateThread": 24,
    "CreateWindowExA": 48,
    "DefWindowProcA": 16,
    "DeleteDC": 4,
    "DeleteObject": 4,
    "DestroyWindow": 4,
    "DeviceIoControl": 32,
    "DialogBoxParamA": 20,
    "DispatchMessageA": 4,
    "DrawTextA": 20,
    "EndDialog": 8,
    "EndDoc": 4,
    "EndPage": 4,
    "EnumPrintersA": 28,
    "FileTimeToDosDateTime": 12,
    "FileTimeToLocalFileTime": 8,
    "FillRect": 12,
    "FreeLibrary": 4,
    "GetClientRect": 8,
    "GetComputerNameA": 8,
    "GetDesktopWindow": 0,
    "GetDeviceCaps": 8,
    "GetDlgItem": 8,
    "GetDriveTypeA": 4,
    "GetFileSize": 8,
    "GetFileTime": 16,
    "GetKeyState": 4,
    "GetLastError": 0,
    "GetLogicalDrives": 0,
    "GetMessageA": 16,
    "GetModuleFileNameA": 12,
    "GetNearestColor": 8,
    "GetStockObject": 4,
    "GetSystemInfo": 4,
    "GetSystemTimeAsFileTime": 4,
    "GetTickCount": 0,
    "GetUserNameA": 8,
    "GetVolumeInformationA": 32,
    "GetWindowLongA": 8,
    "GlobalAlloc": 8,
    "GlobalFree": 4,
    "GlobalLock": 4,
    "GlobalMemoryStatus": 4,
    "GlobalUnlock": 4,
    "IntersectRect": 12,
    "IsBadReadPtr": 8,
    "LineTo": 12,
    "LoadCursorA": 8,
    "LoadIconA": 8,
    "LoadLibraryA": 4,
    "LoadLibraryExA": 12,
    "LocalAlloc": 8,
    "LocalFree": 4,
    "MessageBoxA": 16,
    "MoveToEx": 16,
    "MulDiv": 12,
    "MultiByteToWideChar": 24,
    "OffsetRect": 12,
    "OutputDebugStringA": 4,
    "PeekMessageA": 20,
    "PostQuitMessage": 4,
    "PtInRect": 12,
    "QueryPerformanceCounter": 4,
    "QueryPerformanceFrequency": 4,
    "ReadFile": 20,
    "RegisterClassExA": 4,
    "RemoveFontResourceA": 4,
    "ResetEvent": 4,
    "ResumeThread": 4,
    "SelectObject": 8,
    "SendMessageA": 16,
    "SetBkColor": 8,
    "SetBkMode": 8,
    "SetCurrentDirectoryA": 4,
    "SetCursor": 4,
    "SetEvent": 4,
    "SetFilePointer": 16,
    "SetTextAlign": 8,
    "SetTextColor": 8,
    "ShowCursor": 4,
    "ShowWindow": 8,
    "Sleep": 4,
    "StartDocA": 8,
    "StartPage": 4,
    "StretchDIBits": 52,
    "SuspendThread": 4,
    "SystemParametersInfoA": 16,
    "TerminateThread": 8,
    "TextOutA": 20,
    "TranslateMessage": 4,
    "VirtualQuery": 12,
    "WaitForMultipleObjects": 16,
    "WaitForSingleObject": 8,
    "WaitMessage": 0,
    "WriteFile": 20,
    "lstrcpyA": 8,
    "lstrlenA": 4,
    "wvsprintfA": 12,
    # winmm
    "midiOutClose": 4,
    "midiOutOpen": 20,
    "midiOutShortMsg": 8,
    "timeKillEvent": 4,
    "timeSetEvent": 20,
    # DirectX / codec DLLs (called via `_<LIB>_DLL_<Func>` names)
    "DirectDrawCreate": 12,
    "DirectInputCreateA": 16,
    "DirectSoundCreate": 12,
    "GetFileVersionInfoA": 16,
    "GetFileVersionInfoSizeA": 8,
    "VerQueryValueA": 16,
    "AVIFileExit": 0,
    "AVIFileGetStream": 16,
    "AVIFileInfoA": 12,
    "AVIFileInit": 0,
    "AVIFileOpenA": 16,
    "AVIFileRelease": 4,
    "AVIStreamAddRef": 4,
    "AVIStreamGetFrame": 8,
    "AVIStreamGetFrameClose": 4,
    "AVIStreamGetFrameOpen": 8,
    "AVIStreamInfoA": 12,
    "AVIStreamLength": 4,
    "AVIStreamRead": 28,
    "AVIStreamReadFormat": 16,
    "AVIStreamRelease": 4,
    "AVIStreamStart": 4,
    "acmStreamClose": 8,
    "acmStreamConvert": 12,
    "acmStreamOpen": 32,
    "acmStreamPrepareHeader": 12,
    "acmStreamSize": 16,
    "acmStreamUnprepareHeader": 12,
}


def call_target_symbol(target: Argument) -> Optional[str]:
    """The symbol a call goes through: the name of a direct call target, or
    the absolute import slot of a `call [__imp__X]`-style indirect call."""
    if isinstance(target, AsmGlobalSymbol):
        return target.symbol_name
    if (
        isinstance(target, AsmAddressMode)
        and target.base is None
        and isinstance(target.addend, AsmGlobalSymbol)
    ):
        return target.addend.symbol_name
    return None


def is_stdcall_import(target: Argument) -> bool:
    """Whether a call target is an undecorated Win32-style DLL import, which
    uses the stdcall convention (callee pops arguments)."""
    sym = call_target_symbol(target)
    return sym is not None and bool(RE_DLL_IMPORT.match(sym))


def is_register_indirect_call(target: Argument) -> bool:
    """Whether a call target is an indirect call through a register (a COM /
    virtual method call, e.g. `call eax` or `call [ecx + 0x7c]`), as opposed
    to a direct call or a call through an absolute import slot."""
    if isinstance(target, Register):
        return True
    if isinstance(target, AsmAddressMode) and target.base is not None:
        return True
    return False


def callee_cleanup_bytes(
    target: Argument,
    context_arg_bytes: Optional[Dict[str, int]] = None,
    file_arg_bytes: Optional[Dict[str, int]] = None,
) -> Optional[int]:
    """Number of stack bytes a call target is known to pop itself: 0 for a
    known-cdecl callee, None when the convention cannot be determined from the
    name. Sources, in strict precedence order (see STDCALL_API_ARG_BYTES):

      1. an explicit context/prototype (`context_arg_bytes`, from
         __attribute__((stdcall)) declarations),
      2. a decorated stdcall suffix: an `__imp__X_N` import name, or a Ghidra
         `.set name@N` decoration recorded for the file (`file_arg_bytes`),
      3. the built-in Win32 API table.

    (Conservative structural inference, the lowest priority, lives in
    compute_call_cleanup and only runs when this returns None.)"""
    context_arg_bytes = context_arg_bytes or {}
    file_arg_bytes = file_arg_bytes or {}
    sym = call_target_symbol(target)
    if sym is None:
        return None
    # 1. Explicit context/prototype (overrides everything below).
    if sym in context_arg_bytes:
        return context_arg_bytes[sym]
    # 2. Decorated stdcall suffix: inline `__imp__X_N`, then file `.set name@N`.
    m = RE_STDCALL_IMPORT.match(sym)
    if m:
        return int(m.group(1))
    if sym in file_arg_bytes:
        return file_arg_bytes[sym]
    # 3. The built-in Win32 API convenience table.
    dll = RE_DLL_IMPORT.match(sym)
    if dll is not None:
        api = STDCALL_API_ARG_BYTES.get(dll.group(1))
        if api is not None:
            return api
    elif sym.startswith("__imp__"):
        api = STDCALL_API_ARG_BYTES.get(sym[len("__imp__") :])
        if api is not None:
            return api
    elif sym.startswith("_"):
        api = STDCALL_API_ARG_BYTES.get(sym[1:])
        if api is not None:
            return api
    return None


def switch_jump_table_labels(
    instr: Instruction, asm_data: AsmData
) -> Optional[List[str]]:
    """For an indirect `jmp [index*4 + table]`, find the jump table in the
    file's data and return the list of case labels."""
    for arg in instr.args:
        if not isinstance(arg, AsmAddressMode):
            continue
        for sub in traverse_arg(arg.addend):
            if not isinstance(sub, AsmGlobalSymbol):
                continue
            entry = asm_data.values.get(sub.symbol_name)
            if entry is None or not entry.data:
                continue
            targets = []
            for item in entry.data:
                if isinstance(item, bytes):
                    break
                target = item.as_symbol_without_addend()
                if target is None:
                    break
                targets.append(target)
            if targets:
                return targets
    return None


# The ESP-delta dataflow state at one program point: the (nonpositive) offset
# of esp relative to its value at function entry, plus the offset ebp holds
# when it is a copy of esp (`push ebp; mov ebp, esp` frames), or None if ebp
# holds an unrelated value.
EspState = Tuple[int, Optional[int]]


# Callee-saved registers, whose push/pop pairs are register saves (never call
# arguments). The cdecl scratch registers eax/ecx/edx are deliberately excluded
# (an `add esp`/`pop ecx` cleanup after a call restores them, and they are the
# regs actually pushed as arguments), so they never spoof a save/restore pair.
SAVED_PUSH_REGS = {EBX, ESI, EDI, EBP}


def compute_save_pushes(body: List[BodyPart]) -> Set[int]:
    """Indices of prologue `push` instructions that save a register or the
    frame pointer, rather than pushing an outgoing call argument. Computed
    purely structurally (before the ESP dataflow runs), so the stdcall/indirect
    cleanup inference does not miscount prologue and callee-save pushes as call
    arguments (H3).

    Register saves live in the prologue: the initial run of frame setup
    (`sub esp`, `push ebp; mov ebp, esp`, callee-saved pushes) before the first
    call/branch/return or non-save push. Only callee-saved registers that are
    restored later, and each at most once, count -- so that a callee-saved
    register pushed again as a call argument (a real pattern, e.g. `push edi`
    to pass a pointer while edi is also a saved register) is left as an
    argument, and an entry `push esi` that is passed as an argument to the very
    first call (never restored) is not mistaken for a save."""
    restored: Set[Register] = set()
    for part in body:
        if isinstance(part, Instruction):
            base, _ = split_width_suffix(part.mnemonic)
            if base == "pop" and part.args and part.args[0] in SAVED_PUSH_REGS:
                assert isinstance(part.args[0], Register)
                restored.add(part.args[0])

    def is_frame_ptr_save(index: int) -> bool:
        nxt = next(
            (p for p in body[index + 1 :] if isinstance(p, Instruction)), None
        )
        return (
            nxt is not None
            and split_width_suffix(nxt.mnemonic)[0] == "mov"
            and len(nxt.args) == 2
            and nxt.args[0] == EBP
            and nxt.args[1] == ESP
        )

    saves: Set[int] = set()
    saved_regs: Set[Register] = set()
    for i, part in enumerate(body):
        if not isinstance(part, Instruction):
            continue  # labels are transparent within the prologue
        base, _ = split_width_suffix(part.mnemonic)
        arg0 = part.args[0] if part.args else None
        if part.mnemonic == "push":
            if arg0 == EBP and is_frame_ptr_save(i):
                # The frame-pointer save; its restore is `leave`'s implicit pop
                # (no explicit `pop ebp`), so it is not in `restored`.
                saves.add(i)
                continue
            if (
                isinstance(arg0, Register)
                and arg0 in SAVED_PUSH_REGS
                and arg0 in restored
                and arg0 not in saved_regs
            ):
                saves.add(i)
                saved_regs.add(arg0)
                continue
            break  # a non-save push: the prologue's argument region begins.
        if (
            part.function_target is not None
            or part.jump_target is not None
            or part.is_return
            or part.is_conditional
        ):
            break  # the first call/branch/return ends the prologue.
    return saves


def is_fs_zero_operand(arg: Argument) -> bool:
    """Whether a (fs-segment) memory operand is exactly [0x0], the head of
    the SEH exception handler chain."""
    return (
        isinstance(arg, AsmAddressMode)
        and arg.base is None
        and isinstance(arg.addend, AsmLiteral)
        and arg.addend.value == 0
    )


CHKSTK_NAMES = {"__chkstk", "_chkstk", "__alloca_probe", "_alloca_probe"}


class X86ChkstkPattern(AsmPattern):
    """MSVC emits `mov eax, N; call __chkstk` instead of `sub esp, N` for
    frames larger than a page (__chkstk touches each new stack page in order
    to trigger guard-page growth). Semantically the pair is exactly a frame
    allocation; rewrite it into the `sub` so the ESP-delta pass sees it."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        parts = matcher.input[matcher.index : matcher.index + 2]
        if len(parts) < 2:
            return None
        mov, call = parts
        if (
            isinstance(mov, Instruction)
            and isinstance(call, Instruction)
            and mov.mnemonic == "mov"
            and mov.args[0] == EAX
            and isinstance(mov.args[1], AsmLiteral)
            and call.mnemonic == "call"
            and isinstance(call.args[0], AsmGlobalSymbol)
            and call.args[0].symbol_name in CHKSTK_NAMES
        ):
            sub = AsmInstruction("sub", [ESP, AsmLiteral(mov.args[1].value)])
            return Replacement([sub], 2, clobbers=[EAX])
        return None


class X86SehPattern(AsmPattern):
    """MSVC's structured exception handling bookkeeping. After the
    `push ebp; mov ebp, esp` frame setup, an SEH-using function pushes a
    16-byte exception registration record and installs it at fs:[0]:

        push -0x1                    # trylevel
        push <scopetable>
        push <handler>               # e.g. __except_handler3
        mov eax, fs:[0]              # old head of the handler chain
        push eax
        mov fs:[0], esp              # install the record

    and the epilogue stores the saved head back to fs:[0]. None of this is
    visible in the function's C-level semantics (__except blocks are entered
    only through the SEH dispatcher), so the prologue is replaced by a bare
    16-byte frame allocation and the epilogue store is dropped. Nonstandard
    fs: accesses are left alone and fail translation with a clear error."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        parts = matcher.input[matcher.index : matcher.index + 6]
        if len(parts) == 6 and all(isinstance(p, Instruction) for p in parts):
            p1, p2, p3, p4, p5, p6 = parts
            assert isinstance(p1, Instruction) and isinstance(p2, Instruction)
            assert isinstance(p3, Instruction) and isinstance(p4, Instruction)
            assert isinstance(p5, Instruction) and isinstance(p6, Instruction)
            if (
                p1.mnemonic == "push"
                and isinstance(p1.args[0], AsmLiteral)
                and p1.args[0].value in (-1, 0xFFFFFFFF)
                and p2.mnemonic == "push"
                and isinstance(p2.args[0], AsmGlobalSymbol)
                and p3.mnemonic == "push"
                and isinstance(p3.args[0], AsmGlobalSymbol)
                and "except_handler" in p3.args[0].symbol_name
                and p4.mnemonic == "mov.fs"
                and isinstance(p4.args[0], Register)
                and is_fs_zero_operand(p4.args[1])
                and p5.mnemonic == "push"
                and p5.args[0] == p4.args[0]
                and p6.mnemonic == "mov.fs"
                and is_fs_zero_operand(p6.args[0])
                and p6.args[1] == ESP
            ):
                sub = AsmInstruction("sub", [ESP, AsmLiteral(16)])
                return Replacement([sub], 6, clobbers=[p4.args[0]])
        # The epilogue store restoring the saved chain head, and the
        # trylevel-management stores MSVC sometimes emits through fs:[0].
        part = matcher.input[matcher.index]
        if (
            isinstance(part, Instruction)
            and part.mnemonic == "mov.fs"
            and len(part.args) == 2
            and is_fs_zero_operand(part.args[0])
            and isinstance(part.args[1], Register)
            and part.args[1] != ESP
        ):
            return Replacement([], 1, clobbers=[])
        return None


RE_SWITCHD = re.compile(r"^_switchD_(\w+?)_switchD$")
RE_SWITCHD_TARGET = re.compile(r"^_switchD_(\w+?)_(?:caseD_\w+|default)$")


class X86JumpTablePattern(AsmPattern):
    """Resolve Ghidra jump tables that are referenced by raw address.

    Ghidra usually labels jump tables, in which case the indirect jmp
    references the label and `switch_jump_table_labels` (plus flow_graph's
    jump table machinery) finds the table in the file's data. Sometimes the
    table address has no label of its own, though, and the jmp reads
    `jmp [eax*4 + 0x41391c]`. The table is still recoverable from Ghidra's
    switch label convention: such a jmp is itself labeled
    `_switchD_<id>_switchD`, and the table's `.long` entries all target
    `_switchD_<id>_caseD_*` labels of this function, appearing as a single
    consecutive run in the file's data (possibly followed by one shared
    `_switchD_*_default` entry). Register the recovered table in asm_data
    under a synthetic name and rewrite the jmp to reference it."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        part = matcher.input[matcher.index]
        if (
            not isinstance(part, Instruction)
            or part.mnemonic != "jmp"
            or not isinstance(part.args[0], AsmAddressMode)
        ):
            return None
        # Identify the switch id from Ghidra's labels around the jmp: the
        # label on the jmp itself (`_switchD_<id>_switchD`, when it survived
        # label pruning), or the first case label after the jmp (MSVC places
        # the first case body directly after the switch dispatch). The
        # bounds-check `ja` guard is *not* a reliable source: nested switches
        # share a default label carrying the outer switch's id.
        switch_id: Optional[str] = None
        if matcher.index > 0:
            prev = matcher.input[matcher.index - 1]
            if isinstance(prev, Label):
                for name in prev.names:
                    m = RE_SWITCHD.match(name)
                    if m:
                        switch_id = m.group(1)
                        break
        if switch_id is None:
            for j in range(matcher.index + 1, min(matcher.index + 3, len(matcher.input))):
                nxt = matcher.input[j]
                if isinstance(nxt, Label):
                    for name in nxt.names:
                        m = RE_SWITCHD_TARGET.match(name)
                        if m:
                            switch_id = m.group(1)
                            break
                    break
        if switch_id is None:
            return None
        # The table must be referenced by a raw address (a literal in the
        # address mode's addend); labeled tables are already handled.
        addr = part.args[0]
        literals = [
            sub for sub in traverse_arg(addr.addend) if isinstance(sub, AsmLiteral)
        ]
        table_addrs = [lit for lit in literals if lit.value > 0xFFFF]
        if len(table_addrs) != 1:
            return None
        table_addr = table_addrs[0]
        table_name = self.synthesize_table(matcher, switch_id, table_addr.value)
        if table_name is None:
            return None

        def replace_literal(arg: Argument) -> Argument:
            if arg is table_addr:
                return AsmGlobalSymbol(table_name)
            if isinstance(arg, BinOp):
                return BinOp(
                    arg.op, replace_literal(arg.lhs), replace_literal(arg.rhs)
                )
            return arg

        new_jmp = AsmInstruction(
            "jmp",
            [AsmAddressMode(addr.base, replace_literal(addr.addend), None)],
        )
        return Replacement([new_jmp], 1, clobbers=[])

    @staticmethod
    def synthesize_table(
        matcher: AsmMatcher, switch_id: str, table_addr: int
    ) -> Optional[str]:
        table_name = f"_m2c_jtbl_{table_addr:x}"
        if table_name in matcher.asm_data.values:
            return table_name
        re_case = re.compile(rf"^_switchD_{re.escape(switch_id)}_(caseD_\w+|default)$")
        re_shared_default = re.compile(r"^_switchD_\w+_default$")
        # Flatten the file's data in order and find the unique consecutive
        # run of switch entries for this switch id.
        items: List[Optional[str]] = []
        for name, entry in matcher.asm_data.values.items():
            if name.startswith("_m2c_jtbl_"):
                continue  # Tables synthesized by earlier matches.
            for item in entry.data:
                if isinstance(item, bytes):
                    items.append(None)
                else:
                    items.append(item.as_symbol_without_addend())
        runs: List[List[str]] = []
        current: List[str] = []
        for sym in items:
            if sym is not None and re_case.match(sym):
                current.append(sym)
            elif current:
                # A shared default entry may close the table.
                if sym is not None and re_shared_default.match(sym):
                    current.append(sym)
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        # A run consisting only of `default` entries is the tail of some
        # other switch's table, not a table of this switch.
        runs = [run for run in runs if any("_caseD_" in sym for sym in run)]
        if len(runs) != 1 or len(runs[0]) < 2:
            return None
        targets = runs[0]
        # All targets must be labels of the current function.
        if any(target not in matcher.labels for target in targets):
            return None
        sort_order = (f"_m2c_jtbl_{table_addr:x}", 0)
        entry = AsmDataEntry(sort_order, is_readonly=True, is_jtbl=True)
        for target in targets:
            entry.data.append(AsmSymbolicData(AsmGlobalSymbol(target), 4))
        matcher.asm_data.values[table_name] = entry
        return table_name


class X86StackRewritePattern(AsmPattern):
    """Whole-body rewrite that eliminates x86 stack pointer motion.

    m2c's stack machinery assumes a stack pointer that is fixed after the
    prologue, which x86 code violates constantly (push/pop, caller-side
    argument pushes, `add esp, N` cleanup). This pass runs a linear dataflow
    analysis over the body computing the ESP delta from function entry at
    every instruction, then rewrites all stack operations into accesses at
    fixed offsets from a virtual frame base (esp at its deepest point):

    - the frame size is the maximum stack depth; a synthetic `sub esp, N` is
      inserted at the top so that get_stack_info sees the full frame;
    - `[esp + k]` at delta d becomes `[esp + k + frame_size + d]`, and
      `[ebp + k]` similarly resolves through the tracked ebp copy;
    - `push reg` with a matching `pop reg` of the same slot (callee-saved
      register saves, mid-function spills) becomes a plain store/load pair;
    - other `push`es become `storearg.fictive` argument stores that feed the
      next `call`'s argument list;
    - prologue `push ecx`-style frame allocation and all `sub/add esp, N`
      become pure delta bookkeeping (no emitted instruction);
    - each `call` gets its current argument-region base (and its argument
      byte count, when a cleanup `add esp, N`/`pop ecx` follows or the callee
      is a known-stdcall import) appended as literal args;
    - `jmp` to a label outside the function becomes `tailcall.fictive`.
    """

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        if matcher.index != 0:
            return None
        try:
            new_body = rewrite_stack_ops(
                matcher.input, matcher.arch, matcher.asm_data, matcher.labels
            )
        except DecompFailure as e:
            # An inconsistent dataflow usually means an undecorated stdcall
            # callee (which pops its own arguments) was taken for cdecl.
            # Retry with structural stdcall inference enabled for direct
            # calls, and keep the result only if the dataflow then becomes
            # consistent; otherwise re-raise the original error. Functions
            # that analyze fine without inference are never affected.
            try:
                new_body = rewrite_stack_ops(
                    matcher.input,
                    matcher.arch,
                    matcher.asm_data,
                    matcher.labels,
                    infer_direct_stdcall=True,
                )
            except DecompFailure:
                raise e from None
        return Replacement(new_body, len(matcher.input), clobbers=[])


def rewrite_stack_ops(
    body: List[BodyPart],
    arch: ArchAsm,
    asm_data: AsmData,
    labels: Set[str],
    *,
    infer_direct_stdcall: bool = False,
) -> List[BodyPart]:
    label_pos: Dict[str, int] = {}
    for i, part in enumerate(body):
        if not isinstance(part, Instruction):
            for name in part.names:
                label_pos[name] = i

    def instr_str(item: Instruction) -> str:
        return f"`{item}` {item.meta.loc_str()}"

    # How many argument bytes each call's callee pops from the stack. cdecl
    # callees pop nothing (the caller emits an `add esp, N`/`pop` cleanup);
    # stdcall callees pop their arguments. MSVC's stdcall imports are not
    # always name-decorated (`__imp__X_8`), so this is also inferred
    # structurally: a call whose pushed arguments are not cleaned up by the
    # caller must be to a stdcall callee that popped them itself.
    ESP_NEUTRAL_STOP = {
        "call",
        "push",
        "pop",
        "pushad",
        "popad",
        "jmp",
        "ret",
        "leave",
        "storearg.fictive",
        "tailcall.fictive",
    }

    # Register saves/frame setup, identified structurally (independent of the
    # ESP dataflow, which is not available yet). A backward argument scan must
    # not mistake these for outgoing call arguments -- otherwise a call right
    # after the prologue or after a callee-save push (common for early indirect
    # / undecorated stdcall calls) gets a phantom cleanup byte count that can
    # make the dataflow look internally consistent while corrupting the stack
    # model. See H3.
    save_pushes = compute_save_pushes(body)

    def call_arg_bytes(call_index: int) -> int:
        """Bytes pushed as outgoing arguments for the call at `call_index`,
        found by scanning back over the contiguous run of arg-building
        instructions (pushes interleaved with esp-neutral setup). Labels are
        transparent: an argument may be pushed just before a merge label whose
        incoming paths each push it (e.g. `push x; jmp L` / `push y; L: call`).
        Register-save / frame pushes are not arguments and stop the scan."""
        total = 0
        j = call_index - 1
        while j >= 0:
            part = body[j]
            if not isinstance(part, Instruction):
                j -= 1
                continue
            base, _ = split_width_suffix(part.mnemonic)
            if part.mnemonic == "push":
                if j in save_pushes:
                    break  # a callee-save / frame-pointer save, not an argument.
                total += 4
            elif base == "push":
                break  # sub-word push; give up.
            elif base == "mov" and len(part.args) == 2 and part.args[0] == EBP and part.args[1] == ESP:
                break  # `mov ebp, esp` frame setup: nothing before it is an arg.
            elif (
                base in ESP_NEUTRAL_STOP
                or part.function_target is not None
                or part.jump_target is not None
                or part.is_conditional
                or part.is_return
            ):
                break  # a call/branch/return barrier bounds the arg region.
            elif ESP in part.outputs or ESP in part.clobbers:
                break
            j -= 1
        return total

    # Callee cleanup information beyond name decoration, kept in separate
    # precedence tiers (see callee_cleanup_bytes): user-context stdcall
    # prototypes (highest), and Ghidra-exported `.set sym, "name@N"`
    # decorations for this file.
    context_arg_bytes: Dict[str, int] = dict(
        getattr(arch, "context_stdcall_arg_bytes", {})
    )
    file_arg_bytes: Dict[str, int] = dict(asm_data.stdcall_arg_bytes)

    call_cleanup: Dict[int, int] = {}

    def caller_cleans_up(call_index: int) -> bool:
        """Whether the caller restores esp for this call's arguments: either
        a cleanup (`add esp, N` / `pop ecx`) directly after the call, or an
        MSVC batched cleanup further down the same straight-line region -- a
        single `add esp, N` covering several calls' arguments at once, which
        must include this call's if it pops more than the argument bytes
        pushed after this call returned."""
        extra = 0  # argument bytes pushed since the call returned
        j = call_index + 1
        while j < len(body):
            part = body[j]
            if not isinstance(part, Instruction):
                return False  # a label: control flow merge; give up.
            base, _ = split_width_suffix(part.mnemonic)
            if (
                base == "add"
                and part.args[0] == ESP
                and isinstance(part.args[1], AsmLiteral)
            ):
                if part.args[1].value > extra:
                    return True
                extra -= part.args[1].value
            elif base == "pop" and part.args[0] in (ECX, EDX):
                if extra == 0:
                    return True
                extra -= 4
            elif part.mnemonic == "push":
                extra += 4
            elif base == "call" and part.function_target is not None:
                # Nested call; account for what its callee pops (computed
                # already: calls are classified back to front).
                extra -= call_cleanup.get(j, 0)
                if extra < 0:
                    # A later stdcall callee popped more than was pushed
                    # after this call, so the pushes taken for this call's
                    # arguments belonged to that callee instead.
                    return True
            elif (
                base in ESP_NEUTRAL_STOP
                or part.function_target is not None
                or part.is_return
                or part.jump_target is not None
                or part.is_conditional
                or ESP in part.outputs
                or ESP in part.clobbers
            ):
                return False
            j += 1
        return False

    def compute_call_cleanup(call_index: int) -> int:
        call = body[call_index]
        assert isinstance(call, Instruction)
        target = call.args[0]
        known = callee_cleanup_bytes(target, context_arg_bytes, file_arg_bytes)
        if known is not None:
            return known
        # The callee's convention is not known from its name. stdcall callees
        # pop their own arguments; cdecl callees (including MSVC's
        # batched-cleanup pattern, where one `add esp, N` restores several
        # calls' arguments at once) always have the caller restore esp. We
        # treat a call as self-cleaning when no caller cleanup follows and
        # either:
        #  - its name marks it as a Win32/DirectX DLL import
        #    (`_<LIB>_DLL_<Func>`), or
        #  - it is an indirect call through a register (`call [ecx+0x7c]`,
        #    `call eax`), i.e. a COM/virtual method, or
        #  - `infer_direct_stdcall` is set: the retry pass extends the same
        #    inference to direct calls (undecorated stdcall callees), relying
        #    on the dataflow consistency check to validate the result.
        if is_stdcall_import(target):
            return call_arg_bytes(call_index)
        if is_register_indirect_call(target) and not caller_cleans_up(call_index):
            return call_arg_bytes(call_index)
        if (
            infer_direct_stdcall
            and call_target_symbol(target) is not None
            and not caller_cleans_up(call_index)
        ):
            return call_arg_bytes(call_index)
        return 0

    # Classify calls back to front so that the forward scan in
    # caller_cleans_up can account for later callees' stack pops.
    for i in reversed(range(len(body))):
        part = body[i]
        if isinstance(part, Instruction) and part.function_target is not None:
            base, _ = split_width_suffix(part.mnemonic)
            if base == "call":
                call_cleanup[i] = compute_call_cleanup(i)

    # The esp delta ebp holds after the function's `mov ebp, esp` frame setup.
    # `pushad`/`popad` and mid-function scratch use of ebp can lose the tracked
    # ebp value on a path; recovering it from this frame constant lets the
    # epilogue's `mov esp, ebp`/`leave` still resolve. Ambiguous (conflicting)
    # setups leave it None, disabling the fallback.
    frame_ebp: Optional[int] = None
    frame_ebp_ambiguous = False

    # Pass 1: dataflow analysis computing the ESP delta at entry to every
    # reachable instruction.
    states: List[Optional[EspState]] = [None] * len(body)

    def merge(a: EspState, b: EspState, item: Instruction) -> EspState:
        if a[0] != b[0]:
            raise DecompFailure(
                f"x86 stack analysis failed: conflicting stack depths "
                f"({-a[0]:#x} vs {-b[0]:#x}) at {instr_str(item)}"
            )
        return (a[0], a[1] if a[1] == b[1] else None)

    def step(item: Instruction, st: EspState, index: int) -> None:
        """Push the out-state(s) of `item` onto the worklist."""
        nonlocal frame_ebp, frame_ebp_ambiguous
        esp, ebp = st
        base, _ = split_width_suffix(item.mnemonic)
        args = item.args
        fallthrough = True

        def jump_to(name: str, target_st: EspState) -> None:
            pos = label_pos.get(name)
            if pos is not None:
                worklist.append((pos, target_st))

        if item.is_return:
            return
        if base == "push":
            if item.mnemonic != "push":
                raise DecompFailure(
                    f"unsupported sub-word x86 push: {instr_str(item)}"
                )
            st = (esp - 4, ebp)
        elif base == "pop":
            st = (esp + 4, None if args[0] == EBP else ebp)
        elif base == "pushad":
            st = (esp - 32, ebp)
        elif base == "popad":
            # popad restores ebp to its value at the matching pushad, which
            # for a frame pointer is the frame value.
            st = (esp + 32, frame_ebp if not frame_ebp_ambiguous else None)
        elif base in ("sub", "add") and args[0] == ESP:
            if not isinstance(args[1], AsmLiteral):
                raise DecompFailure(
                    f"cannot statically analyze stack adjustment {instr_str(item)}"
                )
            st = (esp - args[1].value if base == "sub" else esp + args[1].value, ebp)
        elif base == "mov" and args[0] == EBP and args[1] == ESP:
            if frame_ebp is not None and frame_ebp != esp:
                frame_ebp_ambiguous = True
            frame_ebp = esp
            st = (esp, esp)
        elif base == "mov" and args[0] == ESP:
            src_ebp = ebp if ebp is not None else (
                frame_ebp if not frame_ebp_ambiguous else None
            )
            if args[1] != EBP or src_ebp is None:
                raise DecompFailure(
                    f"cannot statically analyze stack adjustment {instr_str(item)}"
                )
            st = (src_ebp, src_ebp)
        elif base == "lea" and args[0] == ESP:
            src = args[1]
            if (
                isinstance(src, AsmAddressMode)
                and isinstance(src.addend, AsmLiteral)
                and (src.base == ESP or (src.base == EBP and ebp is not None))
            ):
                src_esp = esp if src.base == ESP else ebp
                assert src_esp is not None
                st = (src_esp + src.addend.value, ebp)
            else:
                raise DecompFailure(
                    f"cannot statically analyze stack adjustment {instr_str(item)}"
                )
        elif base == "leave":
            src_ebp = ebp if ebp is not None else (
                frame_ebp if not frame_ebp_ambiguous else None
            )
            if src_ebp is None:
                raise DecompFailure(
                    f"`leave` without a tracked ebp frame: {instr_str(item)}"
                )
            st = (src_ebp + 4, None)
        elif base == "and" and args[0] == ESP:
            # Stack alignment (`and esp, -8`). The rewritten frame is an
            # abstraction anyway, so treat alignment as a no-op; other masks
            # are rejected below.
            if not (
                isinstance(args[1], AsmLiteral)
                and args[1].value & 0xFFFFFFFF
                in (0xFFFFFFF0, 0xFFFFFFF8, 0xFFFFFFFC)
            ):
                raise DecompFailure(
                    f"cannot statically analyze stack adjustment {instr_str(item)}"
                )
        elif item.function_target is not None:
            st = (esp + call_cleanup.get(index, 0), ebp)
        elif base == "jmp":
            if isinstance(item.jump_target, JumpTarget):
                if item.jump_target.target in label_pos:
                    jump_to(item.jump_target.target, st)
                # else: a tail call; terminal.
                return
            # Indirect jump: a jump table.
            targets = switch_jump_table_labels(item, asm_data)
            if targets is None:
                raise DecompFailure(
                    f"Unable to determine jump table for {instr_str(item)}"
                )
            for target in targets:
                jump_to(target, st)
            return
        else:
            if ESP in item.outputs:
                raise DecompFailure(
                    f"cannot statically analyze stack adjustment {instr_str(item)}"
                )
            if EBP in item.outputs or EBP in item.clobbers:
                st = (esp, None)
        if isinstance(item.jump_target, JumpTarget):
            jump_to(item.jump_target.target, st)
            if not item.is_conditional:
                fallthrough = False
        if fallthrough:
            worklist.append((index + 1, st))

    worklist: List[Tuple[int, EspState]] = [(0, (0, None))]
    while worklist:
        index, st = worklist.pop()
        if index >= len(body):
            continue
        part = body[index]
        if not isinstance(part, Instruction):
            # Labels pass the state through.
            prev = states[index]
            if prev is not None:
                st = (st[0], st[1] if st[1] == prev[1] else None)
                if st[0] != prev[0]:
                    raise DecompFailure(
                        f"x86 stack analysis failed: conflicting stack depths "
                        f"({-st[0]:#x} vs {-prev[0]:#x}) at label {part}"
                    )
                if st == prev:
                    continue
            states[index] = st
            worklist.append((index + 1, st))
            continue
        prev = states[index]
        if prev is not None:
            st = merge(prev, st, part)
            if st == prev:
                continue
        states[index] = st
        step(part, st, index)

    # The frame is a fixed-size region covering the deepest stack extent.
    frame_size = 0
    for i, part in enumerate(body):
        st = states[i]
        if st is None or not isinstance(part, Instruction):
            continue
        depth = -st[0]
        base, _ = split_width_suffix(part.mnemonic)
        if base == "push":
            depth += 4
        elif base == "pushad":
            depth += 32
        elif base == "sub" and part.args[0] == ESP:
            assert isinstance(part.args[1], AsmLiteral)
            depth += part.args[1].value
        frame_size = max(frame_size, depth)

    # Pass 2: classify pops (cleanup pops directly after calls discard
    # argument slots) and pushes (saves/spills with a matching pop, prologue
    # frame allocations, and call argument stores).
    cleanup_pops: Set[int] = set()
    prev_instr: Optional[Instruction] = None
    prev_index = -1
    for i, part in enumerate(body):
        if not isinstance(part, Instruction):
            prev_instr = None
            continue
        base, _ = split_width_suffix(part.mnemonic)
        if (
            base == "pop"
            and part.args[0] in (ECX, EDX)
            and prev_instr is not None
            and (
                prev_instr.function_target is not None or prev_index in cleanup_pops
            )
        ):
            cleanup_pops.add(i)
        prev_instr = part
        prev_index = i

    def cdecl_arg_bytes(call_index: int) -> int:
        """For a cdecl call: the number of argument bytes attributable to
        this call based on the caller-side cleanup that follows, or -1 when
        no cleanup is found (translation then takes all pending argument
        stores). Handles immediate cleanup (`add esp, N`, cleanup pops),
        MSVC batched cleanup (one add covering several calls' arguments),
        and argument regions consumed by a later stdcall callee (in which
        case the pushes preceding this call belonged to that callee)."""
        consume = -1
        extra = 0  # argument bytes pushed since this call returned
        j = call_index + 1
        while j < len(body):
            part = body[j]
            if not isinstance(part, Instruction):
                break
            base, _ = split_width_suffix(part.mnemonic)
            if (
                base == "add"
                and part.args[0] == ESP
                and isinstance(part.args[1], AsmLiteral)
            ):
                n = part.args[1].value
                if n > extra:
                    return (0 if consume < 0 else consume) + n - extra
                extra -= n
            elif j in cleanup_pops:
                if extra == 0:
                    consume = (0 if consume < 0 else consume) + 4
                else:
                    extra -= 4
            elif part.mnemonic == "push":
                extra += 4
            elif base == "call" and part.function_target is not None:
                extra -= call_cleanup.get(j, 0)
                if extra < 0:
                    # A later stdcall callee popped more than was pushed
                    # after this call: the remaining pushes were its
                    # arguments, not this call's.
                    return 0 if consume < 0 else consume
            elif (
                base in ESP_NEUTRAL_STOP
                or part.function_target is not None
                or part.is_return
                or part.jump_target is not None
                or part.is_conditional
                or ESP in part.outputs
                or ESP in part.clobbers
            ):
                break
            j += 1
        return consume

    # Locations (offsets from the virtual frame base) that non-cleanup pops
    # read back into the same register, plus `leave`'s implicit pop of ebp.
    pop_locs: Set[Tuple[int, Register]] = set()
    for i, part in enumerate(body):
        st = states[i]
        if not isinstance(part, Instruction) or st is None or i in cleanup_pops:
            continue
        base, _ = split_width_suffix(part.mnemonic)
        if base == "pop" and isinstance(part.args[0], Register):
            pop_locs.add((frame_size + st[0], part.args[0]))
        elif base == "leave" and st[1] is not None:
            pop_locs.add((frame_size + st[1], EBP))

    def is_save_push(index: int) -> bool:
        part = body[index]
        st = states[index]
        assert isinstance(part, Instruction) and st is not None
        arg = part.args[0]
        return (
            isinstance(arg, Register)
            and (frame_size + st[0] - 4, arg) in pop_locs
        )

    # The prologue: the initial run of frame-establishing instructions.
    # Register pushes within it that are never popped back allocate frame
    # space (MSVC's `push ecx` idiom); their stored value is meaningless.
    alloc_pushes: Set[int] = set()
    last_frame_op = -1
    for i, part in enumerate(body):
        if not isinstance(part, Instruction) or states[i] is None:
            break
        if (
            part.function_target is not None
            or part.jump_target is not None
            or part.is_return
        ):
            break
        base, _ = split_width_suffix(part.mnemonic)
        if base == "mov" and part.args[0] == EBP and part.args[1] == ESP:
            last_frame_op = i
        elif base == "sub" and part.args[0] == ESP:
            last_frame_op = i
        elif base == "push" and is_save_push(i):
            last_frame_op = i
    for i in range(last_frame_op + 1):
        part = body[i]
        if (
            isinstance(part, Instruction)
            and states[i] is not None
            and split_width_suffix(part.mnemonic)[0] == "push"
            and isinstance(part.args[0], Register)
            and not is_save_push(i)
        ):
            alloc_pushes.add(i)

    # Pass 3: emit the rewritten body.
    # When ebp is used as a frame pointer, its save/setup/restore
    # (`push ebp; mov ebp, esp; ...; mov esp, ebp; pop ebp` / `leave`) are pure
    # bookkeeping: every `[ebp+k]` access is rewritten against esp, so ebp is
    # never read, and it is left holding its caller value throughout.
    ebp_is_frame_ptr = frame_ebp is not None and not frame_ebp_ambiguous
    new_body: List[BodyPart] = []

    def emit(mnemonic: str, args: List[Argument], meta: InstructionMeta) -> None:
        new_body.append(arch.parse(mnemonic, args, meta.derived()))

    if frame_size > 0:
        first_meta = next(
            (p.meta for p in body if isinstance(p, Instruction)),
            InstructionMeta.missing(),
        )
        emit("sub", [ESP, AsmLiteral(frame_size)], first_meta)

    for i, part in enumerate(body):
        if not isinstance(part, Instruction):
            new_body.append(part)
            continue
        st = states[i]
        if st is None:
            # Unreachable code; leave it untouched.
            new_body.append(part)
            continue
        esp, ebp = st
        base, _ = split_width_suffix(part.mnemonic)
        args = part.args

        def adjust(addend: Argument, amount: int) -> Argument:
            if amount == 0:
                return addend
            if isinstance(addend, AsmLiteral):
                return AsmLiteral(addend.value + amount)
            if (
                isinstance(addend, BinOp)
                and addend.op == "+"
                and isinstance(addend.rhs, AsmLiteral)
            ):
                return BinOp("+", addend.lhs, AsmLiteral(addend.rhs.value + amount))
            return BinOp("+", addend, AsmLiteral(amount))

        def rewrite_operand(arg: Argument) -> Argument:
            if not isinstance(arg, AsmAddressMode):
                return arg
            base, addend = arg.base, arg.addend
            # The parser represents `[reg - N]` (and other displacement-only
            # negative forms) as base=None with a `reg - N` BinOp addend,
            # unlike `[reg + N]` which sets base=reg. Normalize so esp/ebp
            # frame accesses are recognized either way.
            if (
                base is None
                and isinstance(addend, BinOp)
                and addend.op in ("+", "-")
                and isinstance(addend.lhs, Register)
                and isinstance(addend.rhs, AsmLiteral)
            ):
                base = addend.lhs
                sign = 1 if addend.op == "+" else -1
                addend = AsmLiteral(sign * addend.rhs.value)
            elif base is None and isinstance(addend, Register):
                base, addend = addend, AsmLiteral(0)
            if isinstance(addend, AsmLiteral) and base in (ESP, EBP):
                # Large negative frame displacements can be exported in
                # unsigned form (`[ebp + 0xfffff4b4]` meaning `[ebp - 0xb4c]`).
                value = addend.value & 0xFFFFFFFF
                if value >= 0x80000000:
                    value -= 0x100000000
                addend = AsmLiteral(value)
            if base == ESP:
                return AsmAddressMode(ESP, adjust(addend, frame_size + esp), None)
            if base == EBP and ebp is not None:
                return AsmAddressMode(ESP, adjust(addend, frame_size + ebp), None)
            return arg

        if base == "push":
            loc = frame_size + esp - 4
            if i in alloc_pushes:
                pass  # Pure frame allocation; the value is meaningless.
            elif args[0] == EBP and ebp_is_frame_ptr:
                pass  # Saving the caller's frame pointer; bookkeeping only.
            elif is_save_push(i):
                assert isinstance(args[0], Register)
                emit(
                    "mov",
                    [AsmAddressMode(ESP, AsmLiteral(loc), None), args[0]],
                    part.meta,
                )
            else:
                # An outgoing function call argument.
                emit(
                    "storearg.fictive",
                    [AsmLiteral(loc), rewrite_operand(args[0])],
                    part.meta,
                )
        elif base == "pop":
            if args[0] == EBP and ebp_is_frame_ptr:
                pass  # Restoring the caller's frame pointer; bookkeeping only.
            elif i not in cleanup_pops:
                loc = frame_size + esp
                emit(
                    part.mnemonic.replace("pop", "mov", 1),
                    [rewrite_operand(args[0]), AsmAddressMode(ESP, AsmLiteral(loc), None)],
                    part.meta,
                )
        elif base in ("pushad", "popad"):
            pass
        elif base in ("sub", "add") and args[0] == ESP:
            pass
        elif base == "and" and args[0] == ESP:
            pass  # Stack alignment; a no-op on the abstracted frame.
        elif base == "mov" and args[0] == ESP:
            pass
        elif base == "mov" and args[0] == EBP and args[1] == ESP and ebp_is_frame_ptr:
            pass  # `mov ebp, esp` frame setup; ebp accesses resolve to esp.
        elif base == "mov" and isinstance(args[0], Register) and args[1] == ESP:
            # Taking the address of the stack into a general register.
            emit(
                "lea",
                [args[0], AsmAddressMode(ESP, AsmLiteral(frame_size + esp), None)],
                part.meta,
            )
        elif base == "lea" and args[0] == ESP:
            pass
        elif base == "leave":
            # `leave` = `mov esp, ebp; pop ebp`; pure frame teardown.
            pass
        elif part.function_target is not None and base == "call":
            # Annotate the call with the base of its stack argument region and
            # the number of argument bytes belonging to it, so translation can
            # split the pending stack arguments across nested calls.
            consume = call_cleanup.get(i, 0)
            if consume == 0:
                consume = cdecl_arg_bytes(i)
            emit(
                "call",
                [rewrite_operand(args[0]), AsmLiteral(frame_size + esp), AsmLiteral(consume)],
                part.meta,
            )
        elif base == "jmp" and isinstance(part.jump_target, JumpTarget):
            if part.jump_target.target in label_pos:
                new_body.append(part)
            else:
                # Tail call to another function.
                emit(
                    "tailcall.fictive",
                    [AsmGlobalSymbol(part.jump_target.target)],
                    part.meta,
                )
        else:
            new_args = [rewrite_operand(arg) for arg in args]
            if new_args != args:
                emit(part.mnemonic, new_args, part.meta)
            else:
                new_body.append(part)

    return new_body


# Builders for read-modify-write ALU instructions: (args, old dst value,
# source operand values) -> new dst value.
AluBuilder = Callable[[InstrArgs, Expression, List[Expression]], Expression]
# Builders for single-operand read-modify-write instructions.
UnaryBuilder = Callable[[InstrArgs, Expression], Expression]

# How an instruction affects the flag pseudo-registers:
# - "cmp": full compare-style flags of (dst, src), evaluated *before* the
#   destination is overwritten (like eval_arm_cmp); used by sub and cmp.
#   The c flag is a borrow; see eval_x86_cmp.
# - "add": width-aware flags of an addition result (add/adc), including
#   c = carry-out and hi = (CF==0 && ZF==0).
# - "sbb": subtract-with-borrow flags (sbb): the flags of lhs - (src + CF),
#   computed like a compare so c is a borrow (not logic-op flags).
# - "logic": z/n/hi/ge/gt from the result compared against zero, and
#   c = v = 0 (real x86 semantics for and/or/xor/test; an acceptable
#   approximation for shifts, whose carry-out is rarely consumed).
# - "keep_c": like "logic" but preserving the previous carry flag and
#   setting v from the result (inc/dec semantics); the composite hi predicate
#   is recomputed from the preserved carry plus the new zero flag.
# - "clobber": flags are structurally clobbered but no symbolic value is
#   recorded (rotates, multiplications, divisions).
# - "none": flags are untouched (not/bswap).
FLAGS_CMP = "cmp"
FLAGS_ADD = "add"
FLAGS_SBB = "sbb"
FLAGS_LOGIC = "logic"
FLAGS_KEEP_C = "keep_c"
FLAGS_CLOBBER = "clobber"
FLAGS_NONE = "none"


# --- x87 FPU eval helpers (used by X86Arch._parse_fpu) ---
#
# The x87 register stack is eliminated by X86FpuRewritePattern (m2c/x86_fpu.py),
# which rewrites every reachable x87 instruction into a fictive form carrying
# explicit flat virtual registers f0..f7 (e.g. `fadd $f1, $f0`, `fstp.s [m],
# $f2`). The handlers below give those fictive forms their semantics. Virtual
# registers carry Type.floatish() (width-less, like a physical 80-bit slot);
# concrete f32/f64 width only exists at memory boundaries.


def fpu_float_type(width: int) -> Type:
    """The C float type for an x87 memory operand of the given byte width."""
    return Type.f64() if width == 8 else Type.f32()


def fpu_int_type(width: int) -> Type:
    """The signed C integer type for an fild/fistp operand of a given width."""
    return {2: Type.s16(), 4: Type.s32(), 8: Type.s64()}[width]


def is_f64_expr(expr: Expression) -> bool:
    return expr.type.is_float() and expr.type.get_size_bits() == 64


def fpu_binop(
    op: str, lhs: Expression, rhs: Expression, *, reverse: bool = False
) -> Expression:
    """Build an x87 arithmetic binop. Result is f64 when either operand is
    known-f64 (matching C's float->double promotion for the common
    `float op double_constant` pattern), else f32. `reverse` handles the
    fsubr/fdivr forms, which compute `rhs op lhs`."""
    a, b = (rhs, lhs) if reverse else (lhs, rhs)
    if is_f64_expr(lhs) or is_f64_expr(rhs):
        return BinaryOp.f64(a, op, b)
    return BinaryOp.f32(a, op, b)


def f32_literal(value: float) -> Literal:
    """A float constant as an f32-typed Literal holding its IEEE-754 bits."""
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    return Literal(bits, type=Type.f32())


# x87 constant loads. 0/1 render as numeric literals; the transcendental
# constants stay named so matching source can #define them to the exact bits.
FPU_CONSTANTS: Dict[str, Callable[[], Expression]] = {
    "fld1": lambda: f32_literal(1.0),
    "fldz": lambda: f32_literal(0.0),
    "fldpi": lambda: fn_op("M2C_PI", [], Type.f32()),
    "fldl2e": lambda: fn_op("M2C_LOG2E", [], Type.f32()),
    "fldl2t": lambda: fn_op("M2C_LOG2T", [], Type.f32()),
    "fldlg2": lambda: fn_op("M2C_LOG10_2", [], Type.f32()),
    "fldln2": lambda: fn_op("M2C_LN2", [], Type.f32()),
}


# Name of the symbolic x87 status-word marker (fn_op) threaded from an
# fcom-family compare through fnstsw ax to the test-ah idiom (see §3 of the
# design spec). Carries the compare's two operands as its arguments.
FNSTSW_MARKER = "M2C_FNSTSW"


def fnstsw_marker_operands(expr: Expression) -> Optional[Tuple[Expression, Expression]]:
    """If `expr` is an x87 compare status-word marker, its (lhs, rhs) operands;
    otherwise None."""
    uw = early_unwrap(expr)
    if (
        isinstance(uw, FuncCall)
        and isinstance(uw.function, GlobalSymbol)
        and uw.function.symbol_name == FNSTSW_MARKER
        and len(uw.args) == 2
    ):
        return uw.args[0], uw.args[1]
    return None


def fpu_compare_condition(
    lhs: Expression, rhs: Expression, op: str
) -> Condition:
    """A float comparison `lhs op rhs`, as f64 when either operand is
    known-f64 (matching the arithmetic width rule) else f32."""
    if is_f64_expr(lhs) or is_f64_expr(rhs):
        return BinaryOp.dcmp(lhs, op, rhs)
    return BinaryOp.fcmp(lhs, op, rhs)


# TEST AH, mask after FNSTSW AX: the relational operator that is true exactly
# when the x87 compare's ZF would be 1 (i.e. what `jz`/`setz` should read).
# C0 (AH 0x01) = "st0 < src", C3 (AH 0x40) = "st0 == src"; ZF = ((AH & mask)
# == 0). See §3.2. Unordered (NaN) outcomes are folded into the signed
# direction, as MSVC's mask choice implies.
FNSTSW_MASK_OPS: Dict[int, str] = {
    0x01: ">=",  # ZF=1 <=> not(st0 < src)
    0x40: "!=",  # ZF=1 <=> not(st0 == src)
    0x41: ">",   # ZF=1 <=> not(st0 < src) and not(st0 == src)
    0x05: ">=",  # C0|C2: unordered folded in
    0x45: ">",   # C0|C2|C3
    0x44: "!=",  # C2|C3
}


# x87 arithmetic op -> (C operator, reverse-operands flag).
FPU_ARITH_OPS: Dict[str, Tuple[str, bool]] = {
    "fadd": ("+", False),
    "fsub": ("-", False),
    "fsubr": ("-", True),
    "fmul": ("*", False),
    "fdiv": ("/", False),
    "fdivr": ("/", True),
    "faddp": ("+", False),
    "fsubp": ("-", False),
    "fsubrp": ("-", True),
    "fmulp": ("*", False),
    "fdivp": ("/", False),
    "fdivrp": ("/", True),
    "fiadd": ("+", False),
    "fisub": ("-", False),
    "fisubr": ("-", True),
    "fimul": ("*", False),
    "fidiv": ("/", False),
    "fidivr": ("/", True),
}

# x87 unary op -> builder (value -> new value). frndint stays a visible
# intrinsic (its result depends on the rounding-control word, not modeled);
# f2xm1 renders as its libm identity exp2f(x)-1 (its |x|<=1 domain is elided).
FPU_UNARY_OPS: Dict[str, Callable[[Expression], Expression]] = {
    "fchs": lambda v: UnaryOp("-", v, type=Type.floatish()),
    "fabs": lambda v: fn_op("fabsf", [v], Type.f32()),
    "fsqrt": lambda v: fn_op("sqrtf", [v], Type.f32()),
    "fsin": lambda v: fn_op("sinf", [v], Type.f32()),
    "fcos": lambda v: fn_op("cosf", [v], Type.f32()),
    "frndint": lambda v: fn_op("M2C_RNDINT", [v], Type.f32()),
    "f2xm1": lambda v: BinaryOp.f32(fn_op("exp2f", [v], Type.f32()), "-", f32_literal(1.0)),
}

# x87 two-operand transcendentals. Each is `builder(st0, st1) -> value`,
# written into `dst` (0 = st0/top, 1 = st1); `pop` also kills the top. The
# rewrite passes [st0, st1] as the fictive operands.
FPU_BINARY_OPS: Dict[str, Tuple[Callable[[Expression, Expression], Expression], int, bool]] = {
    # fpatan: st1 = atan2(st1, st0), pop.
    "fpatan": (lambda st0, st1: fn_op("atan2f", [st1, st0], Type.f32()), 1, True),
    # fyl2x: st1 = st1 * log2(st0), pop.
    "fyl2x": (
        lambda st0, st1: BinaryOp.f32(st1, "*", fn_op("log2f", [st0], Type.f32())),
        1,
        True,
    ),
    # fyl2xp1: st1 = st1 * log2(st0 + 1), pop.
    "fyl2xp1": (
        lambda st0, st1: BinaryOp.f32(
            st1,
            "*",
            fn_op("log2f", [BinaryOp.f32(st0, "+", f32_literal(1.0))], Type.f32()),
        ),
        1,
        True,
    ),
    # fscale: st0 = st0 * 2**trunc(st1) = ldexpf(st0, (int)st1).
    "fscale": (
        lambda st0, st1: fn_op(
            "ldexpf", [st0, handle_convert(st1, Type.s32(), Type.floatish())], Type.f32()
        ),
        0,
        False,
    ),
    # fprem/fprem1: st0 = fmod(st0, st1).
    "fprem": (lambda st0, st1: fn_op("fmodf", [st0, st1], Type.f32()), 0, False),
    "fprem1": (lambda st0, st1: fn_op("fmodf", [st0, st1], Type.f32()), 0, False),
}


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

    # f0 (the bottom x87 slot) holds a float/double return, disambiguated from
    # eax the same way MIPS disambiguates v0/f0. f0 is deliberately *not* in
    # all_return_regs: that list drives the default call-output set, and x87
    # values must survive calls uncleared (a call only pushes its own float
    # return via the per-call `fpret` annotation, §4.3).
    base_return_regs = [(EAX, False), (Register("f0"), True)]
    all_return_regs = [EAX, EDX]
    argument_regs: List[Register] = []
    simple_temp_regs = [ECX, EDX]
    flag_regs = [Register(r) for r in ["n", "z", "c", "v", "hi", "ge", "gt"]]
    # `fsw` carries the symbolic x87 status word between an fcom-family compare
    # and the fnstsw/test-ah idiom that consumes it (see _parse_fpu / eval_cmp).
    # A temp reg so calls clobber it; not a flag reg so it is not swept by the
    # integer-flag machinery.
    fsw_reg = Register("fsw")
    temp_regs = [EAX] + simple_temp_regs + flag_regs + [fsw_reg]
    saved_regs = [EBX, ESI, EDI, EBP, EIP]
    # Raw x87 stack-register names (`st(0)`..`st(7)`, spelled `st0`..`st7`).
    # These reach `parse` only during initial parsing; X86FpuRewritePattern
    # rewrites every reachable use into a flat virtual register before
    # translation, so they never carry real semantics.
    fpu_regs = [Register(f"st{i}") for i in range(8)]
    # The bottom-anchored flat x87 virtual registers produced by the FPU
    # prepass. Their `f` prefix gives them float treatment (Register.is_float)
    # for free. They are in neither temp_regs (so calls don't clear them --
    # x87 values live across calls in this corpus) nor saved_regs (so they get
    # no spurious initial "caller value", keeping float-return detection clean).
    float_regs = [Register(f"f{i}") for i in range(8)]
    # Sub-registers are parsed as their own Register instances so that operand
    # widths survive until normalize_instruction, which rewrites them into
    # full registers plus a width-suffixed mnemonic.
    all_regs = (
        saved_regs
        + temp_regs
        + [stack_pointer_reg]
        + fpu_regs
        + float_regs
        + list(SUB_REGS.keys())
    )

    aliased_regs: Dict[str, Register] = {}

    asm_patterns: List[AsmPattern] = [
        X86ChkstkPattern(),
        X86SehPattern(),
        X86JumpTablePattern(),
        X86StackRewritePattern(),
        X86FpuRewritePattern(),
    ]

    def __init__(self) -> None:
        # Callee-cleanup byte counts for stdcall functions declared in the
        # user-provided context (via __attribute__((stdcall))), keyed by asm
        # symbol name. Populated by load_context.
        self.context_stdcall_arg_bytes: Dict[str, int] = {}
        # x87 FPU stack deltas for context functions with a float/double return
        # type (+1: they leave their result in st(0)), keyed by asm symbol name.
        # Seeds the FPU prepass so a call whose result is returned directly (a
        # wrapper with no local FPU op of its own) is still modeled as producing
        # st(0). Populated by load_context.
        self.context_fpu_call_deltas: Dict[str, int] = {}

    def load_context(self, typemap: TypeMap) -> None:
        """Record per-callee ABI facts derived from the user-provided context
        that the x86 prepasses cannot recover from the assembly alone:

        - stdcall callee-cleanup byte counts (functions declared with
          __attribute__((stdcall))): the number of stack bytes the callee pops
          on return, i.e. the sum of its parameter sizes rounded up to 4-byte
          slots.
        - x87 stack deltas for float/double-returning functions (+1): such a
          callee leaves its result on the FPU stack, which the FPU prepass must
          know even for a caller that has no FPU instruction of its own.

        Context declarations use asm symbol names (e.g. `_MessageBoxA` for
        MSVC-compiled 32-bit code)."""
        from .c_types import parse_struct_member, resolve_typedefs
        from m2c_pycparser import c_ast as ca

        def record(mapping: Dict[str, int], name: str, value: int) -> None:
            mapping[name] = value
            # x86 asm symbols carry a leading-underscore platform prefix
            # (`_MessageBoxA` for `MessageBoxA`); match either spelling.
            if not name.startswith("_"):
                mapping[f"_{name}"] = value

        for name, fn in typemap.functions.items():
            # A float/double return leaves a value on the x87 stack (delta +1).
            if fn.ret_type is not None and self._ctype_is_float(fn.ret_type, typemap):
                record(self.context_fpu_call_deltas, name, 1)

            if not fn.is_stdcall or fn.params is None or fn.is_variadic:
                continue
            total = 0
            for param in fn.params:
                tp, _ = resolve_typedefs(param.type, typemap)
                if isinstance(tp, (ca.PtrDecl, ca.ArrayDecl, ca.FuncDecl)):
                    # Pointers, and arrays/functions, which decay to pointers.
                    size = 4
                else:
                    size, _, _ = parse_struct_member(
                        param.type,
                        param.name or "<anonymous>",
                        typemap,
                        allow_unsized=False,
                    )
                total += (size + 3) & ~3
            record(self.context_stdcall_arg_bytes, name, total)

    @staticmethod
    def _ctype_is_float(ret_type: CType, typemap: TypeMap) -> bool:
        """Whether a context function's return type is `float` or `double`
        (so the callee returns its result in st(0))."""
        from .c_types import resolve_typedefs
        from m2c_pycparser import c_ast as ca

        tp, _ = resolve_typedefs(ret_type, typemap)
        return (
            isinstance(tp, ca.TypeDecl)
            and isinstance(tp.type, ca.IdentifierType)
            and ("float" in tp.type.names or "double" in tp.type.names)
        )

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
            prefix = {"repz": "repe", "repnz": "repne"}.get(
                instr.mnemonic, instr.mnemonic
            )
            return AsmInstruction(f"{prefix}.{op}", [])

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

    # Condition code -> (flag register read, negated).
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
        # The c pseudo-register holds x86's carry flag, which after cmp/sub
        # is a *borrow*: c = (u32)lhs < (u32)rhs. (This is the inverse of
        # ARM's carry; see eval_x86_cmp.) jc/jb therefore test c directly.
        "c": (_flag_c, False),
        "b": (_flag_c, False),
        "nae": (_flag_c, False),
        "nc": (_flag_c, True),
        "ae": (_flag_c, True),
        "nb": (_flag_c, True),
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

    # Instruction spec tables, grouped by operand behavior; parse() below
    # interprets these.

    # dst is read and written (register or memory), sources are read.
    # base mnemonic -> (flags behavior, value builder).
    instrs_alu: Dict[str, Tuple[str, AluBuilder]] = {
        "add": (FLAGS_ADD, lambda a, l, s: handle_add_real(l, s[0], a)),
        "adc": (FLAGS_ADD, adc_expr),
        "sub": (FLAGS_CMP, lambda a, l, s: sub_expr(l, s[0])),
        "sbb": (FLAGS_SBB, sbb_expr),
        "and": (FLAGS_LOGIC, lambda a, l, s: replace_bitand(BinaryOp.int(l, "&", s[0]))),
        "or": (FLAGS_LOGIC, lambda a, l, s: handle_or(l, s[0])),
        "xor": (FLAGS_LOGIC, lambda a, l, s: BinaryOp.int(l, "^", s[0])),
        "shl": (
            FLAGS_LOGIC,
            lambda a, l, s: fold_mul_chains(BinaryOp.int(l, "<<", as_intish(s[0]))),
        ),
        "sal": (
            FLAGS_LOGIC,
            lambda a, l, s: fold_mul_chains(BinaryOp.int(l, "<<", as_intish(s[0]))),
        ),
        "shr": (FLAGS_LOGIC, lambda a, l, s: shift_right_expr(l, s[0], signed=False)),
        "sar": (FLAGS_LOGIC, lambda a, l, s: shift_right_expr(l, s[0], signed=True)),
        # Rotates only affect the carry/overflow flags on real hardware; we
        # treat the flags as clobbered.
        "rol": (
            FLAGS_CLOBBER,
            lambda a, l, s: fn_op("ROTATE_LEFT", [l, as_intish(s[0])], Type.intish()),
        ),
        "ror": (
            FLAGS_CLOBBER,
            lambda a, l, s: fn_op("ROTATE_RIGHT", [l, as_intish(s[0])], Type.intish()),
        ),
        "shrd": (FLAGS_LOGIC, shrd_expr),
        "shld": (FLAGS_LOGIC, shld_expr),
    }
    # Shift/rotate instructions, whose count operand may be `cl` (making the
    # width suffix meaningless) and is never sign-extended.
    instrs_shift: Set[str] = {"shl", "sal", "shr", "sar", "rol", "ror", "shrd", "shld"}

    # single operand, read and written.
    instrs_unary: Dict[str, Tuple[str, UnaryBuilder]] = {
        "inc": (FLAGS_KEEP_C, inc_expr),
        "dec": (FLAGS_KEEP_C, lambda a, v: sub_expr(v, Literal(1))),
        # neg's flags are those of `cmp 0, v`, computed in the eval fn (in
        # particular c = (v != 0), matching x86's CF after neg).
        "neg": (FLAGS_CMP, lambda a, v: neg_expr(v)),
        "not": (FLAGS_NONE, lambda a, v: handle_bitinv(v)),
        "bswap": (FLAGS_NONE, lambda a, v: fn_op("BSWAP32", [v], Type.intish())),
    }

    # dst is written only (not read), src is read; no flags.
    instrs_dst_write: Set[str] = {"mov", "movsx", "movzx", "lea"}
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
        "repe.cmpsw": ([ESI, EDI, ECX], [ESI, EDI, ECX, Register("z")], True, False),
        "repe.cmpsd": ([ESI, EDI, ECX], [ESI, EDI, ECX, Register("z")], True, False),
    }
    # Fictive x87 mnemonics produced by X86FpuRewritePattern, dispatched to
    # _parse_fpu. (Raw x87 forms -- which use st(i) names, never f0..f7 --
    # reach `parse` only during initial parsing and fall through to the
    # unknown-instruction handler; the FPU prepass then rewrites them.)
    instrs_fpu: Set[str] = {
        "fld", "fild", "fld1", "fldz", "fldpi", "fldl2e", "fldl2t", "fldlg2",
        "fldln2", "fmov", "fmovpop", "fpop", "fst", "fstp", "fistp",
        "fstparg", "fstarg",
        "fadd", "fsub", "fsubr", "fmul", "fdiv", "fdivr",
        "faddp", "fsubp", "fsubrp", "fmulp", "fdivp", "fdivrp",
        "fiadd", "fisub", "fisubr", "fimul", "fidiv", "fidivr",
        "fchs", "fabs", "fsqrt", "fsin", "fcos", "fxch",
        "fcom", "fcomp", "fcompp", "fucom", "fucomp", "fucompp", "ftst",
        "ficom", "ficomp", "frndint", "fscale", "f2xm1", "fprem", "fprem1", "fpatan",
        "fyl2x", "fyl2xp1", "fxam",
    }
    # Non-rep string instructions (single element): mnemonic -> (inputs,
    # outputs, load, store).
    instrs_string_single: Dict[
        str, Tuple[List[Register], List[Register], bool, bool]
    ] = {
        "stosd": ([EDI, EAX], [EDI], False, True),
        "stosw": ([EDI, EAX], [EDI], False, True),
        "stosb": ([EDI, EAX], [EDI], False, True),
        "movsd": ([ESI, EDI], [ESI, EDI], True, True),
        "movsw": ([ESI, EDI], [ESI, EDI], True, True),
        "movsb": ([ESI, EDI], [ESI, EDI], True, True),
    }

    @classmethod
    def _unsupported_eval(
        cls, instr_str: str, reason: str = "phase 2b"
    ) -> Callable[[NodeState, InstrArgs], object]:
        def eval_fn(s: NodeState, a: InstrArgs) -> None:
            raise DecompFailure(
                f"x86 instruction evaluation is not implemented yet ({reason}): {instr_str}"
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

    @classmethod
    def _flag_outputs(
        cls, flags_kind: str
    ) -> Tuple[List[Register], List[Register]]:
        """(outputs, clobbers) among the flag registers for a flags kind."""
        if flags_kind == FLAGS_NONE:
            return [], []
        if flags_kind == FLAGS_CLOBBER:
            return [], list(cls.flag_regs)
        if flags_kind == FLAGS_KEEP_C:
            # inc/dec preserve the carry flag.
            return [r for r in cls.flag_regs if r != cls._flag_c], []
        return list(cls.flag_regs), []

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

        def write_dst(
            s: NodeState, a: InstrArgs, val: Expression, store_type: Type
        ) -> Optional[Expression]:
            """Write `val` to the first operand. Returns the (wrapped) value
            for register destinations, None for memory destinations."""
            dst = args[0]
            if isinstance(dst, Register):
                return s.set_reg(dst, val)
            src_reg = args[1] if len(args) > 1 and isinstance(args[1], Register) else None
            store = mem_store(a, 0, val, src_reg, store_type)
            if store is not None:
                # The register argument to store_memory is only used for
                # stack spill/restore bookkeeping; fall back to EAX for
                # register-less sources (immediates).
                s.store_memory(store, src_reg if src_reg is not None else EAX)
            return None

        if base == "ret":
            assert len(args) <= 1, "ret takes at most one (immediate) operand"
            inputs = [cls.stack_pointer_reg]
            is_return = True
            eval_fn = None
        elif base == "jmp":
            assert len(args) == 1
            target = args[0]
            if isinstance(target, Register):
                # Indirect jump through a register (computed goto).
                inputs = [target]
                jump_target = target
                is_conditional = True
                eval_fn = lambda s, a: s.set_switch_expr(a.regs[target])
            elif isinstance(target, AsmAddressMode):
                src_operand(target)
                regs = [loc for loc in inputs if isinstance(loc, Register)]
                if regs:
                    # Jump through memory, e.g. `jmp [eax*4 + switchdata]`:
                    # a jump table. Treated like an indirect jump through the
                    # index register; the loaded value drives the switch.
                    jump_target = regs[0]
                    is_conditional = True

                    def eval_jmp_table(s: NodeState, a: InstrArgs) -> None:
                        expr = mem_load(a, 0, Type.reg32(likely_float=False))
                        s.set_switch_expr(expr)

                    eval_fn = eval_jmp_table
                else:
                    # Register-less jump through an absolute address, e.g.
                    # `jmp [__imp__GetTickCount]`: a tail call through an
                    # import thunk.
                    outputs = list(cls.all_return_regs)
                    clobbers = list(cls.temp_regs)
                    function_target = target
                    is_return = True

                    def eval_jmp_import(s: NodeState, a: InstrArgs) -> None:
                        fn = mem_load(a, 0, Type.reg32(likely_float=False))
                        s.make_function_call(fn, outputs)

                    eval_fn = eval_jmp_import
            else:
                jump_target = get_jump_target(target)
                eval_fn = None
        elif base == "tailcall.fictive":
            # `jmp` to a label outside the function (see the stack rewrite
            # pass): call it and return its return value.
            assert len(args) == 1
            outputs = list(cls.all_return_regs)
            clobbers = list(cls.temp_regs)
            function_target = args[0]
            is_return = True
            eval_fn = lambda s, a: s.make_function_call(a.sym_imm(0), outputs)
        elif base.startswith("j") and base[1:] in cls.condition_flags:
            assert len(args) == 1
            flag, negated = cls.condition_flags[base[1:]]
            inputs = [flag]
            jump_target = get_jump_target(args[0])
            is_conditional = True

            def eval_jcc(s: NodeState, a: InstrArgs) -> None:
                cond = condition_from_expr(a.regs[flag])
                if negated:
                    cond = cond.negated()
                s.set_branch_condition(cond)

            eval_fn = eval_jcc
        elif base == "loop":
            assert len(args) == 1
            inputs = [ECX]
            outputs = [ECX]
            jump_target = get_jump_target(args[0])
            is_conditional = True

            def eval_loop(s: NodeState, a: InstrArgs) -> None:
                val = s.set_reg(ECX, sub_expr(a.regs[ECX], Literal(1)))
                s.set_branch_condition(BinaryOp.icmp(val, "!=", Literal(0)))

            eval_fn = eval_loop
        elif base == "call":
            # The stack rewrite pass appends two literals: the frame location
            # of [esp] at call time (the base of this call's stack argument
            # region) and the number of argument bytes (-1 if unknown). The FPU
            # prepass may append two more: the x87 virtual register the callee
            # pushes as a float return (fpret, or -1), and the one it consumes
            # off the stack as an argument (fconsume, or -1).
            assert len(args) in (1, 3, 5)
            target = args[0]
            arg_base: Optional[int] = None
            arg_bytes = -1
            fpret = -1
            fconsume = -1
            if len(args) >= 3:
                assert isinstance(args[1], AsmLiteral) and isinstance(
                    args[2], AsmLiteral
                )
                arg_base = args[1].value
                arg_bytes = args[2].value
            if len(args) == 5:
                assert isinstance(args[3], AsmLiteral) and isinstance(
                    args[4], AsmLiteral
                )
                fpret = args[3].value
                fconsume = args[4].value
            inputs = list(cls.argument_regs)
            outputs = list(cls.all_return_regs)
            clobbers = list(cls.temp_regs)
            function_target = target
            if fpret >= 0:
                # The callee returns its result in st(0), which becomes a new
                # virtual register the call defines. A float/double-returning
                # callee does NOT return anything in eax/edx (they are merely
                # clobbered, via temp_regs), so drop them from the return
                # outputs -- otherwise the return-register heuristic would
                # prefer eax and bit-reinterpret it as the float result.
                outputs = [Register(f"f{fpret}")]
            if fconsume >= 0:
                # The callee consumes st(0) as an argument; it is live into the
                # call (passed as an argument) and killed by it.
                consumed = Register(f"f{fconsume}")
                inputs.append(consumed)
                clobbers.append(consumed)
            if isinstance(target, Register):
                inputs.append(target)
            elif isinstance(target, AsmAddressMode):
                src_operand(target)
            elif not isinstance(target, (AsmGlobalSymbol, AsmLiteral)):
                raise DecompFailure(f"Invalid x86 call target in `{instr_str}`")

            def eval_call(s: NodeState, a: InstrArgs) -> None:
                if isinstance(target, Register):
                    fn: Expression = a.regs[target]
                elif isinstance(target, AsmAddressMode):
                    fn = mem_load(a, 0, Type.reg32(likely_float=False))
                else:
                    fn = a.sym_imm(0)
                if fconsume >= 0 and arg_base is not None:
                    # Pass the consumed st(0) value as this call's argument (an
                    # ftol-style helper takes exactly one float in st0).
                    consumed = Register(f"f{fconsume}")
                    s.subroutine_args[arg_base] = as_type(
                        a.regs[consumed], Type.floatish(), silent=True
                    )
                    del s.regs[consumed]
                if arg_base is None:
                    s.make_function_call(fn, outputs)
                    return
                # Select the pending stack argument stores belonging to this
                # call, and re-key them to the offsets used by function_abi
                # (+4: the ABI is described from the callee's point of view,
                # where [esp] holds the return address).
                leftover = {
                    loc: val
                    for loc, val in s.subroutine_args.items()
                    if loc < arg_base or (arg_bytes >= 0 and loc >= arg_base + arg_bytes)
                }
                selected = {
                    loc - arg_base + 4: val
                    for loc, val in s.subroutine_args.items()
                    if loc not in leftover
                }
                s.subroutine_args.clear()
                s.subroutine_args.update(selected)
                s.make_function_call(fn, outputs)
                # Argument stores for a later call (pushed before this call's
                # arguments) stay pending.
                s.subroutine_args.update(leftover)

            eval_fn = eval_call
        elif base == "storearg.fictive":
            # A rewritten `push` that passes a stack argument to the next
            # `call` (see the stack rewrite pass). args[0] is the frame
            # location of the argument slot.
            assert len(args) == 2 and isinstance(args[0], AsmLiteral)
            arg_loc = args[0].value
            src_operand(args[1])
            outputs = [StackLocation(offset=arg_loc, symbolic_offset=None)]
            is_store = True

            def eval_storearg(s: NodeState, a: InstrArgs) -> None:
                s.subroutine_args[arg_loc] = op_value(a, 1, 4)

            eval_fn = eval_storearg
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

            def eval_dst_write(s: NodeState, a: InstrArgs) -> None:
                dst, src = args
                if base == "lea":
                    assert isinstance(dst, Register) and isinstance(
                        src, AsmAddressMode
                    ), f"bad lea operands in `{instr_str}`"
                    if src.base is not None and isinstance(src.addend, AsmLiteral):
                        # Plain base + offset; for esp-relative addresses this
                        # takes the address of a stack variable.
                        val = handle_addi_real(
                            dst, src.base, a.regs[src.base], Literal(src.addend.value), a
                        )
                    else:
                        addend = address_expr(src.addend, a)
                        if src.base is not None:
                            val = handle_add_real(a.regs[src.base], addend, a)
                        else:
                            val = fold_mul_chains(addend)
                    s.set_reg(dst, val)
                    return
                if base in ("movsx", "movzx"):
                    # The width suffix reflects the (narrower) source operand.
                    assert isinstance(dst, Register)
                    assert width in (1, 2), f"bad {base} source width"
                    if base == "movsx":
                        tp = Type.s8() if width == 1 else Type.s16()
                    else:
                        tp = Type.u8() if width == 1 else Type.u16()
                    if isinstance(src, Register):
                        val = as_type(a.regs[src], tp, silent=False, unify=False)
                    else:
                        val = mem_load(a, 1, tp)
                    s.set_reg(dst, val)
                    return
                # mov. Only sign-extend full-width immediates; a byte
                # immediate like `mov cl, 0xff` reads better unsigned.
                val = op_value(a, 1, width, sign_extend_imm=(width == 4))
                if (
                    isinstance(dst, Register)
                    and isinstance(src, Register)
                    and a.stack_info.is_stack_reg(src)
                    and not a.stack_info.is_stack_reg(dst)
                ):
                    # `mov reg, esp`: taking the address of the stack.
                    val = handle_addi_real(dst, src, val, Literal(0), a)
                write_dst(s, a, val, width_type(width))

            eval_fn = eval_dst_write
        elif base in cls.instrs_alu:
            assert len(args) in (2, 3)  # shrd/shld take three operands
            dst = args[0]
            for src in args[1:]:
                src_operand(src)
            dest_operand(dst, also_read=True)
            flags_kind, alu_builder = cls.instrs_alu[base]
            if base in ("adc", "sbb"):
                inputs.append(cls._flag_c)
            flag_outs, flag_clobbers = cls._flag_outputs(flags_kind)
            outputs.extend(flag_outs)
            clobbers.extend(flag_clobbers)
            is_effectful = is_store

            def eval_alu(s: NodeState, a: InstrArgs) -> None:
                w = width
                if base in cls.instrs_shift and isinstance(args[-1], Register):
                    # The width suffix came from a `cl` shift count; the
                    # shift itself operates on the full destination.
                    w = 4
                if (
                    base in ("add", "sub")
                    and isinstance(args[0], Register)
                    and a.stack_info.is_stack_reg(args[0])
                    and isinstance(args[1], AsmLiteral)
                ):
                    # Stack pointer adjustment (prologue/epilogue). The frame
                    # size is tracked by get_stack_info; leave esp alone.
                    return
                if base == "xor" and args[0] == args[1]:
                    # xor r, r: the idiom for zeroing a register.
                    assert isinstance(args[0], Register)
                    zero = s.set_reg(args[0], Literal(0))
                    set_x86_flags_from_result(s, zero, w)
                    return
                lhs = op_value(a, 0, w)
                sign_ext = base not in cls.instrs_shift
                srcs = [
                    op_value(a, i, w, sign_extend_imm=sign_ext)
                    for i in range(1, len(args))
                ]
                # Build the new value before touching the flags: sbb (and adc)
                # read the incoming carry flag for their value, and the
                # FLAGS_SBB flag write below overwrites `c` with sbb's own
                # borrow. Computing the value only reads registers (no state
                # change), so ordering it first is safe for every op and is
                # what lets sbb see the pre-instruction carry rather than the
                # borrow it is about to produce.
                val = alu_builder(a, lhs, srcs)

                if flags_kind == FLAGS_CMP:
                    # Compare-style flags are based on the values *before*
                    # the destination is overwritten.
                    eval_x86_cmp(s, lhs, srcs[0], w)
                elif flags_kind == FLAGS_SBB:
                    # sbb: subtract-with-borrow flags = flags of
                    # lhs - (src + carry-in), a compare (c is a borrow), also
                    # taken before the destination is overwritten.
                    eval_x86_cmp(
                        s, lhs, BinaryOp.intptr(srcs[0], "+", carry_in(a)), w
                    )

                def set_alu_flags(result: Expression) -> None:
                    if flags_kind == FLAGS_ADD:
                        set_x86_flags_from_add(s, lhs, result, w)
                    elif flags_kind == FLAGS_LOGIC:
                        set_x86_flags_from_result(s, result, w)

                if isinstance(args[0], Register):
                    val = s.set_reg(args[0], val)
                    set_alu_flags(val)
                else:
                    # For memory destinations, set flags before the store so
                    # that flag expressions refer to pre-store values.
                    set_alu_flags(val)
                    write_dst(s, a, val, width_type(w))

            eval_fn = eval_alu
        elif base in cls.instrs_unary:
            assert len(args) == 1
            dest_operand(args[0], also_read=True)
            if isinstance(args[0], AsmAddressMode):
                is_load = True
            elif isinstance(args[0], Register) and args[0] not in inputs:
                inputs.append(args[0])
            flags_kind, unary_builder = cls.instrs_unary[base]
            if flags_kind == FLAGS_KEEP_C:
                # inc/dec preserve the carry flag but fold it into the composite
                # unsigned-above predicate (ja/jbe), so they read it.
                inputs.append(cls._flag_c)
            flag_outs, flag_clobbers = cls._flag_outputs(flags_kind)
            outputs.extend(flag_outs)
            clobbers.extend(flag_clobbers)
            is_effectful = is_store

            def eval_unary(s: NodeState, a: InstrArgs) -> None:
                old = op_value(a, 0, width)
                if flags_kind == FLAGS_CMP:
                    # neg: flags of `cmp 0, old` (c = borrow = (old != 0)).
                    eval_x86_cmp(s, Literal(0), old, width)
                # inc/dec keep CF; fold the preserved carry into the composite
                # unsigned-above predicate (read before it can be overwritten).
                keep_carry = a.regs[cls._flag_c] if flags_kind == FLAGS_KEEP_C else None

                def set_unary_flags(result: Expression) -> None:
                    if flags_kind == FLAGS_KEEP_C:
                        set_x86_flags_from_result(
                            s, result, width, set_c_v=False, preserved_carry=keep_carry
                        )
                        s.set_reg(
                            cls._flag_v,
                            fn_op("M2C_OVERFLOW", [result], Type.boolean()),
                        )

                val = unary_builder(a, old)
                if isinstance(args[0], Register):
                    val = s.set_reg(args[0], val)
                    set_unary_flags(val)
                else:
                    set_unary_flags(val)
                    write_dst(s, a, val, width_type(width))

            eval_fn = eval_unary
        elif base in cls.instrs_cmp:
            assert len(args) == 2
            for arg in args:
                src_operand(arg)
            outputs = list(cls.flag_regs)
            is_effectful = False

            def eval_cmp(s: NodeState, a: InstrArgs) -> None:
                if (
                    base == "test"
                    and width == 1
                    and args[0] == EAX
                    and isinstance(args[1], AsmLiteral)
                    and EAX in s.regs
                ):
                    # `test ah, mask` after `fnstsw ax`: if eax still holds the
                    # x87 status-word marker, translate the compare directly to
                    # a float condition instead of materializing bit arithmetic
                    # on the status word (see §3.2).
                    operands = fnstsw_marker_operands(a.regs[EAX])
                    op = FNSTSW_MASK_OPS.get(args[1].value & 0xFF)
                    if operands is not None and op is not None:
                        cond = fpu_compare_condition(operands[0], operands[1], op)
                        for flag in cls.flag_regs:
                            s.set_reg(flag, cond)
                        return
                lhs = op_value(a, 0, width)
                if base == "test":
                    # CF = OF = 0; other flags are based on lhs & rhs. The
                    # dominant idiom `test r, r` just compares r against 0.
                    if args[0] == args[1]:
                        val = lhs
                    else:
                        rhs = op_value(a, 1, width)
                        val = replace_bitand(BinaryOp.int(lhs, "&", rhs))
                    set_x86_flags_from_result(s, val, width)
                else:
                    rhs = op_value(a, 1, width)
                    eval_x86_cmp(s, lhs, rhs, width)

            eval_fn = eval_cmp
        elif base.startswith("set") and base[3:] in cls.condition_flags:
            assert len(args) == 1
            flag, negated = cls.condition_flags[base[3:]]
            inputs = [flag]
            dest_operand(args[0], also_read=False)
            is_effectful = is_store

            def eval_setcc(s: NodeState, a: InstrArgs) -> None:
                cond = condition_from_expr(a.regs[flag])
                if negated:
                    cond = cond.negated()
                # setcc writes a 0/1 byte; for register destinations this is
                # modeled as writing the full register (usually zeroed
                # beforehand by `xor r, r`).
                val = Cast(expr=cond, reinterpret=False, silent=True, type=Type.u8())
                write_dst(s, a, val, Type.int_of_size(8))

            eval_fn = eval_setcc
        elif base == "cdq":
            assert not args
            inputs = [EAX]
            outputs = [EDX]
            is_effectful = False
            eval_fn = lambda s, a: s.set_reg(
                EDX, BinaryOp.sint(a.regs[EAX], ">>", Literal(31))
            )
        elif base == "rdtsc":
            assert not args
            outputs = [EAX, EDX]
            clobbers = list(cls.flag_regs)
            is_effectful = False

            def eval_rdtsc(s: NodeState, a: InstrArgs) -> None:
                val = fn_op("M2C_RDTSC", [], Type.u64())
                s.set_reg(EAX, val)
                s.set_reg(EDX, fn_op("SECOND_REG", [val], Type.reg32(likely_float=False)))

            eval_fn = eval_rdtsc
        elif base in cls.instrs_string_single and not args:
            # Non-rep string instructions perform a single element operation
            # and advance esi/edi (assuming the direction flag is clear).
            str_inputs, str_outputs, is_load, is_store = cls.instrs_string_single[base]
            inputs = list(str_inputs)
            outputs = list(str_outputs)
            elem = {"b": 1, "w": 2, "d": 4}[base[-1]]

            def eval_string_single(s: NodeState, a: InstrArgs) -> None:
                op = base[:-1]
                tp = width_type(elem)
                if op in ("stos", "movs"):
                    if op == "stos":
                        value: Expression = a.regs[EAX]
                        if elem < 4:
                            value = as_type(value, tp, silent=True, unify=False)
                    else:
                        value = deref(a.regs[ESI], a.regs, a.stack_info, size=elem)
                    dest = deref(a.regs[EDI], a.regs, a.stack_info, size=elem, store=True)
                    dest.type.unify(tp)
                    s.write_statement(StoreStmt(source=as_type(value, tp, silent=False), dest=dest))
                    s.set_reg(EDI, BinaryOp.intptr(a.regs[EDI], "+", Literal(elem)))
                    if op == "movs":
                        s.set_reg(ESI, BinaryOp.intptr(a.regs[ESI], "+", Literal(elem)))
                else:
                    raise DecompFailure(f"x86 `{instr_str}` is not supported")

            eval_fn = eval_string_single
        elif base in ("mul", "imul", "div", "idiv") and len(args) <= 1:
            # One-operand forms operate on edx:eax.
            inputs = [EAX] if base in ("mul", "imul") else [EAX, EDX]
            if args:
                src_operand(args[0])
            outputs = [EAX, EDX]
            clobbers = list(cls.flag_regs)
            is_effectful = False

            def eval_muldiv(s: NodeState, a: InstrArgs) -> None:
                if not args:
                    raise DecompFailure(f"x86 `{instr_str}` is missing its operand")
                src = op_value(a, 0, width)
                acc = a.regs[EAX]
                if base in ("mul", "imul"):
                    hi_op = "MULT_HI" if base == "imul" else "MULTU_HI"
                    s.set_reg(EAX, BinaryOp.int(acc, "*", src))
                    s.set_reg(EDX, fold_divmod(BinaryOp.int(acc, hi_op, src)))
                else:
                    # The 64-bit dividend edx:eax is assumed to be the
                    # sign/zero-extension of eax (set up via cdq or
                    # `xor edx, edx`), which is how compilers emit 32-bit
                    # division; edx's value is not consulted.
                    if base == "idiv":
                        quot = BinaryOp.sint(acc, "/", src)
                        rem = BinaryOp.sint(acc, "%", src)
                    else:
                        quot = BinaryOp.uint(acc, "/", src)
                        rem = BinaryOp.uint(acc, "%", src)
                    s.set_reg(EAX, quot)
                    s.set_reg(EDX, rem)

            eval_fn = eval_muldiv
        elif base == "imul":
            # Two/three-operand forms only write the destination register.
            assert len(args) in (2, 3) and isinstance(args[0], Register)
            inputs = [args[0]] if len(args) == 2 else []
            for src in args[1:]:
                src_operand(src)
            outputs = [args[0]]
            clobbers = list(cls.flag_regs)
            is_effectful = False

            def eval_imul(s: NodeState, a: InstrArgs) -> None:
                assert isinstance(args[0], Register)
                if len(args) == 2:
                    lhs: Expression = a.regs[args[0]]
                    rhs = op_value(a, 1, width)
                else:
                    lhs = op_value(a, 1, width)
                    rhs = op_value(a, 2, width)
                s.set_reg(args[0], fold_mul_chains(BinaryOp.int(lhs, "*", rhs)))

            eval_fn = eval_imul
        elif base == "xchg":
            assert len(args) == 2
            for arg in args:
                src_operand(arg)
            for arg in args:
                dest_operand(arg, also_read=True)
            is_effectful = is_store

            def eval_xchg(s: NodeState, a: InstrArgs) -> None:
                vals = [op_value(a, 0, width), op_value(a, 1, width)]
                for i in (0, 1):
                    dst = args[i]
                    val = vals[1 - i]
                    if isinstance(dst, Register):
                        s.set_reg(dst, val)
                    else:
                        other = args[1 - i]
                        src_reg = other if isinstance(other, Register) else None
                        store = mem_store(a, i, val, src_reg, width_type(width))
                        if store is not None:
                            s.store_memory(
                                store, src_reg if src_reg is not None else EAX
                            )

            eval_fn = eval_xchg
        elif base == "mov.fs" and len(args) == 2:
            # An fs-segment absolute access (on Win32, the Thread Information
            # Block). The canonical SEH-chain bookkeeping at fs:[0] is
            # removed by X86SehPattern before translation; any other access
            # (e.g. the stack bounds at fs:[4]/fs:[8]) is modeled as an
            # explicit opaque platform read/write.
            fs_dst, fs_src = args
            if (
                isinstance(fs_dst, Register)
                and isinstance(fs_src, AsmAddressMode)
                and fs_src.base is None
                and isinstance(fs_src.addend, AsmLiteral)
            ):
                offset = fs_src.addend.value
                outputs = [fs_dst]
                is_load = True

                def eval_fs_load(s: NodeState, a: InstrArgs) -> None:
                    assert isinstance(fs_dst, Register)
                    s.set_reg(
                        fs_dst,
                        fn_op("M2C_FS_LOAD", [Literal(offset)], width_type(width)),
                    )

                eval_fn = eval_fs_load
            elif (
                isinstance(fs_dst, AsmAddressMode)
                and fs_dst.base is None
                and isinstance(fs_dst.addend, AsmLiteral)
            ):
                store_offset = fs_dst.addend.value
                src_operand(fs_src)
                is_store = True

                def eval_fs_store(s: NodeState, a: InstrArgs) -> None:
                    value = op_value(a, 1, width)
                    s.write_statement(
                        void_fn_op("M2C_FS_STORE", [Literal(store_offset), value])
                    )

                eval_fn = eval_fs_store
        elif mnemonic in cls.instrs_string:
            str_inputs, str_outputs, is_load, is_store = cls.instrs_string[mnemonic]
            inputs = list(str_inputs)
            outputs = list(str_outputs)
            elem_size = {"b": 1, "w": 2, "d": 4}[mnemonic[-1]]

            def eval_string_op(s: NodeState, a: InstrArgs) -> None:
                count = as_intish(a.regs[ECX])
                if elem_size != 1:
                    count = fold_mul_chains(
                        BinaryOp.int(count, "*", Literal(elem_size))
                    )
                op = mnemonic.split(".")[1][:-1] if "." in mnemonic else ""
                if op == "movs":
                    # rep movsX: copy ecx elements from [esi] to [edi].
                    s.write_statement(
                        void_fn_op(
                            "M2C_MEMCPY", [a.regs[EDI], a.regs[ESI], count]
                        )
                    )
                    s.set_reg(ESI, BinaryOp.intptr(a.regs[ESI], "+", count))
                    s.set_reg(EDI, BinaryOp.intptr(a.regs[EDI], "+", count))
                    s.set_reg(ECX, Literal(0))
                elif op == "stos":
                    # rep stosX: fill ecx elements at [edi] with al/ax/eax.
                    value = a.regs[EAX]
                    if elem_size < 4:
                        value = as_type(
                            value, width_type(elem_size), silent=True, unify=False
                        )
                    fn_name = {1: "M2C_MEMSET", 2: "M2C_MEMSET16", 4: "M2C_MEMSET32"}[
                        elem_size
                    ]
                    s.write_statement(
                        void_fn_op(fn_name, [a.regs[EDI], value, as_intish(a.regs[ECX])])
                    )
                    s.set_reg(EDI, BinaryOp.intptr(a.regs[EDI], "+", count))
                    s.set_reg(ECX, Literal(0))
                elif op == "scas":
                    # repne scasb: scan [edi...] for the byte in al, leaving
                    # edi one past the match and decrementing ecx once per
                    # byte scanned. With al = 0 this is the strlen idiom; for
                    # other (or statically unknown) al values, model the scan
                    # as a memchr. Both assume the byte is found before ecx
                    # runs out, which holds for the compiler-emitted
                    # strlen/strcpy expansions this models.
                    al = early_unwrap(a.regs[EAX])
                    if isinstance(al, Literal) and al.value & 0xFF == 0:
                        length: Expression = fn_op(
                            "M2C_STRLEN", [a.regs[EDI]], Type.u32()
                        )
                    else:
                        needle = as_type(
                            a.regs[EAX], Type.u8(), silent=True, unify=False
                        )
                        found = fn_op(
                            "M2C_MEMCHR",
                            [a.regs[EDI], needle, as_intish(a.regs[ECX])],
                            Type.ptr(),
                        )
                        length = BinaryOp.intptr(found, "-", a.regs[EDI])
                    advance = BinaryOp.int(length, "+", Literal(1))
                    s.set_reg(
                        ECX, BinaryOp.int(a.regs[ECX], "-", advance)
                    )
                    s.set_reg(EDI, BinaryOp.intptr(a.regs[EDI], "+", advance))
                    s.set_reg(Register("z"), Literal(1, type=Type.boolean()))
                else:
                    # repe cmpsX: memcmp-style comparison; only the z flag
                    # outcome is modeled.
                    assert op == "cmps"
                    cmp = fn_op(
                        "M2C_MEMCMP", [a.regs[ESI], a.regs[EDI], count], Type.s32()
                    )
                    s.set_reg(ESI, BinaryOp.intptr(a.regs[ESI], "+", count))
                    s.set_reg(EDI, BinaryOp.intptr(a.regs[EDI], "+", count))
                    s.set_reg(ECX, Literal(0))
                    s.set_reg(Register("z"), BinaryOp.icmp(cmp, "==", Literal(0)))

            eval_fn = eval_string_op
        elif base in cls.instrs_ignore:
            is_effectful = False
            eval_fn = None
        elif base in ("fnstsw", "fstsw", "fldcw", "fstcw", "fnstcw") or (
            base in cls.instrs_fpu
            and any(isinstance(a, Register) and a.is_float() for a in args)
        ):
            # A fictive x87 instruction emitted by X86FpuRewritePattern (raw
            # x87 forms name st(i), never f0..f7, so the f-register gate
            # excludes them). The status/control-word ops have no f-register
            # operand, so they are dispatched by name.
            return self._parse_fpu(base, width, args, meta, instr_str)
        else:
            # Unknown instruction (x87 FPU, etc.). Guess a structural shape
            # so that file parsing and flow graph construction still work;
            # evaluation will fail with a clear error.
            eval_fn = cls._unsupported_eval(instr_str, "unknown instruction")
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

    def _parse_fpu(
        self,
        base: str,
        width: int,
        args: List[Argument],
        meta: InstructionMeta,
        instr_str: str,
    ) -> Instruction:
        """Build an Instruction for a fictive x87 op (see the FPU eval helpers
        above and m2c/x86_fpu.py). Virtual registers f0..f7 are ordinary
        registers to the rest of m2c; the only novelty is that popping ops
        *kill* their consumed register (drop it from outputs, list it in
        clobbers, and `del s.regs[...]` in eval) so a popped value stops
        existing -- which is what makes float-return detection sound."""
        cls = type(self)
        inputs: List[Location] = []
        outputs: List[Location] = []
        clobbers: List[Location] = []
        is_load = False
        is_store = False
        eval_fn: Optional[Callable[[NodeState, InstrArgs], object]]

        def add_operand_inputs(arg: Argument) -> None:
            for loc in cls._operand_inputs(arg):
                if loc not in inputs:
                    inputs.append(loc)

        def not_implemented(reason: str) -> Callable[[NodeState, InstrArgs], None]:
            for arg in args:
                add_operand_inputs(arg)

            def fail(s: NodeState, a: InstrArgs) -> None:
                raise DecompFailure(
                    f"x87 instruction evaluation is not implemented yet "
                    f"({reason}): {instr_str}"
                )

            return fail

        # --- Loads: dst register + memory source ---
        if base in ("fld", "fild"):
            dst = args[0]
            assert isinstance(dst, Register)
            add_operand_inputs(args[1])
            outputs = [dst]
            is_load = True
            if base == "fld":
                ftype = fpu_float_type(width)

                def eval_fld(s: NodeState, a: InstrArgs) -> None:
                    s.set_reg(dst, mem_load(a, 1, ftype))

                eval_fn = eval_fld
            else:
                itype = fpu_int_type(width)

                def eval_fild(s: NodeState, a: InstrArgs) -> None:
                    val = mem_load(a, 1, itype)
                    s.set_reg(dst, handle_convert(val, Type.floatish(), itype))

                eval_fn = eval_fild

        # --- Constants: 0/1 as numeric literals, pi/log constants as named
        # macros so matching source can #define them (spec Q5). ---
        elif base in FPU_CONSTANTS:
            dst = args[0]
            assert isinstance(dst, Register)
            outputs = [dst]
            make_const = FPU_CONSTANTS[base]

            def eval_fconst(s: NodeState, a: InstrArgs) -> None:
                s.set_reg(dst, make_const())

            eval_fn = eval_fconst

        # --- Register moves (fld/fst st(i), fstp st(i) with i>0) ---
        elif base in ("fmov", "fmovpop"):
            dst, src = args
            assert isinstance(dst, Register) and isinstance(src, Register)
            inputs = [src]
            outputs = [dst]
            pop = base == "fmovpop"
            if pop:
                clobbers = [src]

            def eval_fmov(s: NodeState, a: InstrArgs) -> None:
                s.set_reg(dst, a.regs[src])
                if pop:
                    del s.regs[src]

            eval_fn = eval_fmov

        # --- Float call arguments: fstp/fst into the next call's argument
        # window, routed to subroutine_args like a rewritten push (§4.5). ---
        elif base in ("fstparg", "fstarg"):
            assert isinstance(args[0], AsmLiteral) and isinstance(args[1], Register)
            arg_loc = args[0].value
            src = args[1]
            inputs = [src]
            outputs = [StackLocation(offset=arg_loc, symbolic_offset=None)]
            is_store = True
            pop = base == "fstparg"
            if pop:
                clobbers = [src]
            ftype = fpu_float_type(width)

            def eval_fstparg(s: NodeState, a: InstrArgs) -> None:
                s.subroutine_args[arg_loc] = as_type(a.regs[src], ftype, silent=True)
                if pop:
                    del s.regs[src]

            eval_fn = eval_fstparg

        # --- Pop-discard (fstp st(0)) ---
        elif base == "fpop":
            reg = args[0]
            assert isinstance(reg, Register)
            inputs = [reg]
            clobbers = [reg]

            def eval_fpop(s: NodeState, a: InstrArgs) -> None:
                del s.regs[reg]

            eval_fn = eval_fpop

        # --- Stores (fst/fstp to memory) ---
        elif base in ("fst", "fstp"):
            src = args[1]
            assert isinstance(src, Register)
            add_operand_inputs(args[0])
            if src not in inputs:
                inputs.append(src)
            stack_loc = cls._stack_location(args[0]) if isinstance(
                args[0], AsmAddressMode
            ) else None
            if stack_loc is not None:
                outputs.append(stack_loc)
            is_store = True
            pop = base == "fstp"
            if pop:
                clobbers = [src]
            ftype = fpu_float_type(width)

            def eval_fst(s: NodeState, a: InstrArgs) -> None:
                store = mem_store(a, 0, a.regs[src], src, ftype)
                if store is not None:
                    s.store_memory(store, src)
                if pop:
                    del s.regs[src]

            eval_fn = eval_fst

        # --- Non-popping arithmetic: dst register op src (register or mem) ---
        elif base in ("fadd", "fsub", "fsubr", "fmul", "fdiv", "fdivr"):
            dst, src = args
            assert isinstance(dst, Register)
            inputs = [dst]
            if isinstance(src, Register):
                if src not in inputs:
                    inputs.append(src)
            else:
                add_operand_inputs(src)
                is_load = True
            outputs = [dst]
            op, reverse = FPU_ARITH_OPS[base]
            ftype = fpu_float_type(width)

            def eval_arith(s: NodeState, a: InstrArgs) -> None:
                lhs = a.regs[dst]
                rhs = a.regs[src] if isinstance(src, Register) else mem_load(
                    a, 1, ftype
                )
                s.set_reg(dst, fpu_binop(op, lhs, rhs, reverse=reverse))

            eval_fn = eval_arith

        # --- Popping arithmetic (faddp st(i), st): dst op st0, then pop st0 ---
        elif base in ("faddp", "fsubp", "fsubrp", "fmulp", "fdivp", "fdivrp"):
            dst, src = args
            assert isinstance(dst, Register) and isinstance(src, Register)
            inputs = [dst, src]
            outputs = [dst]
            clobbers = [src]
            op, reverse = FPU_ARITH_OPS[base]

            def eval_arithp(s: NodeState, a: InstrArgs) -> None:
                lhs = a.regs[dst]
                rhs = a.regs[src]
                s.set_reg(dst, fpu_binop(op, lhs, rhs, reverse=reverse))
                del s.regs[src]

            eval_fn = eval_arithp

        # --- Unary operations on the top of stack ---
        elif base in FPU_UNARY_OPS:
            reg = args[0]
            assert isinstance(reg, Register)
            inputs = [reg]
            outputs = [reg]
            builder = FPU_UNARY_OPS[base]

            def eval_unary_fpu(s: NodeState, a: InstrArgs) -> None:
                s.set_reg(reg, builder(a.regs[reg]))

            eval_fn = eval_unary_fpu

        # --- fxch: swap two slots ---
        elif base == "fxch":
            ra, rb = args
            assert isinstance(ra, Register) and isinstance(rb, Register)
            inputs = [ra, rb]
            outputs = [ra, rb]

            def eval_fxch(s: NodeState, a: InstrArgs) -> None:
                va = a.regs[ra]
                vb = a.regs[rb]
                s.set_reg(ra, vb)
                s.set_reg(rb, va)

            eval_fn = eval_fxch

        # --- Compares: store a symbolic status-word marker into `fsw`, killing
        # any popped operands. The fnstsw/test-ah idiom below consumes it. ---
        elif base in ("fcom", "fcomp", "fucom", "fucomp", "fcompp", "fucompp",
                      "ftst", "ficom", "ficomp"):
            top = args[0]
            assert isinstance(top, Register)
            inputs = [top]
            popped: List[Register] = []
            if base in ("fcomp", "fucomp", "ficomp"):
                popped = [top]
            elif base in ("fcompp", "fucompp"):
                assert isinstance(args[1], Register)
                popped = [top, args[1]]
            clobbers = list(popped)
            outputs = [cls.fsw_reg]
            rhs_arg = args[1] if len(args) > 1 else None
            if isinstance(rhs_arg, Register):
                if rhs_arg not in inputs:
                    inputs.append(rhs_arg)
            elif rhs_arg is not None:
                add_operand_inputs(rhs_arg)
                is_load = True
            is_int_cmp = base in ("ficom", "ficomp")
            itype = fpu_int_type(width) if is_int_cmp else None

            def eval_fcom(s: NodeState, a: InstrArgs) -> None:
                lhs = a.regs[top]
                if base == "ftst":
                    rhs: Expression = f32_literal(0.0)
                elif is_int_cmp:
                    assert itype is not None
                    rhs = handle_convert(mem_load(a, 1, itype), Type.floatish(), itype)
                elif isinstance(rhs_arg, Register):
                    rhs = a.regs[rhs_arg]
                else:
                    rhs = mem_load(a, 1, fpu_float_type(width))
                s.set_reg(cls.fsw_reg, fn_op(FNSTSW_MARKER, [lhs, rhs], Type.u16()))
                for reg in popped:
                    del s.regs[reg]

            eval_fn = eval_fcom

        # --- fnstsw ax: move the status-word marker into eax for the test. ---
        elif base in ("fnstsw", "fstsw"):
            assert isinstance(args[0], Register)
            eax = args[0]
            inputs = [cls.fsw_reg]
            outputs = [eax]

            def eval_fnstsw(s: NodeState, a: InstrArgs) -> None:
                if cls.fsw_reg in s.regs:
                    s.set_reg(eax, s.regs[cls.fsw_reg])
                else:
                    # A stray fnstsw with no preceding compare: surface it.
                    s.set_reg(eax, fn_op(FNSTSW_MARKER, [], Type.u16()))

            eval_fn = eval_fnstsw

        # --- fistp: store the top as an integer (truncating cast), then pop.
        # The rounding mode is set globally in this corpus (see spec §5.2), so
        # a C truncation cast matches the game's ambient chop mode. ---
        elif base == "fistp":
            src = args[1]
            assert isinstance(src, Register)
            add_operand_inputs(args[0])
            if src not in inputs:
                inputs.append(src)
            stack_loc = cls._stack_location(args[0]) if isinstance(
                args[0], AsmAddressMode
            ) else None
            if stack_loc is not None:
                outputs.append(stack_loc)
            is_store = True
            clobbers = [src]
            itype = fpu_int_type(width)

            def eval_fistp(s: NodeState, a: InstrArgs) -> None:
                casted = handle_convert(a.regs[src], itype, Type.floatish())
                store = mem_store(a, 0, casted, None, itype)
                if store is not None:
                    s.store_memory(store, src)
                del s.regs[src]

            eval_fn = eval_fistp

        # --- Integer-operand arithmetic: top op (float)int_load ---
        elif base in ("fiadd", "fisub", "fisubr", "fimul", "fidiv", "fidivr"):
            dst = args[0]
            assert isinstance(dst, Register)
            inputs = [dst]
            add_operand_inputs(args[1])
            outputs = [dst]
            is_load = True
            op, reverse = FPU_ARITH_OPS[base]
            itype = fpu_int_type(width)

            def eval_iarith(s: NodeState, a: InstrArgs) -> None:
                lhs = a.regs[dst]
                rhs = handle_convert(mem_load(a, 1, itype), Type.floatish(), itype)
                s.set_reg(dst, fpu_binop(op, lhs, rhs, reverse=reverse))

            eval_fn = eval_iarith

        # --- Control word: kept as visible intrinsics (see spec §5.2). The
        # rounding/precision mode is not modeled, so surface the load/store so
        # a human sees the mode changes rather than pretending they vanish. ---
        elif base == "fldcw":
            add_operand_inputs(args[0])
            is_load = True

            def eval_fldcw(s: NodeState, a: InstrArgs) -> None:
                s.write_statement(
                    void_fn_op("M2C_FLDCW", [mem_load(a, 0, Type.u16())])
                )

            eval_fn = eval_fldcw
        elif base in ("fstcw", "fnstcw"):
            add_operand_inputs(args[0])
            stack_loc = cls._stack_location(args[0]) if isinstance(
                args[0], AsmAddressMode
            ) else None
            if stack_loc is not None:
                outputs.append(stack_loc)
            is_store = True

            def eval_fstcw(s: NodeState, a: InstrArgs) -> None:
                store = mem_store(
                    a, 0, fn_op("M2C_FSTCW", [], Type.u16()), None, Type.u16()
                )
                if store is not None:
                    s.store_memory(store, EAX)

            eval_fn = eval_fstcw

        # --- Two-operand transcendentals (fpatan/fyl2x/fscale/fprem/...) ---
        elif base in FPU_BINARY_OPS:
            st0, st1 = args
            assert isinstance(st0, Register) and isinstance(st1, Register)
            builder, dst_idx, pop = FPU_BINARY_OPS[base]
            dst = st1 if dst_idx == 1 else st0
            inputs = [st0, st1] if st0 != st1 else [st0]
            outputs = [dst]
            if pop:
                clobbers = [st0]  # the top is consumed

            def eval_binary_fpu(s: NodeState, a: InstrArgs) -> None:
                val = builder(a.regs[st0], a.regs[st1])
                s.set_reg(dst, val)
                if pop:
                    del s.regs[st0]

            eval_fn = eval_binary_fpu

        # --- Anything left really is unhandled: fail cleanly. ---
        else:
            eval_fn = not_implemented("x87 op")

        return Instruction(
            mnemonic=base if width == 4 else base + WIDTH_SUFFIXES[width],
            args=args,
            meta=meta,
            inputs=inputs,
            clobbers=clobbers,
            outputs=outputs,
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=False,
            is_store=is_store,
            is_load=is_load,
            is_effectful=is_store,
            eval_fn=eval_fn,
        )

    def default_function_abi_candidate_slots(self) -> List[AbiArgSlot]:
        return []

    def arg_name(self, loc: ArgLoc) -> str:
        if loc.offset is not None:
            if loc.offset >= 4 and loc.offset % 4 == 0:
                return f"arg{(loc.offset - 4) // 4}"
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
        """cdecl: all arguments are passed on the stack, pushed right to left.
        Offsets are relative to the stack pointer at function entry; slot 0
        (the return address) is skipped, so the first argument is at +4."""
        known_slots: List[AbiArgSlot] = []
        offset = 4
        if fn_sig.params_known:
            for i, param in enumerate(fn_sig.params):
                param_type = param.type.decay()
                size, align = param_type.get_parameter_size_align_bytes()
                size = (size + 3) & ~3
                # 32-bit x86 (cdecl/stdcall) passes every stack argument in
                # 4-byte slots regardless of the value's natural alignment: an
                # 8-byte `double` still occupies two consecutive slots starting
                # at a 4-byte boundary (a leading `double` is at [esp+4], not
                # [esp+8]). Cap the alignment at 4 while keeping the 8-byte
                # size, so `double`/`long long` arguments land at the right
                # offsets. (get_parameter_size_align_bytes returns (8, 8) for
                # scalars, which is the host/64-bit ABI's alignment.)
                align = min(align, 4)
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
        # For calls, unknown extra arguments are collected from the pending
        # stack argument stores instead (make_function_call requires
        # possible_slots to be registers).
        if not for_call and (not fn_sig.params_known or fn_sig.is_variadic):
            for i in range(8):
                candidate_slots.append(
                    AbiArgSlot(
                        ArgLoc(offset + 4 * i, len(known_slots) + i, None),
                        Type.any_reg(),
                    )
                )
        return Abi(arg_slots=known_slots, possible_slots=candidate_slots)

    def function_return(self, expr: Expression) -> Dict[Register, Expression]:
        # Return values are in eax (edx holds the high half of u64's) or, for
        # float/double returns, in x87 st(0). A call that returns a float
        # pushes it onto the FPU stack; the FPU prepass annotates the call with
        # exactly which virtual register that is (fpret), so every f0..f7 needs
        # an entry here even though only one is ever an output of a given call.
        result = {
            EAX: as_type(expr, Type.intptr(), silent=True, unify=False),
            EDX: fn_op("SECOND_REG", [expr], Type.reg32(likely_float=False)),
        }
        for i in range(8):
            result[Register(f"f{i}")] = Cast(
                expr, reinterpret=True, silent=True, type=Type.floatish()
            )
        return result
