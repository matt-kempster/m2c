# Float memory, register, width, and pop-count arms.
test:
    FLD dword ptr [f32_a]
    FCOM dword ptr [f32_b]
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    FSTP ST(0)
    RET

fcom_q:
    FLD qword ptr [f64_a]
    FCOM qword ptr [f64_b]
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    FSTP ST(0)
    RET

fcom_reg:
    FLD dword ptr [f32_a]
    FLD dword ptr [f32_b]
    FCOM ST(1)
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    FSTP ST(0)
    FSTP ST(0)
    RET

fcomp_dword:
    FLD dword ptr [f32_a]
    FCOMP dword ptr [f32_b]
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    RET

fucom_reg:
    FLD dword ptr [f32_a]
    FLD dword ptr [f32_b]
    FUCOM ST(1)
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    FSTP ST(0)
    FSTP ST(0)
    RET

fucomp_q:
    FLD qword ptr [f64_a]
    FUCOMP qword ptr [f64_b]
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    RET

fcompp_case:
    FLD dword ptr [f32_a]
    FLD dword ptr [f32_b]
    FCOMPP
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    RET

fucompp_case:
    FLD dword ptr [f32_a]
    FLD dword ptr [f32_b]
    FUCOMPP
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    RET

ftst_case:
    FLD dword ptr [f32_a]
    FTST
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    FSTP ST(0)
    RET

# Integer word/dword and non-pop/pop arms.
ficom_word:
    FLD dword ptr [f32_a]
    FICOM word ptr [i16_a]
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    FSTP ST(0)
    RET

ficom_dword:
    FLD dword ptr [f32_a]
    FICOM dword ptr [i32_a]
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    FSTP ST(0)
    RET

ficomp_word:
    FLD dword ptr [f32_a]
    FICOMP word ptr [i16_a]
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    RET

ficomp_dword:
    FLD dword ptr [f32_a]
    FICOMP dword ptr [i32_a]
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    RET

# The last comparison before one status read wins.
serial_last_wins:
    FLD dword ptr [f32_a]
    FCOM dword ptr [f32_b]
    FCOM dword ptr [f32_c]
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    FSTP ST(0)
    RET

# The reaching compare may live in a predecessor block.
cross_block:
    FLD dword ptr [f32_a]
    FCOMP dword ptr [f32_b]
    JMP .Lcross_status
.Lcross_status:
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    RET

# Two reaching status definitions must not be folded through their phi.
phi_status:
    TEST ECX, ECX
    JZ .Lphi_right
    FLD dword ptr [f32_a]
    FCOMP dword ptr [f32_b]
    JMP .Lphi_join
.Lphi_right:
    FLD dword ptr [f32_a]
    FCOMP dword ptr [f32_c]
.Lphi_join:
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    RET

# A second consumer of the status copy retains a readable M2C_FCMP fallback.
extra_status_use:
    FLD dword ptr [f32_a]
    FCOMP dword ptr [f32_b]
    FNSTSW AX
    MOV EDX, EAX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    ADD EAX, EDX
    RET

# Negative/fallback cases retired from the unit-only matcher tests.
requires_compare:
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    RET

clobber_eax:
    FLD dword ptr [f32_a]
    FCOMP dword ptr [f32_b]
    FNSTSW AX
    MOV EAX, 0
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    RET

clobber_fsw:
    FLD dword ptr [f32_a]
    FCOMP dword ptr [f32_b]
    CALL _clobber_fsw
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    RET

unsupported_mask:
    FLD dword ptr [f32_a]
    FCOMP dword ptr [f32_b]
    FNSTSW AX
    TEST AH, 0x80
    SETZ AL
    MOVZX EAX, AL
    RET

# Interleaved unrelated instructions and a separate byte test remain intact.
interleaved:
    FLD dword ptr [f32_a]
    FCOMP dword ptr [f32_b]
    MOV ECX, 7
    FNSTSW AX
    ADD EDX, 1
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    RET

unrelated_byte_test:
    FLD dword ptr [f32_a]
    FCOMP dword ptr [f32_b]
    FNSTSW AX
    TEST CL, 0x41
    SETNZ DL
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    MOVZX EDX, DL
    ADD EAX, EDX
    RET

# Both memory operands are captured in order and before either is invalidated.
capture_order:
    FLD dword ptr [lhs]
    FCOMP dword ptr [rhs]
    MOV dword ptr [lhs], 0
    MOV dword ptr [rhs], 0
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    RET

# Raw st(i) forms in unreachable code pass through without crashing the rewrite.
unreachable_raw:
    JMP .Lraw_ret
    FCOM ST(0), ST(1)
    FNSTSW AX
.Lraw_ret:
    MOV EAX, 0
    RET
