/*
 * This header contains macros emitted by mips_to_c in "valid syntax" mode,
 * which can be enabled by passing `--valid-syntax` on the command line.
 *
 * In this mode, unhandled types and expressions are emitted as macros so
 * that the output is compilable without human intervention.
 */

#ifndef MIPS2C_MACROS_H
#define MIPS2C_MACROS_H

/* Unknown types */
typedef s32 MIPS2C_UNK;
typedef s8  MIPS2C_UNK8;
typedef s16 MIPS2C_UNK16;
typedef s32 MIPS2C_UNK32;
typedef s64 MIPS2C_UNK64;

/* Unknown field access, like `*(type_ptr) &expr->unk_offset` */
#define MIPS2C_FIELD(expr, type_ptr, offset) (*(type_ptr)((s8 *)(expr) + (offset)))

/* Bitwise (reinterpret) cast */
#define MIPS2C_BITWISE(type, expr) ((type)(expr))

/* Unaligned reads */
#define MIPS2C_LWL(expr) (expr)
#define MIPS2C_FIRST3BYTES(expr) (expr)
#define MIPS2C_UNALIGNED32(expr) (expr)

/* Unhandled instructions */
#define MIPS2C_ERROR(desc) (0)
#define MIPS2C_TRAP_IF(cond) (0)
#define MIPS2C_BREAK() (0)
#define MIPS2C_SYNC() (0)

/* Carry bit from partially-implemented instructions */
#define MIPS2C_CARRY 0

#endif
