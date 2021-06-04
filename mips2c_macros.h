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
#define MIPS2C_UNK   s32
#define MIPS2C_UNK8  s8
#define MIPS2C_UNK16 s16
#define MIPS2C_UNK32 s32
#define MIPS2C_UNK64 s64

/* Bitwise (reinterpret) cast */
#define MIPS2C_BITWISE(type, expr) ((type)(expr))

/* Unknown field access, like `(type) expr->unk_offset` */
#define MIPS2C_FIELD(expr, type, offset) (*(type *)((s8 *)(expr) + (offset)))

/* Unhandled instructions */
#define MIPS2C_ERROR(desc) (0)
#define MIPS2C_TRAP_IF(cond) (0)
#define MIPS2C_BREAK() (0)
#define MIPS2C_SYNC() (0)

#endif
