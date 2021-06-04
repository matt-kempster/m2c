#ifndef _MIPS2C_MACROS_H_
#define _MIPS2C_MACROS_H_

// Unknown types
#define MIPS2C_UNK      s32
#define MIPS2C_UNK8     s8
#define MIPS2C_UNK16    s16
#define MIPS2C_UNK32    s32
#define MIPS2C_UNK64    s64

// Bitwise (reinterpret) cast
#define MIPS2C_BITWISE(_type, _expr) (*(_type *)&(_expr))

// Unknown field access, like `(_type) _expr->unk_offset`
#define MIPS2C_FIELD(_expr, _type, _offset) (*(_type *) ((s8 *)(_expr) + (_offset)))

#endif
