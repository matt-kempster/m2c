Variables:
int underaligned_var;
underaligned underaligned_var2;

Functions:

Structs:
Overaligned1: size 0x10, align 16
  0x0: x (int)
ContainsOveraligned: size 0x20, align 16
  0x0: x (int)
  0x10: a (struct Overaligned1)
Overaligned2: size 0x10, align 16
  0x0: x (int)
UnderalignedNoop: size 0x4, align 4
  0x0: x (int)
UnderalignedMemberNoop: size 0x4, align 4
  0x0: x (int)
UnderalignedMemberViaTypedef: size 0x4, align 1
  0x0: y (underaligned)
MultipleAttrs: size 0x40, align 32
  0x0: x (int)
  0x20: y (int)
Alignas: size 0x20, align 16
  0x0: x (int)
  0x10: y (int)
Packed: size 0x6, align 2
  0x0: x (char)
  0x1: y (int)
PackedMember: size 0x5, align 1
  0x0: x (char)
  0x1: y (int)
PragmaPack1: size 0x5, align 1
  0x0: a (char)
  0x1: nicetry (int)
PragmaPack2: size 0x6, align 2
  0x0: a (char)
  0x2: nicetry (int)
PragmaPackNone1: size 0x8, align 4
  0x0: a (char)
  0x4: b (int)
PragmaPackNone2: size 0x8, align 4
  0x0: a (char)
  0x4: b (int)
PackedWithOveralignedMember: size 0x20, align 16
  0x0: x (char)
  0x1: y (int)
  0x10: z (int)

Enums:

