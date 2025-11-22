Variables:
int underaligned_var;
underaligned underaligned_var2;

Functions:

Structs:
A: size 0x10, align 16
  0x0: x (int)
ContainsA: size 0x20, align 16
  0x0: x (int)
  0x10: a (struct A)
B: size 0x10, align 16
  0x0: x (int)
C: size 0x4, align 4
  0x0: x (int)
D: size 0x4, align 4
  0x0: x (int)
E: size 0x4, align 1
  0x0: y (underaligned)
F: size 0x40, align 32
  0x0: x (int)
  0x20: y (int)
G: size 0x20, align 16
  0x0: x (int)
  0x10: y (int)

Enums:

