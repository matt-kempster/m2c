Variables:
globali: int
globalf: float
func_decl: void (void)
test: short (struct SomeStruct *arg, unsigned char should, union SomeUnion union_arg, ...)

Functions:
func_decl: void(void)
test: short(struct SomeStruct *, unsigned char, SomeUnion, ...)

Structs:
SomeUnion: size 8, align 8
  0x0: double_innerfield (double) char_innerfield (char)
SomeBitfield: size 8, align 4
SomeStruct: size 72, align 8
  0x0: int_field (int)
  0x4: float_field (float)
  0x8: pointer_field (void *)
  0x10: data_field (SomeUnion) data_field.double_innerfield (double) data_field.char_innerfield (char)
  0x18: enum_field (enum SomeEnum)
  0x20: long_long_field (long long)
  0x28: bitfield_field (SomeBitfield)
  0x30: array_arithmetic_1 (int [1 + 1])
  0x38: array_arithmetic_2 (int [2 - 1])
  0x3c: array_arithmetic_3 (int [1 * 1])
  0x40: array_arithmetic_4 (int [1 << 1])
  0x48: array_arithmetic_5 (int [1 >> 1])

