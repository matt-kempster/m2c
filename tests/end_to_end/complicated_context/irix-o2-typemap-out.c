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
  0: double_innerfield (double) char_innerfield (char)
SomeBitfield: size 8, align 4
SomeStruct: size 72, align 8
  0: int_field (int)
  4: float_field (float)
  8: pointer_field (void *)
  16: data_field (SomeUnion) data_field.double_innerfield (double) data_field.char_innerfield (char)
  24: enum_field (enum SomeEnum)
  32: long_long_field (long long)
  40: bitfield_field (SomeBitfield)
  48: array_arithmetic_1 (int [1 + 1])
  56: array_arithmetic_2 (int [2 - 1])
  60: array_arithmetic_3 (int [1 * 1])
  64: array_arithmetic_4 (int [1 << 1])
  72: array_arithmetic_5 (int [1 >> 1])

