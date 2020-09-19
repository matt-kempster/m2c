Variables:
globali: int
globalf: float
extern_inner_struct_field: struct SubStruct
func_decl: void (void)
test: short (struct SomeStruct *arg, unsigned char should, union SomeUnion union_arg, ...)

Functions:
func_decl: void(void)
test: short(struct SomeStruct *, unsigned char, union SomeUnion, ...)

Structs:
SomeUnion: size 8, align 8
  0x0: double_innerfield (double) char_innerfield (char)
SomeBitfield: size 8, align 4
SubStruct: size 4, align 4
  0x0: x (int)
SomeStruct: size 104, align 8
  0x0: int_field (int)
  0x4: float_field (float)
  0x8: pointer_field (void *)
  0x10: data_field (union SomeUnion) data_field.double_innerfield (double) data_field.char_innerfield (char)
  0x18: enum_field (enum SomeEnum)
  0x1c: anon_enum_field (anon enum)
  0x20: anon_struct_field (anon struct) anon_struct_field.sub (int)
  0x24: anon_union_field1 (int) anon_union_field2 (float)
  0x28: inner_struct_field (struct SubStruct) inner_struct_field.x (int)
  0x30: long_long_field (long long)
  0x38: bitfield_field (struct SomeBitfield)
  0x40: array_arithmetic_1 (int [1 + 1])
  0x48: array_arithmetic_2 (int [2 - 1])
  0x4c: array_arithmetic_3 (int [1 * 1])
  0x50: array_arithmetic_4 (int [1 << 1])
  0x58: array_arithmetic_5 (int [2 >> 1])
  0x5c: array_arithmetic_6 (int [SECOND_ELEM])

Enums:
FIRST_ELEM: 0
SECOND_ELEM: 2
THIRD_ELEM: 6
FOURTH_ELEM: 7
FIFTH_ELEM: 8
MORE_ENUM: 2
YET_MORE_ENUM: 2

