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
SomeStruct: size 568, align 8
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
  0x40: array_arithmetic_1 (char [1 + 1])
  0x42: array_arithmetic_2 (char [2 - 1])
  0x43: array_arithmetic_3 (char [1 * 1])
  0x44: array_arithmetic_4 (char [1 << 1])
  0x46: array_arithmetic_5 (char [2 >> 1])
  0x47: array_arithmetic_6 (char [SECOND_ELEM])
  0x49: array_arithmetic_7 (char [16 + (~1)])
  0x57: array_arithmetic_8 (char [16 + (!0)])
  0x68: array_arithmetic_9 (char [16 + (-1)])
  0x77: array_arithmetic_10 (char [16 + (5 / 2)])
  0x89: array_arithmetic_11 (char [16 + ((-5) / 2)])
  0x97: array_arithmetic_12 (char [16 + ((-5) / (-2))])
  0xa9: array_arithmetic_13 (char [16 + (5 / (-2))])
  0xb7: array_arithmetic_14 (char [16 + (5 % 2)])
  0xc8: array_arithmetic_15 (char [16 + ((-5) % 2)])
  0xd7: array_arithmetic_16 (char [16 + ((-5) % (-2))])
  0xe8: array_arithmetic_17 (char [16 + (5 % (-2))])
  0xf7: array_arithmetic_18 (char [16 + (0 && (1 / 0))])
  0x107: array_arithmetic_19 (char [16 + (1 && 0)])
  0x117: array_arithmetic_20 (char [16 + (1 && 1)])
  0x128: array_arithmetic_21 (char [16 + (1 || (1 / 0))])
  0x139: array_arithmetic_22 (char [16 + (0 || 0)])
  0x149: array_arithmetic_23 (char [16 + (0 || 1)])
  0x15a: array_arithmetic_24 (char [16 + (1 >= 0)])
  0x16b: array_arithmetic_25 (char [16 + (1 > 0)])
  0x17c: array_arithmetic_26 (char [16 + (1 <= 0)])
  0x18c: array_arithmetic_27 (char [16 + (1 < 0)])
  0x19c: array_arithmetic_28 (char [16 + (1 == 0)])
  0x1ac: array_arithmetic_29 (char [16 + (1 != 0)])
  0x1bd: array_arithmetic_30 (char [16 + (5 ^ 3)])
  0x1d3: array_arithmetic_31 (char [16 + (5 & 3)])
  0x1e4: array_arithmetic_32 (char [16 + (5 | 3)])
  0x1fb: array_arithmetic_33 (char [16 + ((1) ? (2) : (3))])
  0x20d: array_arithmetic_34 (char [16 + ((0) ? (2) : (3))])
  0x220: array_arithmetic_35 (char [16 + ((2, 3))])
  0x233: end (char)

Enums:
FIRST_ELEM: 0
SECOND_ELEM: 2
THIRD_ELEM: 6
FOURTH_ELEM: 7
FIFTH_ELEM: 8
MORE_ENUM: 2
YET_MORE_ENUM: 2

