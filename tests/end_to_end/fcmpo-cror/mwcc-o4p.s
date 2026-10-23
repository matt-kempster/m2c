.include "macros.inc"

.section .text  # 0x0 - 0x0

# `f1 <= f2`, using decomp-toolkit's symbolic `cror` operand spelling
# (`cror eq, lt, eq`), equivalent to the numeric spelling `cror 2, 0, 2`.
.global test
test:
fcmpo cr0, f1, f2
cror eq, lt, eq
bne .L_test_else
li r3, 1
blr
.L_test_else:
li r3, 0
blr

# `f1 >= f2`, using decomp-toolkit's symbolic `cror` operand spelling
# (`cror eq, gt, eq`), equivalent to the numeric spelling `cror 2, 1, 2`.
.global test2
test2:
fcmpo cr0, f1, f2
cror eq, gt, eq
bne .L_test2_else
li r3, 1
blr
.L_test2_else:
li r3, 0
blr

# `f1 <= f2`, using the numeric `cror` operand spelling (`cror 2, 0, 2`).
# This must keep producing the same result as the symbolic spelling above.
.global test3
test3:
fcmpo cr0, f1, f2
cror 2, 0, 2
bne .L_test3_else
li r3, 1
blr
.L_test3_else:
li r3, 0
blr

# `f1 >= f2`, using the numeric `cror` operand spelling (`cror 2, 1, 2`).
# This must keep producing the same result as the symbolic spelling above.
.global test4
test4:
fcmpo cr0, f1, f2
cror 2, 1, 2
bne .L_test4_else
li r3, 1
blr
.L_test4_else:
li r3, 0
blr
