# Manual regression test for PPC, where the same registers are used for
# function arguments & return values. This test calls a function `foo`
# in a branch, and we expect that the deduced signature of `test` is
# still `void foo(s32)` even though it would be possible for for foo
# to return a float value from $f1 (i.e. `f32 foo(s32, f32)`)

glabel test
mflr r0
cmpwi r3, 0
bne .end
bl foo
.end:
mtlr r0
blr
