.set noat      # allow manual use of $at
.set noreorder # don't insert nops after branches

.text

glabel static_fn

/* 0000B4 004000B4 AFA40000 */   sw    $a0, ($sp)

glabel test
/* 0000B8 004000B8 3C030041 */  lui   $v1, %hi(extern_float)
/* 0000BC 004000BC 24630134 */  addiu $v1, $v1, %lo(extern_float)
/* 0000C0 004000C0 3C020041 */  lui   $v0, %hi(static_int)
/* 0000C4 004000C4 3C0143E4 */  lui   $at, 0x43e4
/* 0000C8 004000C8 44813000 */  mtc1  $at, $f6
/* 0000CC 004000CC C4640000 */  lwc1  $f4, ($v1)
/* 0000D0 004000D0 24420130 */  addiu $v0, $v0, %lo(static_int)
/* 0000D4 004000D4 8C4E0000 */  lw    $t6, ($v0)
/* 0000D8 004000D8 46062202 */  mul.s $f8, $f4, $f6
/* 0000DC 004000DC 27BDFFE8 */  addiu $sp, $sp, -0x18
/* 0000E0 004000E0 000E78C0 */  sll   $t7, $t6, 3
/* 0000E4 004000E4 01EE7823 */  subu  $t7, $t7, $t6
/* 0000E8 004000E8 000F78C0 */  sll   $t7, $t7, 3
/* 0000EC 004000EC 01EE7821 */  addu  $t7, $t7, $t6
/* 0000F0 004000F0 AFBF0014 */  sw    $ra, 0x14($sp)
/* 0000F4 004000F4 000F78C0 */  sll   $t7, $t7, 3
/* 0000F8 004000F8 3C040041 */  lui   $a0, %hi(static_A)
/* 0000FC 004000FC AC4F0000 */  sw    $t7, ($v0)
/* 000100 00400100 E4680000 */  swc1  $f8, ($v1)
/* 000104 00400104 0C10002C */  jal   static_fn
/* 000108 00400108 24840138 */   addiu $a0, $a0, %lo(static_A)
/* 00010C 0040010C 3C040041 */  lui   $a0, %hi(static_A_ptr)
/* 000110 00400110 0C10002C */  jal   extern_fn
/* 000114 00400114 8C84014C */   lw    $a0, %lo(static_A_ptr)($a0)
/* 000118 00400118 8FBF0014 */  lw    $ra, 0x14($sp)
/* 00011C 0040011C 3C020041 */  lui   $v0, %hi(static_int)
/* 000120 00400120 8C420130 */  lw    $v0, %lo(static_int)($v0)
/* 000124 00400124 03E00008 */  jr    $ra
/* 000128 00400128 27BD0018 */   addiu $sp, $sp, 0x18

/* 00012C 0040012C 00000000 */  nop

.rodata
glabel static_A
.byte 0x01, 0x02, 0x03, 0x04, 0x05

.data
glabel static_A_ptr
.word static_A

.bss
glabel static_int
.word 0
