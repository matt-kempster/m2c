# like switch/irix-g.s but with manual tweaks
.set noat      # allow manual use of $at
.set noreorder # don't insert nops after branches

.rdata
glabel jtbl_400150
.word .L004000D8
.word .L004000E8
.word .L004000EC
.word .L004000F4
.word .L00400110
.word .L00400100
.word .L00400100

.text
glabel test
/* 0000B0 004000B0 248EFFFF */  addiu $t6, $a0, -1
/* 0000B4 004000B4 2DC10006 */  sltiu $at, $t6, 7
/* 0000B8 004000B8 10200015 */  beqz  $at, .L00400110
/* 0000BC 004000BC 00000000 */   nop
/* 0000C0 004000C0 000E7080 */  sll   $t6, $t6, 2
/* 0000C4 004000C4 3C010040 */  lui   $at, %hi(jtbl_400150)
/* 0000C8 004000C8 002E0821 */  addu  $at, $at, $t6
/* 0000CC 004000CC 8C2E0150 */  lw    $t6, %lo(jtbl_400150)($at)
b .next
 nop
.next:
/* 0000D0 004000D0 01C00008 */  jr    $t6
/* 0000D4 004000D4 00000000 */   nop
.L004000D8:
/* 0000D8 004000D8 00840019 */  multu $a0, $a0
/* 0000DC 004000DC 00001012 */  mflo  $v0
/* 0000E0 004000E0 03E00008 */  jr    $ra
/* 0000E4 004000E4 00000000 */   nop
.L004000E8:
/* 0000E8 004000E8 2484FFFF */  addiu $a0, $a0, -1
.L004000EC:
/* 0000EC 004000EC 03E00008 */  jr    $ra
/* 0000F0 004000F0 00041040 */   sll   $v0, $a0, 1
.L004000F4:
/* 0000F4 004000F4 24840001 */  addiu $a0, $a0, 1
/* 0000F8 004000F8 1000000C */  b     .L0040012C
/* 0000FC 004000FC 00000000 */   nop
.L00400100:
/* 000100 00400100 00047840 */  sll   $t7, $a0, 1
/* 000104 00400104 01E02025 */  move  $a0, $t7
/* 000108 00400108 10000008 */  b     .L0040012C
/* 00010C 0040010C 00000000 */   nop
.L00400110:
/* 000110 00400110 04810003 */  bgez  $a0, .L00400120
/* 000114 00400114 0004C043 */   sra   $t8, $a0, 1
/* 000118 00400118 24810001 */  addiu $at, $a0, 1
/* 00011C 0040011C 0001C043 */  sra   $t8, $at, 1
.L00400120:
/* 000120 00400120 03002025 */  move  $a0, $t8
/* 000124 00400124 10000001 */  b     .L0040012C
/* 000128 00400128 00000000 */   nop
.L0040012C:
/* 00012C 0040012C 3C010041 */  lui   $at, %hi(D_410170)
/* 000130 00400130 AC240170 */  sw    $a0, %lo(D_410170)($at)
/* 000134 00400134 03E00008 */  jr    $ra
/* 000138 00400138 24020002 */   addiu $v0, $zero, 2

/* 00013C 0040013C 03E00008 */  jr    $ra
/* 000140 00400140 00000000 */   nop

/* 000144 00400144 03E00008 */  jr    $ra
/* 000148 00400148 00000000 */   nop

/* 00014C 0040014C 00000000 */  nop
