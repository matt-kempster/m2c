.set noat # allow use of $at
.set noreorder # don't insert nops after branches
.set gp=64 # allow use of 64bit registers
.macro glabel label, visibility=global
    .\visibility "\label"
    .type "\label", @function
    "\label":
        .ent "\label"
.endm

glabel
loc_whatever:
nop

test:
addiu $sp, $sp, -0x34
sw $s0, 0X20+16($sp)
beq $zero, $zero, .label2
loc_whatever:
lui $v0, (0x12345678 >> (8 + 0x4 * 2))
.label1: .label2: .label2: .label3: addiu $v0, 0xFFFF # subtract 1
b jumptarget_label
 li $two, 2 # fictive register names are (currently) fine
jlabel jumptarget_label
multu $v0, $v0, $two # multiply by two, ps2-style
la $v1, ("special !@#$%^chars")
addu $v0, $v0, $v1
lui $v1, %hi("more special")
lw $v1, %lo("more special")($v1)
addu $v0, $v0, $v1
addiu $sp, $sp, 0x34

func_other:
jr $ra
sw $zero, ($zero)

yet_another_func:
jr $ra
sw $zero, ($zero)

.rodata
glabel "special !@#$%^chars", global
.word 0x1234
