.extern _g
.globl _test
.globl _load
.section .text,"ax",@progbits
.balign 4
_test:                            ! function: test
                                  ! frame size=0
          MOV.L       L239,R3     ! _g
          FLDI1       FR4
          FMOV.S      FR4,@R3
          RTS
          FMOV      FR4,FR0
_load:                            ! function: load
                                  ! frame size=0
          MOV.L       L239,R3     ! _g
          RTS
          FMOV.S      @R3,FR0
L239:                             
.long _g

