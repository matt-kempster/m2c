.global test
test:
stwu r1, -0x30(r1)

lfs f2, myfloat@sda21(r2)

fneg f1, f2
cmpwi r3, 7
beq .done

fctiwz f4, f1
fadds f1, f2, f2
stfd f4, 0x28(r1)
lwz r0, 0x2c(r1)
stw r0, myint@sda21(r2)
.done:

addi r1, r1, 0x30
blr
