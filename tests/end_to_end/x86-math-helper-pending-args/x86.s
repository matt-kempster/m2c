.section .text

# The first __allmul argument is read before inner() and must be captured there
# when the helper result is used.
test:
    PUSH dword ptr [_g]
    PUSH 1
    CALL _inner
    ADD ESP, 4
    PUSH 2
    PUSH 3
    PUSH EAX
    CALL __allmul
    RET

# The same argument remains lazy when the pure helper result is discarded: no
# dead read of g and no dead capture temporary may be emitted.
dead:
    PUSH dword ptr [_g]
    PUSH 1
    CALL _inner
    ADD ESP, 4
    PUSH 2
    PUSH 3
    PUSH EAX
    CALL __allmul
    MOV EAX, 0
    MOV EDX, 0
    RET
