test:
	push {r4, r5, r6, lr}
	vstmdb sp!, {d8, d9}
	vpush {d10, d11}
	vpush {s24, s25, s26, s27}
	vpush {q7}
	vldr s24, [r0]
	vstr s24, [r1]
	vldr d12, [r0]
	vstr d12, [r1]
	vldr q6, [r0]
	vstr q6, [r1]
	vpop {q7}
	vpop {s24, s25, s26, s27}
	vpop {d10, d11}
	vldm sp!, {d8, d9}
	pop {r4, r5, r6, pc}
