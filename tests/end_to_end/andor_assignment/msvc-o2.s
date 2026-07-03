.section .text
foo:
/* 00000000 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000004 0004  40 */	inc eax
/* 00000005 0005  C3 */	ret
/* 00000006 0006  90 */	nop
/* 00000007 0007  90 */	nop
/* 00000008 0008  90 */	nop
/* 00000009 0009  90 */	nop
/* 0000000A 000A  90 */	nop
/* 0000000B 000B  90 */	nop
/* 0000000C 000C  90 */	nop
/* 0000000D 000D  90 */	nop
/* 0000000E 000E  90 */	nop
/* 0000000F 000F  90 */	nop

test:
/* 00000010 0000  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000014 0004  53 */	push ebx
/* 00000015 0005  8B 5C 24 10 */	mov ebx, dword ptr [esp + 0x10]
/* 00000019 0009  55 */	push ebp
/* 0000001A 000A  8B 6C 24 18 */	mov ebp, dword ptr [esp + 0x18]
/* 0000001E 000E  56 */	push esi
/* 0000001F 000F  8B 74 24 10 */	mov esi, dword ptr [esp + 0x10]
/* 00000023 0013  57 */	push edi
/* 00000024 0014  8D 3C 06 */	lea edi, [esi + eax]
/* 00000027 0017  03 C3 */	add eax, ebx
/* 00000029 0019  85 FF */	test edi, edi
/* 0000002B 001B  75 28 */	jne .L00000055
/* 0000002D 001D  85 C0 */	test eax, eax
/* 0000002F 001F  75 24 */	jne .L00000055
/* 00000031 0021  50 */	push eax
/* 00000032 0022  E8 00 00 00 00 */	call foo
/* 00000037 0027  83 C4 04 */	add esp, 4
/* 0000003A 002A  85 C0 */	test eax, eax
/* 0000003C 002C  75 17 */	jne .L00000055
/* 0000003E 002E  85 ED */	test ebp, ebp
/* 00000040 0030  BF 02 00 00 00 */	mov edi, 2
/* 00000045 0035  75 0E */	jne .L00000055
/* 00000047 0037  33 C9 */	xor ecx, ecx
/* 00000049 0039  85 F6 */	test esi, esi
/* 0000004B 003B  0F 95 C1 */	setne cl
/* 0000004E 003E  83 C1 FE */	add ecx, -2
/* 00000051 0041  8B F1 */	mov esi, ecx
/* 00000053 0043  EB 05 */	jmp .L0000005A
.L00000055:
/* 00000055 0045  BE 01 00 00 00 */	mov esi, 1
.L0000005A:
/* 0000005A 004A  03 F3 */	add esi, ebx
/* 0000005C 004C  85 FF */	test edi, edi
/* 0000005E 004E  74 5B */	je .L000000BB
/* 00000060 0050  85 C0 */	test eax, eax
/* 00000062 0052  74 24 */	je .L00000088
/* 00000064 0054  03 F8 */	add edi, eax
/* 00000066 0056  57 */	push edi
/* 00000067 0057  E8 00 00 00 00 */	call foo
/* 0000006C 005C  83 C4 04 */	add esp, 4
/* 0000006F 005F  85 C0 */	test eax, eax
/* 00000071 0061  74 15 */	je .L00000088
/* 00000073 0063  85 ED */	test ebp, ebp
/* 00000075 0065  74 11 */	je .L00000088
/* 00000077 0067  83 FE 05 */	cmp esi, 5
/* 0000007A 006A  7D 09 */	jge .L00000085
.L0000007C:
/* 0000007C 006C  8D 74 36 02 */	lea esi, [esi + esi + 2]
/* 00000080 0070  83 FE 05 */	cmp esi, 5
/* 00000083 0073  7C F7 */	jl .L0000007C
.L00000085:
/* 00000085 0075  83 C6 05 */	add esi, 5
.L00000088:
/* 00000088 0078  85 FF */	test edi, edi
/* 0000008A 007A  74 2F */	je .L000000BB
/* 0000008C 007C  85 C0 */	test eax, eax
/* 0000008E 007E  74 2B */	je .L000000BB
/* 00000090 0080  03 C7 */	add eax, edi
/* 00000092 0082  50 */	push eax
/* 00000093 0083  E8 00 00 00 00 */	call foo
/* 00000098 0088  83 C4 04 */	add esp, 4
/* 0000009B 008B  85 C0 */	test eax, eax
/* 0000009D 008D  74 1C */	je .L000000BB
/* 0000009F 008F  85 ED */	test ebp, ebp
/* 000000A1 0091  74 18 */	je .L000000BB
/* 000000A3 0093  83 FE 05 */	cmp esi, 5
/* 000000A6 0096  7D 09 */	jge .L000000B1
.L000000A8:
/* 000000A8 0098  8D 74 36 02 */	lea esi, [esi + esi + 2]
/* 000000AC 009C  83 FE 05 */	cmp esi, 5
/* 000000AF 009F  7C F7 */	jl .L000000A8
.L000000B1:
/* 000000B1 00A1  83 C6 05 */	add esi, 5
/* 000000B4 00A4  5F */	pop edi
/* 000000B5 00A5  8B C6 */	mov eax, esi
/* 000000B7 00A7  5E */	pop esi
/* 000000B8 00A8  5D */	pop ebp
/* 000000B9 00A9  5B */	pop ebx
/* 000000BA 00AA  C3 */	ret
.L000000BB:
/* 000000BB 00AB  83 C6 06 */	add esi, 6
/* 000000BE 00AE  5F */	pop edi
/* 000000BF 00AF  8B C6 */	mov eax, esi
/* 000000C1 00B1  5E */	pop esi
/* 000000C2 00B2  5D */	pop ebp
/* 000000C3 00B3  5B */	pop ebx
/* 000000C4 00B4  C3 */	ret
/* 000000C5 00B5  90 */	nop
/* 000000C6 00B6  90 */	nop
/* 000000C7 00B7  90 */	nop
/* 000000C8 00B8  90 */	nop
/* 000000C9 00B9  90 */	nop
/* 000000CA 00BA  90 */	nop
/* 000000CB 00BB  90 */	nop
/* 000000CC 00BC  90 */	nop
/* 000000CD 00BD  90 */	nop
/* 000000CE 00BE  90 */	nop
/* 000000CF 00BF  90 */	nop

