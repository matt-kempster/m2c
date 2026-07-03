.section .text
test_s8:
/* 00000000 0000  0F BE 44 24 04 */	movsx eax, byte ptr [esp + 4]
/* 00000005 0005  99 */	cdq
/* 00000006 0006  2B C2 */	sub eax, edx
/* 00000008 0008  D1 F8 */	sar eax, 1
/* 0000000A 000A  50 */	push eax
/* 0000000B 000B  E8 00 00 00 00 */	call _foo
/* 00000010 0010  0F BE 4C 24 08 */	movsx ecx, byte ptr [esp + 8]
/* 00000015 0015  B8 56 55 55 55 */	mov eax, 0x55555556
/* 0000001A 001A  F7 E9 */	imul ecx
/* 0000001C 001C  8B C2 */	mov eax, edx
/* 0000001E 001E  C1 E8 1F */	shr eax, 0x1f
/* 00000021 0021  03 D0 */	add edx, eax
/* 00000023 0023  52 */	push edx
/* 00000024 0024  E8 00 00 00 00 */	call _foo
/* 00000029 0029  0F BE 4C 24 0C */	movsx ecx, byte ptr [esp + 0xc]
/* 0000002E 002E  B8 67 66 66 66 */	mov eax, 0x66666667
/* 00000033 0033  F7 E9 */	imul ecx
/* 00000035 0035  D1 FA */	sar edx, 1
/* 00000037 0037  8B CA */	mov ecx, edx
/* 00000039 0039  C1 E9 1F */	shr ecx, 0x1f
/* 0000003C 003C  03 D1 */	add edx, ecx
/* 0000003E 003E  52 */	push edx
/* 0000003F 003F  E8 00 00 00 00 */	call _foo
/* 00000044 0044  0F BE 4C 24 10 */	movsx ecx, byte ptr [esp + 0x10]
/* 00000049 0049  B8 93 24 49 92 */	mov eax, 0x92492493
/* 0000004E 004E  F7 E9 */	imul ecx
/* 00000050 0050  03 D1 */	add edx, ecx
/* 00000052 0052  C1 FA 02 */	sar edx, 2
/* 00000055 0055  8B C2 */	mov eax, edx
/* 00000057 0057  C1 E8 1F */	shr eax, 0x1f
/* 0000005A 005A  03 D0 */	add edx, eax
/* 0000005C 005C  52 */	push edx
/* 0000005D 005D  E8 00 00 00 00 */	call _foo
/* 00000062 0062  0F BE 4C 24 14 */	movsx ecx, byte ptr [esp + 0x14]
/* 00000067 0067  B8 67 66 66 66 */	mov eax, 0x66666667
/* 0000006C 006C  F7 E9 */	imul ecx
/* 0000006E 006E  C1 FA 02 */	sar edx, 2
/* 00000071 0071  8B CA */	mov ecx, edx
/* 00000073 0073  C1 E9 1F */	shr ecx, 0x1f
/* 00000076 0076  03 D1 */	add edx, ecx
/* 00000078 0078  52 */	push edx
/* 00000079 0079  E8 00 00 00 00 */	call _foo
/* 0000007E 007E  0F BE 4C 24 18 */	movsx ecx, byte ptr [esp + 0x18]
/* 00000083 0083  B8 1F 85 EB 51 */	mov eax, 0x51eb851f
/* 00000088 0088  F7 E9 */	imul ecx
/* 0000008A 008A  C1 FA 05 */	sar edx, 5
/* 0000008D 008D  8B C2 */	mov eax, edx
/* 0000008F 008F  C1 E8 1F */	shr eax, 0x1f
/* 00000092 0092  03 D0 */	add edx, eax
/* 00000094 0094  52 */	push edx
/* 00000095 0095  E8 00 00 00 00 */	call _foo
/* 0000009A 009A  0F BE 4C 24 1C */	movsx ecx, byte ptr [esp + 0x1c]
/* 0000009F 009F  B8 81 80 80 80 */	mov eax, 0x80808081
/* 000000A4 00A4  F7 E9 */	imul ecx
/* 000000A6 00A6  03 D1 */	add edx, ecx
/* 000000A8 00A8  C1 FA 07 */	sar edx, 7
/* 000000AB 00AB  8B CA */	mov ecx, edx
/* 000000AD 00AD  C1 E9 1F */	shr ecx, 0x1f
/* 000000B0 00B0  03 D1 */	add edx, ecx
/* 000000B2 00B2  52 */	push edx
/* 000000B3 00B3  E8 00 00 00 00 */	call _foo
/* 000000B8 00B8  0F BE 54 24 20 */	movsx edx, byte ptr [esp + 0x20]
/* 000000BD 00BD  81 E2 01 00 00 80 */	and edx, 0x80000001
/* 000000C3 00C3  79 05 */	jns .L000000CA
/* 000000C5 00C5  4A */	dec edx
/* 000000C6 00C6  83 CA FE */	or edx, 0xfffffffe
/* 000000C9 00C9  42 */	inc edx
.L000000CA:
/* 000000CA 00CA  52 */	push edx
/* 000000CB 00CB  E8 00 00 00 00 */	call _foo
/* 000000D0 00D0  0F BE 44 24 24 */	movsx eax, byte ptr [esp + 0x24]
/* 000000D5 00D5  99 */	cdq
/* 000000D6 00D6  B9 03 00 00 00 */	mov ecx, 3
/* 000000DB 00DB  F7 F9 */	idiv ecx
/* 000000DD 00DD  52 */	push edx
/* 000000DE 00DE  E8 00 00 00 00 */	call _foo
/* 000000E3 00E3  0F BE 44 24 28 */	movsx eax, byte ptr [esp + 0x28]
/* 000000E8 00E8  99 */	cdq
/* 000000E9 00E9  B9 05 00 00 00 */	mov ecx, 5
/* 000000EE 00EE  F7 F9 */	idiv ecx
/* 000000F0 00F0  52 */	push edx
/* 000000F1 00F1  E8 00 00 00 00 */	call _foo
/* 000000F6 00F6  0F BE 44 24 2C */	movsx eax, byte ptr [esp + 0x2c]
/* 000000FB 00FB  99 */	cdq
/* 000000FC 00FC  B9 07 00 00 00 */	mov ecx, 7
/* 00000101 0101  F7 F9 */	idiv ecx
/* 00000103 0103  52 */	push edx
/* 00000104 0104  E8 00 00 00 00 */	call _foo
/* 00000109 0109  0F BE 44 24 30 */	movsx eax, byte ptr [esp + 0x30]
/* 0000010E 010E  99 */	cdq
/* 0000010F 010F  B9 0A 00 00 00 */	mov ecx, 0xa
/* 00000114 0114  F7 F9 */	idiv ecx
/* 00000116 0116  52 */	push edx
/* 00000117 0117  E8 00 00 00 00 */	call _foo
/* 0000011C 011C  0F BE 44 24 34 */	movsx eax, byte ptr [esp + 0x34]
/* 00000121 0121  99 */	cdq
/* 00000122 0122  B9 64 00 00 00 */	mov ecx, 0x64
/* 00000127 0127  F7 F9 */	idiv ecx
/* 00000129 0129  52 */	push edx
/* 0000012A 012A  E8 00 00 00 00 */	call _foo
/* 0000012F 012F  0F BE 44 24 38 */	movsx eax, byte ptr [esp + 0x38]
/* 00000134 0134  99 */	cdq
/* 00000135 0135  B9 FF 00 00 00 */	mov ecx, 0xff
/* 0000013A 013A  F7 F9 */	idiv ecx
/* 0000013C 013C  52 */	push edx
/* 0000013D 013D  E8 00 00 00 00 */	call _foo
/* 00000142 0142  83 C4 38 */	add esp, 0x38
/* 00000145 0145  C3 */	ret
/* 00000146 0146  90 */	nop
/* 00000147 0147  90 */	nop
/* 00000148 0148  90 */	nop
/* 00000149 0149  90 */	nop
/* 0000014A 014A  90 */	nop
/* 0000014B 014B  90 */	nop
/* 0000014C 014C  90 */	nop
/* 0000014D 014D  90 */	nop
/* 0000014E 014E  90 */	nop
/* 0000014F 014F  90 */	nop

test_s16:
/* 00000150 0000  0F BF 44 24 04 */	movsx eax, word ptr [esp + 4]
/* 00000155 0005  99 */	cdq
/* 00000156 0006  2B C2 */	sub eax, edx
/* 00000158 0008  D1 F8 */	sar eax, 1
/* 0000015A 000A  50 */	push eax
/* 0000015B 000B  E8 00 00 00 00 */	call _foo
/* 00000160 0010  0F BF 4C 24 08 */	movsx ecx, word ptr [esp + 8]
/* 00000165 0015  B8 56 55 55 55 */	mov eax, 0x55555556
/* 0000016A 001A  F7 E9 */	imul ecx
/* 0000016C 001C  8B C2 */	mov eax, edx
/* 0000016E 001E  C1 E8 1F */	shr eax, 0x1f
/* 00000171 0021  03 D0 */	add edx, eax
/* 00000173 0023  52 */	push edx
/* 00000174 0024  E8 00 00 00 00 */	call _foo
/* 00000179 0029  0F BF 4C 24 0C */	movsx ecx, word ptr [esp + 0xc]
/* 0000017E 002E  B8 67 66 66 66 */	mov eax, 0x66666667
/* 00000183 0033  F7 E9 */	imul ecx
/* 00000185 0035  D1 FA */	sar edx, 1
/* 00000187 0037  8B CA */	mov ecx, edx
/* 00000189 0039  C1 E9 1F */	shr ecx, 0x1f
/* 0000018C 003C  03 D1 */	add edx, ecx
/* 0000018E 003E  52 */	push edx
/* 0000018F 003F  E8 00 00 00 00 */	call _foo
/* 00000194 0044  0F BF 4C 24 10 */	movsx ecx, word ptr [esp + 0x10]
/* 00000199 0049  B8 93 24 49 92 */	mov eax, 0x92492493
/* 0000019E 004E  F7 E9 */	imul ecx
/* 000001A0 0050  03 D1 */	add edx, ecx
/* 000001A2 0052  C1 FA 02 */	sar edx, 2
/* 000001A5 0055  8B C2 */	mov eax, edx
/* 000001A7 0057  C1 E8 1F */	shr eax, 0x1f
/* 000001AA 005A  03 D0 */	add edx, eax
/* 000001AC 005C  52 */	push edx
/* 000001AD 005D  E8 00 00 00 00 */	call _foo
/* 000001B2 0062  0F BF 4C 24 14 */	movsx ecx, word ptr [esp + 0x14]
/* 000001B7 0067  B8 67 66 66 66 */	mov eax, 0x66666667
/* 000001BC 006C  F7 E9 */	imul ecx
/* 000001BE 006E  C1 FA 02 */	sar edx, 2
/* 000001C1 0071  8B CA */	mov ecx, edx
/* 000001C3 0073  C1 E9 1F */	shr ecx, 0x1f
/* 000001C6 0076  03 D1 */	add edx, ecx
/* 000001C8 0078  52 */	push edx
/* 000001C9 0079  E8 00 00 00 00 */	call _foo
/* 000001CE 007E  0F BF 4C 24 18 */	movsx ecx, word ptr [esp + 0x18]
/* 000001D3 0083  B8 1F 85 EB 51 */	mov eax, 0x51eb851f
/* 000001D8 0088  F7 E9 */	imul ecx
/* 000001DA 008A  C1 FA 05 */	sar edx, 5
/* 000001DD 008D  8B C2 */	mov eax, edx
/* 000001DF 008F  C1 E8 1F */	shr eax, 0x1f
/* 000001E2 0092  03 D0 */	add edx, eax
/* 000001E4 0094  52 */	push edx
/* 000001E5 0095  E8 00 00 00 00 */	call _foo
/* 000001EA 009A  0F BF 4C 24 1C */	movsx ecx, word ptr [esp + 0x1c]
/* 000001EF 009F  B8 81 80 80 80 */	mov eax, 0x80808081
/* 000001F4 00A4  F7 E9 */	imul ecx
/* 000001F6 00A6  03 D1 */	add edx, ecx
/* 000001F8 00A8  C1 FA 07 */	sar edx, 7
/* 000001FB 00AB  8B CA */	mov ecx, edx
/* 000001FD 00AD  C1 E9 1F */	shr ecx, 0x1f
/* 00000200 00B0  03 D1 */	add edx, ecx
/* 00000202 00B2  52 */	push edx
/* 00000203 00B3  E8 00 00 00 00 */	call _foo
/* 00000208 00B8  0F BF 4C 24 20 */	movsx ecx, word ptr [esp + 0x20]
/* 0000020D 00BD  B8 B7 60 0B B6 */	mov eax, 0xb60b60b7
/* 00000212 00C2  F7 E9 */	imul ecx
/* 00000214 00C4  03 D1 */	add edx, ecx
/* 00000216 00C6  C1 FA 08 */	sar edx, 8
/* 00000219 00C9  8B C2 */	mov eax, edx
/* 0000021B 00CB  C1 E8 1F */	shr eax, 0x1f
/* 0000021E 00CE  03 D0 */	add edx, eax
/* 00000220 00D0  52 */	push edx
/* 00000221 00D1  E8 00 00 00 00 */	call _foo
/* 00000226 00D6  0F BF 4C 24 24 */	movsx ecx, word ptr [esp + 0x24]
/* 0000022B 00DB  B8 03 00 01 80 */	mov eax, 0x80010003
/* 00000230 00E0  F7 E9 */	imul ecx
/* 00000232 00E2  03 D1 */	add edx, ecx
/* 00000234 00E4  C1 FA 0F */	sar edx, 0xf
/* 00000237 00E7  8B CA */	mov ecx, edx
/* 00000239 00E9  C1 E9 1F */	shr ecx, 0x1f
/* 0000023C 00EC  03 D1 */	add edx, ecx
/* 0000023E 00EE  52 */	push edx
/* 0000023F 00EF  E8 00 00 00 00 */	call _foo
/* 00000244 00F4  0F BF 54 24 28 */	movsx edx, word ptr [esp + 0x28]
/* 00000249 00F9  81 E2 01 00 00 80 */	and edx, 0x80000001
/* 0000024F 00FF  79 05 */	jns .L00000256
/* 00000251 0101  4A */	dec edx
/* 00000252 0102  83 CA FE */	or edx, 0xfffffffe
/* 00000255 0105  42 */	inc edx
.L00000256:
/* 00000256 0106  52 */	push edx
/* 00000257 0107  E8 00 00 00 00 */	call _foo
/* 0000025C 010C  0F BF 44 24 2C */	movsx eax, word ptr [esp + 0x2c]
/* 00000261 0111  99 */	cdq
/* 00000262 0112  B9 03 00 00 00 */	mov ecx, 3
/* 00000267 0117  F7 F9 */	idiv ecx
/* 00000269 0119  52 */	push edx
/* 0000026A 011A  E8 00 00 00 00 */	call _foo
/* 0000026F 011F  0F BF 44 24 30 */	movsx eax, word ptr [esp + 0x30]
/* 00000274 0124  99 */	cdq
/* 00000275 0125  B9 05 00 00 00 */	mov ecx, 5
/* 0000027A 012A  F7 F9 */	idiv ecx
/* 0000027C 012C  52 */	push edx
/* 0000027D 012D  E8 00 00 00 00 */	call _foo
/* 00000282 0132  0F BF 44 24 34 */	movsx eax, word ptr [esp + 0x34]
/* 00000287 0137  99 */	cdq
/* 00000288 0138  B9 07 00 00 00 */	mov ecx, 7
/* 0000028D 013D  F7 F9 */	idiv ecx
/* 0000028F 013F  52 */	push edx
/* 00000290 0140  E8 00 00 00 00 */	call _foo
/* 00000295 0145  0F BF 44 24 38 */	movsx eax, word ptr [esp + 0x38]
/* 0000029A 014A  99 */	cdq
/* 0000029B 014B  B9 0A 00 00 00 */	mov ecx, 0xa
/* 000002A0 0150  F7 F9 */	idiv ecx
/* 000002A2 0152  52 */	push edx
/* 000002A3 0153  E8 00 00 00 00 */	call _foo
/* 000002A8 0158  0F BF 44 24 3C */	movsx eax, word ptr [esp + 0x3c]
/* 000002AD 015D  99 */	cdq
/* 000002AE 015E  B9 64 00 00 00 */	mov ecx, 0x64
/* 000002B3 0163  F7 F9 */	idiv ecx
/* 000002B5 0165  52 */	push edx
/* 000002B6 0166  E8 00 00 00 00 */	call _foo
/* 000002BB 016B  0F BF 44 24 40 */	movsx eax, word ptr [esp + 0x40]
/* 000002C0 0170  99 */	cdq
/* 000002C1 0171  B9 FF 00 00 00 */	mov ecx, 0xff
/* 000002C6 0176  F7 F9 */	idiv ecx
/* 000002C8 0178  52 */	push edx
/* 000002C9 0179  E8 00 00 00 00 */	call _foo
/* 000002CE 017E  83 C4 40 */	add esp, 0x40
/* 000002D1 0181  0F BF 44 24 04 */	movsx eax, word ptr [esp + 4]
/* 000002D6 0186  99 */	cdq
/* 000002D7 0187  B9 68 01 00 00 */	mov ecx, 0x168
/* 000002DC 018C  F7 F9 */	idiv ecx
/* 000002DE 018E  52 */	push edx
/* 000002DF 018F  E8 00 00 00 00 */	call _foo
/* 000002E4 0194  0F BF 44 24 08 */	movsx eax, word ptr [esp + 8]
/* 000002E9 0199  99 */	cdq
/* 000002EA 019A  B9 FE FF 00 00 */	mov ecx, 0xfffe
/* 000002EF 019F  F7 F9 */	idiv ecx
/* 000002F1 01A1  52 */	push edx
/* 000002F2 01A2  E8 00 00 00 00 */	call _foo
/* 000002F7 01A7  83 C4 08 */	add esp, 8
/* 000002FA 01AA  C3 */	ret
/* 000002FB 01AB  90 */	nop
/* 000002FC 01AC  90 */	nop
/* 000002FD 01AD  90 */	nop
/* 000002FE 01AE  90 */	nop
/* 000002FF 01AF  90 */	nop

test_s32_div:
/* 00000300 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000304 0004  50 */	push eax
/* 00000305 0005  E8 00 00 00 00 */	call _foo
/* 0000030A 000A  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 0000030E 000E  99 */	cdq
/* 0000030F 000F  2B C2 */	sub eax, edx
/* 00000311 0011  D1 F8 */	sar eax, 1
/* 00000313 0013  50 */	push eax
/* 00000314 0014  E8 00 00 00 00 */	call _foo
/* 00000319 0019  8B 4C 24 0C */	mov ecx, dword ptr [esp + 0xc]
/* 0000031D 001D  B8 56 55 55 55 */	mov eax, 0x55555556
/* 00000322 0022  F7 E9 */	imul ecx
/* 00000324 0024  8B CA */	mov ecx, edx
/* 00000326 0026  C1 E9 1F */	shr ecx, 0x1f
/* 00000329 0029  03 D1 */	add edx, ecx
/* 0000032B 002B  52 */	push edx
/* 0000032C 002C  E8 00 00 00 00 */	call _foo
/* 00000331 0031  8B 44 24 10 */	mov eax, dword ptr [esp + 0x10]
/* 00000335 0035  99 */	cdq
/* 00000336 0036  83 E2 03 */	and edx, 3
/* 00000339 0039  03 C2 */	add eax, edx
/* 0000033B 003B  C1 F8 02 */	sar eax, 2
/* 0000033E 003E  50 */	push eax
/* 0000033F 003F  E8 00 00 00 00 */	call _foo
/* 00000344 0044  8B 4C 24 14 */	mov ecx, dword ptr [esp + 0x14]
/* 00000348 0048  B8 67 66 66 66 */	mov eax, 0x66666667
/* 0000034D 004D  F7 E9 */	imul ecx
/* 0000034F 004F  D1 FA */	sar edx, 1
/* 00000351 0051  8B C2 */	mov eax, edx
/* 00000353 0053  C1 E8 1F */	shr eax, 0x1f
/* 00000356 0056  03 D0 */	add edx, eax
/* 00000358 0058  52 */	push edx
/* 00000359 0059  E8 00 00 00 00 */	call _foo
/* 0000035E 005E  8B 4C 24 18 */	mov ecx, dword ptr [esp + 0x18]
/* 00000362 0062  B8 AB AA AA 2A */	mov eax, 0x2aaaaaab
/* 00000367 0067  F7 E9 */	imul ecx
/* 00000369 0069  8B CA */	mov ecx, edx
/* 0000036B 006B  C1 E9 1F */	shr ecx, 0x1f
/* 0000036E 006E  03 D1 */	add edx, ecx
/* 00000370 0070  52 */	push edx
/* 00000371 0071  E8 00 00 00 00 */	call _foo
/* 00000376 0076  8B 4C 24 1C */	mov ecx, dword ptr [esp + 0x1c]
/* 0000037A 007A  B8 93 24 49 92 */	mov eax, 0x92492493
/* 0000037F 007F  F7 E9 */	imul ecx
/* 00000381 0081  03 D1 */	add edx, ecx
/* 00000383 0083  C1 FA 02 */	sar edx, 2
/* 00000386 0086  8B C2 */	mov eax, edx
/* 00000388 0088  C1 E8 1F */	shr eax, 0x1f
/* 0000038B 008B  03 D0 */	add edx, eax
/* 0000038D 008D  52 */	push edx
/* 0000038E 008E  E8 00 00 00 00 */	call _foo
/* 00000393 0093  8B 44 24 20 */	mov eax, dword ptr [esp + 0x20]
/* 00000397 0097  99 */	cdq
/* 00000398 0098  83 E2 07 */	and edx, 7
/* 0000039B 009B  03 C2 */	add eax, edx
/* 0000039D 009D  C1 F8 03 */	sar eax, 3
/* 000003A0 00A0  50 */	push eax
/* 000003A1 00A1  E8 00 00 00 00 */	call _foo
/* 000003A6 00A6  8B 4C 24 24 */	mov ecx, dword ptr [esp + 0x24]
/* 000003AA 00AA  B8 39 8E E3 38 */	mov eax, 0x38e38e39
/* 000003AF 00AF  F7 E9 */	imul ecx
/* 000003B1 00B1  D1 FA */	sar edx, 1
/* 000003B3 00B3  8B CA */	mov ecx, edx
/* 000003B5 00B5  C1 E9 1F */	shr ecx, 0x1f
/* 000003B8 00B8  03 D1 */	add edx, ecx
/* 000003BA 00BA  52 */	push edx
/* 000003BB 00BB  E8 00 00 00 00 */	call _foo
/* 000003C0 00C0  8B 4C 24 28 */	mov ecx, dword ptr [esp + 0x28]
/* 000003C4 00C4  B8 67 66 66 66 */	mov eax, 0x66666667
/* 000003C9 00C9  F7 E9 */	imul ecx
/* 000003CB 00CB  C1 FA 02 */	sar edx, 2
/* 000003CE 00CE  8B C2 */	mov eax, edx
/* 000003D0 00D0  C1 E8 1F */	shr eax, 0x1f
/* 000003D3 00D3  03 D0 */	add edx, eax
/* 000003D5 00D5  52 */	push edx
/* 000003D6 00D6  E8 00 00 00 00 */	call _foo
/* 000003DB 00DB  8B 4C 24 2C */	mov ecx, dword ptr [esp + 0x2c]
/* 000003DF 00DF  B8 E9 A2 8B 2E */	mov eax, 0x2e8ba2e9
/* 000003E4 00E4  F7 E9 */	imul ecx
/* 000003E6 00E6  D1 FA */	sar edx, 1
/* 000003E8 00E8  8B CA */	mov ecx, edx
/* 000003EA 00EA  C1 E9 1F */	shr ecx, 0x1f
/* 000003ED 00ED  03 D1 */	add edx, ecx
/* 000003EF 00EF  52 */	push edx
/* 000003F0 00F0  E8 00 00 00 00 */	call _foo
/* 000003F5 00F5  8B 4C 24 30 */	mov ecx, dword ptr [esp + 0x30]
/* 000003F9 00F9  B8 AB AA AA 2A */	mov eax, 0x2aaaaaab
/* 000003FE 00FE  F7 E9 */	imul ecx
/* 00000400 0100  D1 FA */	sar edx, 1
/* 00000402 0102  8B C2 */	mov eax, edx
/* 00000404 0104  C1 E8 1F */	shr eax, 0x1f
/* 00000407 0107  03 D0 */	add edx, eax
/* 00000409 0109  52 */	push edx
/* 0000040A 010A  E8 00 00 00 00 */	call _foo
/* 0000040F 010F  8B 4C 24 34 */	mov ecx, dword ptr [esp + 0x34]
/* 00000413 0113  B8 4F EC C4 4E */	mov eax, 0x4ec4ec4f
/* 00000418 0118  F7 E9 */	imul ecx
/* 0000041A 011A  C1 FA 02 */	sar edx, 2
/* 0000041D 011D  8B CA */	mov ecx, edx
/* 0000041F 011F  C1 E9 1F */	shr ecx, 0x1f
/* 00000422 0122  03 D1 */	add edx, ecx
/* 00000424 0124  52 */	push edx
/* 00000425 0125  E8 00 00 00 00 */	call _foo
/* 0000042A 012A  8B 4C 24 38 */	mov ecx, dword ptr [esp + 0x38]
/* 0000042E 012E  B8 93 24 49 92 */	mov eax, 0x92492493
/* 00000433 0133  F7 E9 */	imul ecx
/* 00000435 0135  03 D1 */	add edx, ecx
/* 00000437 0137  C1 FA 03 */	sar edx, 3
/* 0000043A 013A  8B C2 */	mov eax, edx
/* 0000043C 013C  C1 E8 1F */	shr eax, 0x1f
/* 0000043F 013F  03 D0 */	add edx, eax
/* 00000441 0141  52 */	push edx
/* 00000442 0142  E8 00 00 00 00 */	call _foo
/* 00000447 0147  8B 4C 24 3C */	mov ecx, dword ptr [esp + 0x3c]
/* 0000044B 014B  B8 89 88 88 88 */	mov eax, 0x88888889
/* 00000450 0150  F7 E9 */	imul ecx
/* 00000452 0152  03 D1 */	add edx, ecx
/* 00000454 0154  C1 FA 03 */	sar edx, 3
/* 00000457 0157  8B CA */	mov ecx, edx
/* 00000459 0159  C1 E9 1F */	shr ecx, 0x1f
/* 0000045C 015C  03 D1 */	add edx, ecx
/* 0000045E 015E  52 */	push edx
/* 0000045F 015F  E8 00 00 00 00 */	call _foo
/* 00000464 0164  8B 44 24 40 */	mov eax, dword ptr [esp + 0x40]
/* 00000468 0168  99 */	cdq
/* 00000469 0169  83 E2 0F */	and edx, 0xf
/* 0000046C 016C  03 C2 */	add eax, edx
/* 0000046E 016E  C1 F8 04 */	sar eax, 4
/* 00000471 0171  50 */	push eax
/* 00000472 0172  E8 00 00 00 00 */	call _foo
/* 00000477 0177  83 C4 40 */	add esp, 0x40
/* 0000047A 017A  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 0000047E 017E  B8 79 78 78 78 */	mov eax, 0x78787879
/* 00000483 0183  F7 E9 */	imul ecx
/* 00000485 0185  C1 FA 03 */	sar edx, 3
/* 00000488 0188  8B C2 */	mov eax, edx
/* 0000048A 018A  C1 E8 1F */	shr eax, 0x1f
/* 0000048D 018D  03 D0 */	add edx, eax
/* 0000048F 018F  52 */	push edx
/* 00000490 0190  E8 00 00 00 00 */	call _foo
/* 00000495 0195  8B 4C 24 08 */	mov ecx, dword ptr [esp + 8]
/* 00000499 0199  B8 39 8E E3 38 */	mov eax, 0x38e38e39
/* 0000049E 019E  F7 E9 */	imul ecx
/* 000004A0 01A0  C1 FA 02 */	sar edx, 2
/* 000004A3 01A3  8B CA */	mov ecx, edx
/* 000004A5 01A5  C1 E9 1F */	shr ecx, 0x1f
/* 000004A8 01A8  03 D1 */	add edx, ecx
/* 000004AA 01AA  52 */	push edx
/* 000004AB 01AB  E8 00 00 00 00 */	call _foo
/* 000004B0 01B0  8B 4C 24 0C */	mov ecx, dword ptr [esp + 0xc]
/* 000004B4 01B4  B8 F3 1A CA 6B */	mov eax, 0x6bca1af3
/* 000004B9 01B9  F7 E9 */	imul ecx
/* 000004BB 01BB  C1 FA 03 */	sar edx, 3
/* 000004BE 01BE  8B C2 */	mov eax, edx
/* 000004C0 01C0  C1 E8 1F */	shr eax, 0x1f
/* 000004C3 01C3  03 D0 */	add edx, eax
/* 000004C5 01C5  52 */	push edx
/* 000004C6 01C6  E8 00 00 00 00 */	call _foo
/* 000004CB 01CB  8B 4C 24 10 */	mov ecx, dword ptr [esp + 0x10]
/* 000004CF 01CF  B8 67 66 66 66 */	mov eax, 0x66666667
/* 000004D4 01D4  F7 E9 */	imul ecx
/* 000004D6 01D6  C1 FA 03 */	sar edx, 3
/* 000004D9 01D9  8B CA */	mov ecx, edx
/* 000004DB 01DB  C1 E9 1F */	shr ecx, 0x1f
/* 000004DE 01DE  03 D1 */	add edx, ecx
/* 000004E0 01E0  52 */	push edx
/* 000004E1 01E1  E8 00 00 00 00 */	call _foo
/* 000004E6 01E6  8B 4C 24 14 */	mov ecx, dword ptr [esp + 0x14]
/* 000004EA 01EA  B8 31 0C C3 30 */	mov eax, 0x30c30c31
/* 000004EF 01EF  F7 E9 */	imul ecx
/* 000004F1 01F1  C1 FA 02 */	sar edx, 2
/* 000004F4 01F4  8B C2 */	mov eax, edx
/* 000004F6 01F6  C1 E8 1F */	shr eax, 0x1f
/* 000004F9 01F9  03 D0 */	add edx, eax
/* 000004FB 01FB  52 */	push edx
/* 000004FC 01FC  E8 00 00 00 00 */	call _foo
/* 00000501 0201  8B 4C 24 18 */	mov ecx, dword ptr [esp + 0x18]
/* 00000505 0205  B8 E9 A2 8B 2E */	mov eax, 0x2e8ba2e9
/* 0000050A 020A  F7 E9 */	imul ecx
/* 0000050C 020C  C1 FA 02 */	sar edx, 2
/* 0000050F 020F  8B CA */	mov ecx, edx
/* 00000511 0211  C1 E9 1F */	shr ecx, 0x1f
/* 00000514 0214  03 D1 */	add edx, ecx
/* 00000516 0216  52 */	push edx
/* 00000517 0217  E8 00 00 00 00 */	call _foo
/* 0000051C 021C  8B 4C 24 1C */	mov ecx, dword ptr [esp + 0x1c]
/* 00000520 0220  B8 C9 42 16 B2 */	mov eax, 0xb21642c9
/* 00000525 0225  F7 E9 */	imul ecx
/* 00000527 0227  03 D1 */	add edx, ecx
/* 00000529 0229  C1 FA 04 */	sar edx, 4
/* 0000052C 022C  8B C2 */	mov eax, edx
/* 0000052E 022E  C1 E8 1F */	shr eax, 0x1f
/* 00000531 0231  03 D0 */	add edx, eax
/* 00000533 0233  52 */	push edx
/* 00000534 0234  E8 00 00 00 00 */	call _foo
/* 00000539 0239  8B 4C 24 20 */	mov ecx, dword ptr [esp + 0x20]
/* 0000053D 023D  B8 AB AA AA 2A */	mov eax, 0x2aaaaaab
/* 00000542 0242  F7 E9 */	imul ecx
/* 00000544 0244  C1 FA 02 */	sar edx, 2
/* 00000547 0247  8B CA */	mov ecx, edx
/* 00000549 0249  C1 E9 1F */	shr ecx, 0x1f
/* 0000054C 024C  03 D1 */	add edx, ecx
/* 0000054E 024E  52 */	push edx
/* 0000054F 024F  E8 00 00 00 00 */	call _foo
/* 00000554 0254  8B 4C 24 24 */	mov ecx, dword ptr [esp + 0x24]
/* 00000558 0258  B8 1F 85 EB 51 */	mov eax, 0x51eb851f
/* 0000055D 025D  F7 E9 */	imul ecx
/* 0000055F 025F  C1 FA 03 */	sar edx, 3
/* 00000562 0262  8B C2 */	mov eax, edx
/* 00000564 0264  C1 E8 1F */	shr eax, 0x1f
/* 00000567 0267  03 D0 */	add edx, eax
/* 00000569 0269  52 */	push edx
/* 0000056A 026A  E8 00 00 00 00 */	call _foo
/* 0000056F 026F  8B 4C 24 28 */	mov ecx, dword ptr [esp + 0x28]
/* 00000573 0273  B8 4F EC C4 4E */	mov eax, 0x4ec4ec4f
/* 00000578 0278  F7 E9 */	imul ecx
/* 0000057A 027A  C1 FA 03 */	sar edx, 3
/* 0000057D 027D  8B CA */	mov ecx, edx
/* 0000057F 027F  C1 E9 1F */	shr ecx, 0x1f
/* 00000582 0282  03 D1 */	add edx, ecx
/* 00000584 0284  52 */	push edx
/* 00000585 0285  E8 00 00 00 00 */	call _foo
/* 0000058A 028A  8B 4C 24 2C */	mov ecx, dword ptr [esp + 0x2c]
/* 0000058E 028E  B8 F7 12 DA 4B */	mov eax, 0x4bda12f7
/* 00000593 0293  F7 E9 */	imul ecx
/* 00000595 0295  C1 FA 03 */	sar edx, 3
/* 00000598 0298  8B C2 */	mov eax, edx
/* 0000059A 029A  C1 E8 1F */	shr eax, 0x1f
/* 0000059D 029D  03 D0 */	add edx, eax
/* 0000059F 029F  52 */	push edx
/* 000005A0 02A0  E8 00 00 00 00 */	call _foo
/* 000005A5 02A5  8B 4C 24 30 */	mov ecx, dword ptr [esp + 0x30]
/* 000005A9 02A9  B8 93 24 49 92 */	mov eax, 0x92492493
/* 000005AE 02AE  F7 E9 */	imul ecx
/* 000005B0 02B0  03 D1 */	add edx, ecx
/* 000005B2 02B2  C1 FA 04 */	sar edx, 4
/* 000005B5 02B5  8B CA */	mov ecx, edx
/* 000005B7 02B7  C1 E9 1F */	shr ecx, 0x1f
/* 000005BA 02BA  03 D1 */	add edx, ecx
/* 000005BC 02BC  52 */	push edx
/* 000005BD 02BD  E8 00 00 00 00 */	call _foo
/* 000005C2 02C2  8B 4C 24 34 */	mov ecx, dword ptr [esp + 0x34]
/* 000005C6 02C6  B8 09 CB 3D 8D */	mov eax, 0x8d3dcb09
/* 000005CB 02CB  F7 E9 */	imul ecx
/* 000005CD 02CD  03 D1 */	add edx, ecx
/* 000005CF 02CF  C1 FA 04 */	sar edx, 4
/* 000005D2 02D2  8B C2 */	mov eax, edx
/* 000005D4 02D4  C1 E8 1F */	shr eax, 0x1f
/* 000005D7 02D7  03 D0 */	add edx, eax
/* 000005D9 02D9  52 */	push edx
/* 000005DA 02DA  E8 00 00 00 00 */	call _foo
/* 000005DF 02DF  8B 4C 24 38 */	mov ecx, dword ptr [esp + 0x38]
/* 000005E3 02E3  B8 89 88 88 88 */	mov eax, 0x88888889
/* 000005E8 02E8  F7 E9 */	imul ecx
/* 000005EA 02EA  03 D1 */	add edx, ecx
/* 000005EC 02EC  C1 FA 04 */	sar edx, 4
/* 000005EF 02EF  8B CA */	mov ecx, edx
/* 000005F1 02F1  C1 E9 1F */	shr ecx, 0x1f
/* 000005F4 02F4  03 D1 */	add edx, ecx
/* 000005F6 02F6  52 */	push edx
/* 000005F7 02F7  E8 00 00 00 00 */	call _foo
/* 000005FC 02FC  8B 4C 24 3C */	mov ecx, dword ptr [esp + 0x3c]
/* 00000600 0300  B8 43 08 21 84 */	mov eax, 0x84210843
/* 00000605 0305  F7 E9 */	imul ecx
/* 00000607 0307  03 D1 */	add edx, ecx
/* 00000609 0309  C1 FA 04 */	sar edx, 4
/* 0000060C 030C  8B C2 */	mov eax, edx
/* 0000060E 030E  C1 E8 1F */	shr eax, 0x1f
/* 00000611 0311  03 D0 */	add edx, eax
/* 00000613 0313  52 */	push edx
/* 00000614 0314  E8 00 00 00 00 */	call _foo
/* 00000619 0319  8B 44 24 40 */	mov eax, dword ptr [esp + 0x40]
/* 0000061D 031D  99 */	cdq
/* 0000061E 031E  83 E2 1F */	and edx, 0x1f
/* 00000621 0321  03 C2 */	add eax, edx
/* 00000623 0323  C1 F8 05 */	sar eax, 5
/* 00000626 0326  50 */	push eax
/* 00000627 0327  E8 00 00 00 00 */	call _foo
/* 0000062C 032C  83 C4 40 */	add esp, 0x40
/* 0000062F 032F  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000633 0333  B8 E1 83 0F 3E */	mov eax, 0x3e0f83e1
/* 00000638 0338  F7 E9 */	imul ecx
/* 0000063A 033A  C1 FA 03 */	sar edx, 3
/* 0000063D 033D  8B CA */	mov ecx, edx
/* 0000063F 033F  C1 E9 1F */	shr ecx, 0x1f
/* 00000642 0342  03 D1 */	add edx, ecx
/* 00000644 0344  52 */	push edx
/* 00000645 0345  E8 00 00 00 00 */	call _foo
/* 0000064A 034A  8B 4C 24 08 */	mov ecx, dword ptr [esp + 8]
/* 0000064E 034E  B8 1F 85 EB 51 */	mov eax, 0x51eb851f
/* 00000653 0353  F7 E9 */	imul ecx
/* 00000655 0355  C1 FA 05 */	sar edx, 5
/* 00000658 0358  8B C2 */	mov eax, edx
/* 0000065A 035A  C1 E8 1F */	shr eax, 0x1f
/* 0000065D 035D  03 D0 */	add edx, eax
/* 0000065F 035F  52 */	push edx
/* 00000660 0360  E8 00 00 00 00 */	call _foo
/* 00000665 0365  8B 4C 24 0C */	mov ecx, dword ptr [esp + 0xc]
/* 00000669 0369  B8 81 80 80 80 */	mov eax, 0x80808081
/* 0000066E 036E  F7 E9 */	imul ecx
/* 00000670 0370  03 D1 */	add edx, ecx
/* 00000672 0372  C1 FA 07 */	sar edx, 7
/* 00000675 0375  8B CA */	mov ecx, edx
/* 00000677 0377  C1 E9 1F */	shr ecx, 0x1f
/* 0000067A 037A  03 D1 */	add edx, ecx
/* 0000067C 037C  52 */	push edx
/* 0000067D 037D  E8 00 00 00 00 */	call _foo
/* 00000682 0382  8B 4C 24 10 */	mov ecx, dword ptr [esp + 0x10]
/* 00000686 0386  B8 B7 60 0B B6 */	mov eax, 0xb60b60b7
/* 0000068B 038B  F7 E9 */	imul ecx
/* 0000068D 038D  03 D1 */	add edx, ecx
/* 0000068F 038F  C1 FA 08 */	sar edx, 8
/* 00000692 0392  8B C2 */	mov eax, edx
/* 00000694 0394  C1 E8 1F */	shr eax, 0x1f
/* 00000697 0397  03 D0 */	add edx, eax
/* 00000699 0399  52 */	push edx
/* 0000069A 039A  E8 00 00 00 00 */	call _foo
/* 0000069F 039F  8B 4C 24 14 */	mov ecx, dword ptr [esp + 0x14]
/* 000006A3 03A3  B8 D3 4D 62 10 */	mov eax, 0x10624dd3
/* 000006A8 03A8  F7 E9 */	imul ecx
/* 000006AA 03AA  C1 FA 06 */	sar edx, 6
/* 000006AD 03AD  8B CA */	mov ecx, edx
/* 000006AF 03AF  C1 E9 1F */	shr ecx, 0x1f
/* 000006B2 03B2  03 D1 */	add edx, ecx
/* 000006B4 03B4  52 */	push edx
/* 000006B5 03B5  E8 00 00 00 00 */	call _foo
/* 000006BA 03BA  8B 4C 24 18 */	mov ecx, dword ptr [esp + 0x18]
/* 000006BE 03BE  B8 AD 8B DB 68 */	mov eax, 0x68db8bad
/* 000006C3 03C3  F7 E9 */	imul ecx
/* 000006C5 03C5  C1 FA 0C */	sar edx, 0xc
/* 000006C8 03C8  8B C2 */	mov eax, edx
/* 000006CA 03CA  C1 E8 1F */	shr eax, 0x1f
/* 000006CD 03CD  03 D0 */	add edx, eax
/* 000006CF 03CF  52 */	push edx
/* 000006D0 03D0  E8 00 00 00 00 */	call _foo
/* 000006D5 03D5  8B 4C 24 1C */	mov ecx, dword ptr [esp + 0x1c]
/* 000006D9 03D9  B8 89 B5 F8 14 */	mov eax, 0x14f8b589
/* 000006DE 03DE  F7 E9 */	imul ecx
/* 000006E0 03E0  C1 FA 0D */	sar edx, 0xd
/* 000006E3 03E3  8B CA */	mov ecx, edx
/* 000006E5 03E5  C1 E9 1F */	shr ecx, 0x1f
/* 000006E8 03E8  03 D1 */	add edx, ecx
/* 000006EA 03EA  52 */	push edx
/* 000006EB 03EB  E8 00 00 00 00 */	call _foo
/* 000006F0 03F0  8B 4C 24 20 */	mov ecx, dword ptr [esp + 0x20]
/* 000006F4 03F4  B8 83 DE 1B 43 */	mov eax, 0x431bde83
/* 000006F9 03F9  F7 E9 */	imul ecx
/* 000006FB 03FB  C1 FA 12 */	sar edx, 0x12
/* 000006FE 03FE  8B C2 */	mov eax, edx
/* 00000700 0400  C1 E8 1F */	shr eax, 0x1f
/* 00000703 0403  03 D0 */	add edx, eax
/* 00000705 0405  52 */	push edx
/* 00000706 0406  E8 00 00 00 00 */	call _foo
/* 0000070B 040B  8B 4C 24 24 */	mov ecx, dword ptr [esp + 0x24]
/* 0000070F 040F  B8 6B CA 5F 6B */	mov eax, 0x6b5fca6b
/* 00000714 0414  F7 E9 */	imul ecx
/* 00000716 0416  C1 FA 16 */	sar edx, 0x16
/* 00000719 0419  8B CA */	mov ecx, edx
/* 0000071B 041B  C1 E9 1F */	shr ecx, 0x1f
/* 0000071E 041E  03 D1 */	add edx, ecx
/* 00000720 0420  52 */	push edx
/* 00000721 0421  E8 00 00 00 00 */	call _foo
/* 00000726 0426  8B 4C 24 28 */	mov ecx, dword ptr [esp + 0x28]
/* 0000072A 042A  B8 89 3B E6 55 */	mov eax, 0x55e63b89
/* 0000072F 042F  F7 E9 */	imul ecx
/* 00000731 0431  C1 FA 19 */	sar edx, 0x19
/* 00000734 0434  8B C2 */	mov eax, edx
/* 00000736 0436  C1 E8 1F */	shr eax, 0x1f
/* 00000739 0439  03 D0 */	add edx, eax
/* 0000073B 043B  52 */	push edx
/* 0000073C 043C  E8 00 00 00 00 */	call _foo
/* 00000741 0441  8B 4C 24 2C */	mov ecx, dword ptr [esp + 0x2c]
/* 00000745 0445  B8 05 00 00 80 */	mov eax, 0x80000005
/* 0000074A 044A  F7 E9 */	imul ecx
/* 0000074C 044C  03 D1 */	add edx, ecx
/* 0000074E 044E  C1 FA 1D */	sar edx, 0x1d
/* 00000751 0451  8B CA */	mov ecx, edx
/* 00000753 0453  C1 E9 1F */	shr ecx, 0x1f
/* 00000756 0456  03 D1 */	add edx, ecx
/* 00000758 0458  52 */	push edx
/* 00000759 0459  E8 00 00 00 00 */	call _foo
/* 0000075E 045E  8B 4C 24 30 */	mov ecx, dword ptr [esp + 0x30]
/* 00000762 0462  B8 03 00 00 80 */	mov eax, 0x80000003
/* 00000767 0467  F7 E9 */	imul ecx
/* 00000769 0469  03 D1 */	add edx, ecx
/* 0000076B 046B  C1 FA 1D */	sar edx, 0x1d
/* 0000076E 046E  8B C2 */	mov eax, edx
/* 00000770 0470  C1 E8 1F */	shr eax, 0x1f
/* 00000773 0473  03 D0 */	add edx, eax
/* 00000775 0475  52 */	push edx
/* 00000776 0476  E8 00 00 00 00 */	call _foo
/* 0000077B 047B  8B 44 24 34 */	mov eax, dword ptr [esp + 0x34]
/* 0000077F 047F  99 */	cdq
/* 00000780 0480  81 E2 FF FF FF 3F */	and edx, 0x3fffffff
/* 00000786 0486  03 C2 */	add eax, edx
/* 00000788 0488  C1 F8 1E */	sar eax, 0x1e
/* 0000078B 048B  50 */	push eax
/* 0000078C 048C  E8 00 00 00 00 */	call _foo
/* 00000791 0491  8B 4C 24 38 */	mov ecx, dword ptr [esp + 0x38]
/* 00000795 0495  B8 FF FF FF 7F */	mov eax, 0x7fffffff
/* 0000079A 049A  F7 E9 */	imul ecx
/* 0000079C 049C  C1 FA 1D */	sar edx, 0x1d
/* 0000079F 049F  8B CA */	mov ecx, edx
/* 000007A1 04A1  C1 E9 1F */	shr ecx, 0x1f
/* 000007A4 04A4  03 D1 */	add edx, ecx
/* 000007A6 04A6  52 */	push edx
/* 000007A7 04A7  E8 00 00 00 00 */	call _foo
/* 000007AC 04AC  8B 4C 24 3C */	mov ecx, dword ptr [esp + 0x3c]
/* 000007B0 04B0  B8 01 00 00 20 */	mov eax, 0x20000001
/* 000007B5 04B5  F7 E9 */	imul ecx
/* 000007B7 04B7  C1 FA 1C */	sar edx, 0x1c
/* 000007BA 04BA  8B C2 */	mov eax, edx
/* 000007BC 04BC  C1 E8 1F */	shr eax, 0x1f
/* 000007BF 04BF  03 D0 */	add edx, eax
/* 000007C1 04C1  52 */	push edx
/* 000007C2 04C2  E8 00 00 00 00 */	call _foo
/* 000007C7 04C7  8B 4C 24 40 */	mov ecx, dword ptr [esp + 0x40]
/* 000007CB 04CB  B8 03 00 00 80 */	mov eax, 0x80000003
/* 000007D0 04D0  F7 E9 */	imul ecx
/* 000007D2 04D2  03 D1 */	add edx, ecx
/* 000007D4 04D4  C1 FA 1E */	sar edx, 0x1e
/* 000007D7 04D7  8B CA */	mov ecx, edx
/* 000007D9 04D9  C1 E9 1F */	shr ecx, 0x1f
/* 000007DC 04DC  03 D1 */	add edx, ecx
/* 000007DE 04DE  52 */	push edx
/* 000007DF 04DF  E8 00 00 00 00 */	call _foo
/* 000007E4 04E4  83 C4 40 */	add esp, 0x40
/* 000007E7 04E7  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 000007EB 04EB  B8 01 00 00 40 */	mov eax, 0x40000001
/* 000007F0 04F0  F7 E9 */	imul ecx
/* 000007F2 04F2  C1 FA 1D */	sar edx, 0x1d
/* 000007F5 04F5  8B C2 */	mov eax, edx
/* 000007F7 04F7  C1 E8 1F */	shr eax, 0x1f
/* 000007FA 04FA  03 D0 */	add edx, eax
/* 000007FC 04FC  52 */	push edx
/* 000007FD 04FD  E8 00 00 00 00 */	call _foo
/* 00000802 0502  8B 4C 24 08 */	mov ecx, dword ptr [esp + 8]
/* 00000806 0506  C1 E9 1F */	shr ecx, 0x1f
/* 00000809 0509  51 */	push ecx
/* 0000080A 050A  E8 00 00 00 00 */	call _foo
/* 0000080F 050F  8B 4C 24 0C */	mov ecx, dword ptr [esp + 0xc]
/* 00000813 0513  B8 FF FF FF BF */	mov eax, 0xbfffffff
/* 00000818 0518  F7 E9 */	imul ecx
/* 0000081A 051A  C1 FA 1D */	sar edx, 0x1d
/* 0000081D 051D  8B C2 */	mov eax, edx
/* 0000081F 051F  C1 E8 1F */	shr eax, 0x1f
/* 00000822 0522  03 D0 */	add edx, eax
/* 00000824 0524  52 */	push edx
/* 00000825 0525  E8 00 00 00 00 */	call _foo
/* 0000082A 052A  8B 4C 24 10 */	mov ecx, dword ptr [esp + 0x10]
/* 0000082E 052E  B8 FD FF FF 7F */	mov eax, 0x7ffffffd
/* 00000833 0533  F7 E9 */	imul ecx
/* 00000835 0535  2B D1 */	sub edx, ecx
/* 00000837 0537  C1 FA 1E */	sar edx, 0x1e
/* 0000083A 053A  8B CA */	mov ecx, edx
/* 0000083C 053C  C1 E9 1F */	shr ecx, 0x1f
/* 0000083F 053F  03 D1 */	add edx, ecx
/* 00000841 0541  52 */	push edx
/* 00000842 0542  E8 00 00 00 00 */	call _foo
/* 00000847 0547  8B 4C 24 14 */	mov ecx, dword ptr [esp + 0x14]
/* 0000084B 054B  B8 99 99 99 99 */	mov eax, 0x99999999
/* 00000850 0550  F7 E9 */	imul ecx
/* 00000852 0552  C1 FA 02 */	sar edx, 2
/* 00000855 0555  8B C2 */	mov eax, edx
/* 00000857 0557  C1 E8 1F */	shr eax, 0x1f
/* 0000085A 055A  03 D0 */	add edx, eax
/* 0000085C 055C  52 */	push edx
/* 0000085D 055D  E8 00 00 00 00 */	call _foo
/* 00000862 0562  8B 4C 24 18 */	mov ecx, dword ptr [esp + 0x18]
/* 00000866 0566  B8 6D DB B6 6D */	mov eax, 0x6db6db6d
/* 0000086B 056B  F7 E9 */	imul ecx
/* 0000086D 056D  2B D1 */	sub edx, ecx
/* 0000086F 056F  C1 FA 02 */	sar edx, 2
/* 00000872 0572  8B CA */	mov ecx, edx
/* 00000874 0574  C1 E9 1F */	shr ecx, 0x1f
/* 00000877 0577  03 D1 */	add edx, ecx
/* 00000879 0579  52 */	push edx
/* 0000087A 057A  E8 00 00 00 00 */	call _foo
/* 0000087F 057F  8B 4C 24 1C */	mov ecx, dword ptr [esp + 0x1c]
/* 00000883 0583  B8 99 99 99 99 */	mov eax, 0x99999999
/* 00000888 0588  F7 E9 */	imul ecx
/* 0000088A 058A  D1 FA */	sar edx, 1
/* 0000088C 058C  8B C2 */	mov eax, edx
/* 0000088E 058E  C1 E8 1F */	shr eax, 0x1f
/* 00000891 0591  03 D0 */	add edx, eax
/* 00000893 0593  52 */	push edx
/* 00000894 0594  E8 00 00 00 00 */	call _foo
/* 00000899 0599  8B 44 24 20 */	mov eax, dword ptr [esp + 0x20]
/* 0000089D 059D  99 */	cdq
/* 0000089E 059E  83 E2 03 */	and edx, 3
/* 000008A1 05A1  03 C2 */	add eax, edx
/* 000008A3 05A3  C1 F8 02 */	sar eax, 2
/* 000008A6 05A6  F7 D8 */	neg eax
/* 000008A8 05A8  50 */	push eax
/* 000008A9 05A9  E8 00 00 00 00 */	call _foo
/* 000008AE 05AE  8B 4C 24 24 */	mov ecx, dword ptr [esp + 0x24]
/* 000008B2 05B2  B8 55 55 55 55 */	mov eax, 0x55555555
/* 000008B7 05B7  F7 E9 */	imul ecx
/* 000008B9 05B9  2B D1 */	sub edx, ecx
/* 000008BB 05BB  D1 FA */	sar edx, 1
/* 000008BD 05BD  8B CA */	mov ecx, edx
/* 000008BF 05BF  C1 E9 1F */	shr ecx, 0x1f
/* 000008C2 05C2  03 D1 */	add edx, ecx
/* 000008C4 05C4  52 */	push edx
/* 000008C5 05C5  E8 00 00 00 00 */	call _foo
/* 000008CA 05CA  8B 44 24 28 */	mov eax, dword ptr [esp + 0x28]
/* 000008CE 05CE  99 */	cdq
/* 000008CF 05CF  2B C2 */	sub eax, edx
/* 000008D1 05D1  D1 F8 */	sar eax, 1
/* 000008D3 05D3  F7 D8 */	neg eax
/* 000008D5 05D5  50 */	push eax
/* 000008D6 05D6  E8 00 00 00 00 */	call _foo
/* 000008DB 05DB  8B 54 24 2C */	mov edx, dword ptr [esp + 0x2c]
/* 000008DF 05DF  F7 DA */	neg edx
/* 000008E1 05E1  52 */	push edx
/* 000008E2 05E2  E8 00 00 00 00 */	call _foo
/* 000008E7 05E7  83 C4 2C */	add esp, 0x2c
/* 000008EA 05EA  C3 */	ret
/* 000008EB 05EB  90 */	nop
/* 000008EC 05EC  90 */	nop
/* 000008ED 05ED  90 */	nop
/* 000008EE 05EE  90 */	nop
/* 000008EF 05EF  90 */	nop

test_s32_mod:
/* 000008F0 0000  51 */	push ecx
/* 000008F1 0001  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 000008F5 0005  6A 00 */	push 0
/* 000008F7 0007  89 44 24 04 */	mov dword ptr [esp + 4], eax
/* 000008FB 000B  E8 00 00 00 00 */	call _foo
/* 00000900 0010  8B 4C 24 0C */	mov ecx, dword ptr [esp + 0xc]
/* 00000904 0014  81 E1 01 00 00 80 */	and ecx, 0x80000001
/* 0000090A 001A  79 05 */	jns .L00000911
/* 0000090C 001C  49 */	dec ecx
/* 0000090D 001D  83 C9 FE */	or ecx, 0xfffffffe
/* 00000910 0020  41 */	inc ecx
.L00000911:
/* 00000911 0021  51 */	push ecx
/* 00000912 0022  E8 00 00 00 00 */	call _foo
/* 00000917 0027  8B 44 24 10 */	mov eax, dword ptr [esp + 0x10]
/* 0000091B 002B  B9 03 00 00 00 */	mov ecx, 3
/* 00000920 0030  99 */	cdq
/* 00000921 0031  F7 F9 */	idiv ecx
/* 00000923 0033  52 */	push edx
/* 00000924 0034  E8 00 00 00 00 */	call _foo
/* 00000929 0039  8B 54 24 14 */	mov edx, dword ptr [esp + 0x14]
/* 0000092D 003D  81 E2 03 00 00 80 */	and edx, 0x80000003
/* 00000933 0043  79 05 */	jns .L0000093A
/* 00000935 0045  4A */	dec edx
/* 00000936 0046  83 CA FC */	or edx, 0xfffffffc
/* 00000939 0049  42 */	inc edx
.L0000093A:
/* 0000093A 004A  52 */	push edx
/* 0000093B 004B  E8 00 00 00 00 */	call _foo
/* 00000940 0050  8B 44 24 18 */	mov eax, dword ptr [esp + 0x18]
/* 00000944 0054  B9 05 00 00 00 */	mov ecx, 5
/* 00000949 0059  99 */	cdq
/* 0000094A 005A  F7 F9 */	idiv ecx
/* 0000094C 005C  52 */	push edx
/* 0000094D 005D  E8 00 00 00 00 */	call _foo
/* 00000952 0062  8B 44 24 1C */	mov eax, dword ptr [esp + 0x1c]
/* 00000956 0066  B9 06 00 00 00 */	mov ecx, 6
/* 0000095B 006B  99 */	cdq
/* 0000095C 006C  F7 F9 */	idiv ecx
/* 0000095E 006E  52 */	push edx
/* 0000095F 006F  E8 00 00 00 00 */	call _foo
/* 00000964 0074  8B 44 24 20 */	mov eax, dword ptr [esp + 0x20]
/* 00000968 0078  B9 07 00 00 00 */	mov ecx, 7
/* 0000096D 007D  99 */	cdq
/* 0000096E 007E  F7 F9 */	idiv ecx
/* 00000970 0080  52 */	push edx
/* 00000971 0081  E8 00 00 00 00 */	call _foo
/* 00000976 0086  8B 54 24 24 */	mov edx, dword ptr [esp + 0x24]
/* 0000097A 008A  81 E2 07 00 00 80 */	and edx, 0x80000007
/* 00000980 0090  79 05 */	jns .L00000987
/* 00000982 0092  4A */	dec edx
/* 00000983 0093  83 CA F8 */	or edx, 0xfffffff8
/* 00000986 0096  42 */	inc edx
.L00000987:
/* 00000987 0097  52 */	push edx
/* 00000988 0098  E8 00 00 00 00 */	call _foo
/* 0000098D 009D  8B 44 24 28 */	mov eax, dword ptr [esp + 0x28]
/* 00000991 00A1  B9 09 00 00 00 */	mov ecx, 9
/* 00000996 00A6  99 */	cdq
/* 00000997 00A7  F7 F9 */	idiv ecx
/* 00000999 00A9  52 */	push edx
/* 0000099A 00AA  E8 00 00 00 00 */	call _foo
/* 0000099F 00AF  8B 44 24 2C */	mov eax, dword ptr [esp + 0x2c]
/* 000009A3 00B3  B9 0A 00 00 00 */	mov ecx, 0xa
/* 000009A8 00B8  99 */	cdq
/* 000009A9 00B9  F7 F9 */	idiv ecx
/* 000009AB 00BB  52 */	push edx
/* 000009AC 00BC  E8 00 00 00 00 */	call _foo
/* 000009B1 00C1  8B 44 24 30 */	mov eax, dword ptr [esp + 0x30]
/* 000009B5 00C5  B9 0B 00 00 00 */	mov ecx, 0xb
/* 000009BA 00CA  99 */	cdq
/* 000009BB 00CB  F7 F9 */	idiv ecx
/* 000009BD 00CD  52 */	push edx
/* 000009BE 00CE  E8 00 00 00 00 */	call _foo
/* 000009C3 00D3  8B 44 24 34 */	mov eax, dword ptr [esp + 0x34]
/* 000009C7 00D7  B9 0C 00 00 00 */	mov ecx, 0xc
/* 000009CC 00DC  99 */	cdq
/* 000009CD 00DD  F7 F9 */	idiv ecx
/* 000009CF 00DF  52 */	push edx
/* 000009D0 00E0  E8 00 00 00 00 */	call _foo
/* 000009D5 00E5  8B 44 24 38 */	mov eax, dword ptr [esp + 0x38]
/* 000009D9 00E9  B9 0D 00 00 00 */	mov ecx, 0xd
/* 000009DE 00EE  99 */	cdq
/* 000009DF 00EF  F7 F9 */	idiv ecx
/* 000009E1 00F1  52 */	push edx
/* 000009E2 00F2  E8 00 00 00 00 */	call _foo
/* 000009E7 00F7  8B 44 24 3C */	mov eax, dword ptr [esp + 0x3c]
/* 000009EB 00FB  B9 0E 00 00 00 */	mov ecx, 0xe
/* 000009F0 0100  99 */	cdq
/* 000009F1 0101  F7 F9 */	idiv ecx
/* 000009F3 0103  52 */	push edx
/* 000009F4 0104  E8 00 00 00 00 */	call _foo
/* 000009F9 0109  8B 44 24 40 */	mov eax, dword ptr [esp + 0x40]
/* 000009FD 010D  B9 0F 00 00 00 */	mov ecx, 0xf
/* 00000A02 0112  99 */	cdq
/* 00000A03 0113  F7 F9 */	idiv ecx
/* 00000A05 0115  52 */	push edx
/* 00000A06 0116  E8 00 00 00 00 */	call _foo
/* 00000A0B 011B  8B 54 24 44 */	mov edx, dword ptr [esp + 0x44]
/* 00000A0F 011F  81 E2 0F 00 00 80 */	and edx, 0x8000000f
/* 00000A15 0125  79 05 */	jns .L00000A1C
/* 00000A17 0127  4A */	dec edx
/* 00000A18 0128  83 CA F0 */	or edx, 0xfffffff0
/* 00000A1B 012B  42 */	inc edx
.L00000A1C:
/* 00000A1C 012C  52 */	push edx
/* 00000A1D 012D  E8 00 00 00 00 */	call _foo
/* 00000A22 0132  83 C4 40 */	add esp, 0x40
/* 00000A25 0135  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000A29 0139  99 */	cdq
/* 00000A2A 013A  B9 11 00 00 00 */	mov ecx, 0x11
/* 00000A2F 013F  F7 F9 */	idiv ecx
/* 00000A31 0141  52 */	push edx
/* 00000A32 0142  E8 00 00 00 00 */	call _foo
/* 00000A37 0147  8B 44 24 0C */	mov eax, dword ptr [esp + 0xc]
/* 00000A3B 014B  B9 12 00 00 00 */	mov ecx, 0x12
/* 00000A40 0150  99 */	cdq
/* 00000A41 0151  F7 F9 */	idiv ecx
/* 00000A43 0153  52 */	push edx
/* 00000A44 0154  E8 00 00 00 00 */	call _foo
/* 00000A49 0159  8B 44 24 10 */	mov eax, dword ptr [esp + 0x10]
/* 00000A4D 015D  B9 13 00 00 00 */	mov ecx, 0x13
/* 00000A52 0162  99 */	cdq
/* 00000A53 0163  F7 F9 */	idiv ecx
/* 00000A55 0165  52 */	push edx
/* 00000A56 0166  E8 00 00 00 00 */	call _foo
/* 00000A5B 016B  8B 44 24 14 */	mov eax, dword ptr [esp + 0x14]
/* 00000A5F 016F  B9 14 00 00 00 */	mov ecx, 0x14
/* 00000A64 0174  99 */	cdq
/* 00000A65 0175  F7 F9 */	idiv ecx
/* 00000A67 0177  52 */	push edx
/* 00000A68 0178  E8 00 00 00 00 */	call _foo
/* 00000A6D 017D  8B 44 24 18 */	mov eax, dword ptr [esp + 0x18]
/* 00000A71 0181  B9 15 00 00 00 */	mov ecx, 0x15
/* 00000A76 0186  99 */	cdq
/* 00000A77 0187  F7 F9 */	idiv ecx
/* 00000A79 0189  52 */	push edx
/* 00000A7A 018A  E8 00 00 00 00 */	call _foo
/* 00000A7F 018F  8B 44 24 1C */	mov eax, dword ptr [esp + 0x1c]
/* 00000A83 0193  B9 16 00 00 00 */	mov ecx, 0x16
/* 00000A88 0198  99 */	cdq
/* 00000A89 0199  F7 F9 */	idiv ecx
/* 00000A8B 019B  52 */	push edx
/* 00000A8C 019C  E8 00 00 00 00 */	call _foo
/* 00000A91 01A1  8B 44 24 20 */	mov eax, dword ptr [esp + 0x20]
/* 00000A95 01A5  B9 17 00 00 00 */	mov ecx, 0x17
/* 00000A9A 01AA  99 */	cdq
/* 00000A9B 01AB  F7 F9 */	idiv ecx
/* 00000A9D 01AD  52 */	push edx
/* 00000A9E 01AE  E8 00 00 00 00 */	call _foo
/* 00000AA3 01B3  8B 44 24 24 */	mov eax, dword ptr [esp + 0x24]
/* 00000AA7 01B7  B9 18 00 00 00 */	mov ecx, 0x18
/* 00000AAC 01BC  99 */	cdq
/* 00000AAD 01BD  F7 F9 */	idiv ecx
/* 00000AAF 01BF  52 */	push edx
/* 00000AB0 01C0  E8 00 00 00 00 */	call _foo
/* 00000AB5 01C5  8B 44 24 28 */	mov eax, dword ptr [esp + 0x28]
/* 00000AB9 01C9  B9 19 00 00 00 */	mov ecx, 0x19
/* 00000ABE 01CE  99 */	cdq
/* 00000ABF 01CF  F7 F9 */	idiv ecx
/* 00000AC1 01D1  52 */	push edx
/* 00000AC2 01D2  E8 00 00 00 00 */	call _foo
/* 00000AC7 01D7  8B 44 24 2C */	mov eax, dword ptr [esp + 0x2c]
/* 00000ACB 01DB  B9 1A 00 00 00 */	mov ecx, 0x1a
/* 00000AD0 01E0  99 */	cdq
/* 00000AD1 01E1  F7 F9 */	idiv ecx
/* 00000AD3 01E3  52 */	push edx
/* 00000AD4 01E4  E8 00 00 00 00 */	call _foo
/* 00000AD9 01E9  8B 44 24 30 */	mov eax, dword ptr [esp + 0x30]
/* 00000ADD 01ED  B9 1B 00 00 00 */	mov ecx, 0x1b
/* 00000AE2 01F2  99 */	cdq
/* 00000AE3 01F3  F7 F9 */	idiv ecx
/* 00000AE5 01F5  52 */	push edx
/* 00000AE6 01F6  E8 00 00 00 00 */	call _foo
/* 00000AEB 01FB  8B 44 24 34 */	mov eax, dword ptr [esp + 0x34]
/* 00000AEF 01FF  B9 1C 00 00 00 */	mov ecx, 0x1c
/* 00000AF4 0204  99 */	cdq
/* 00000AF5 0205  F7 F9 */	idiv ecx
/* 00000AF7 0207  52 */	push edx
/* 00000AF8 0208  E8 00 00 00 00 */	call _foo
/* 00000AFD 020D  8B 44 24 38 */	mov eax, dword ptr [esp + 0x38]
/* 00000B01 0211  B9 1D 00 00 00 */	mov ecx, 0x1d
/* 00000B06 0216  99 */	cdq
/* 00000B07 0217  F7 F9 */	idiv ecx
/* 00000B09 0219  52 */	push edx
/* 00000B0A 021A  E8 00 00 00 00 */	call _foo
/* 00000B0F 021F  8B 44 24 3C */	mov eax, dword ptr [esp + 0x3c]
/* 00000B13 0223  B9 1E 00 00 00 */	mov ecx, 0x1e
/* 00000B18 0228  99 */	cdq
/* 00000B19 0229  F7 F9 */	idiv ecx
/* 00000B1B 022B  52 */	push edx
/* 00000B1C 022C  E8 00 00 00 00 */	call _foo
/* 00000B21 0231  8B 44 24 40 */	mov eax, dword ptr [esp + 0x40]
/* 00000B25 0235  B9 1F 00 00 00 */	mov ecx, 0x1f
/* 00000B2A 023A  99 */	cdq
/* 00000B2B 023B  F7 F9 */	idiv ecx
/* 00000B2D 023D  52 */	push edx
/* 00000B2E 023E  E8 00 00 00 00 */	call _foo
/* 00000B33 0243  8B 54 24 44 */	mov edx, dword ptr [esp + 0x44]
/* 00000B37 0247  81 E2 1F 00 00 80 */	and edx, 0x8000001f
/* 00000B3D 024D  79 05 */	jns .L00000B44
/* 00000B3F 024F  4A */	dec edx
/* 00000B40 0250  83 CA E0 */	or edx, 0xffffffe0
/* 00000B43 0253  42 */	inc edx
.L00000B44:
/* 00000B44 0254  52 */	push edx
/* 00000B45 0255  E8 00 00 00 00 */	call _foo
/* 00000B4A 025A  83 C4 40 */	add esp, 0x40
/* 00000B4D 025D  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000B51 0261  99 */	cdq
/* 00000B52 0262  B9 21 00 00 00 */	mov ecx, 0x21
/* 00000B57 0267  F7 F9 */	idiv ecx
/* 00000B59 0269  52 */	push edx
/* 00000B5A 026A  E8 00 00 00 00 */	call _foo
/* 00000B5F 026F  8B 44 24 0C */	mov eax, dword ptr [esp + 0xc]
/* 00000B63 0273  B9 64 00 00 00 */	mov ecx, 0x64
/* 00000B68 0278  99 */	cdq
/* 00000B69 0279  F7 F9 */	idiv ecx
/* 00000B6B 027B  52 */	push edx
/* 00000B6C 027C  E8 00 00 00 00 */	call _foo
/* 00000B71 0281  8B 44 24 10 */	mov eax, dword ptr [esp + 0x10]
/* 00000B75 0285  B9 FF 00 00 00 */	mov ecx, 0xff
/* 00000B7A 028A  99 */	cdq
/* 00000B7B 028B  F7 F9 */	idiv ecx
/* 00000B7D 028D  52 */	push edx
/* 00000B7E 028E  E8 00 00 00 00 */	call _foo
/* 00000B83 0293  8B 44 24 14 */	mov eax, dword ptr [esp + 0x14]
/* 00000B87 0297  B9 68 01 00 00 */	mov ecx, 0x168
/* 00000B8C 029C  99 */	cdq
/* 00000B8D 029D  F7 F9 */	idiv ecx
/* 00000B8F 029F  52 */	push edx
/* 00000B90 02A0  E8 00 00 00 00 */	call _foo
/* 00000B95 02A5  8B 44 24 18 */	mov eax, dword ptr [esp + 0x18]
/* 00000B99 02A9  B9 E8 03 00 00 */	mov ecx, 0x3e8
/* 00000B9E 02AE  99 */	cdq
/* 00000B9F 02AF  F7 F9 */	idiv ecx
/* 00000BA1 02B1  52 */	push edx
/* 00000BA2 02B2  E8 00 00 00 00 */	call _foo
/* 00000BA7 02B7  8B 44 24 1C */	mov eax, dword ptr [esp + 0x1c]
/* 00000BAB 02BB  B9 10 27 00 00 */	mov ecx, 0x2710
/* 00000BB0 02C0  99 */	cdq
/* 00000BB1 02C1  F7 F9 */	idiv ecx
/* 00000BB3 02C3  52 */	push edx
/* 00000BB4 02C4  E8 00 00 00 00 */	call _foo
/* 00000BB9 02C9  8B 44 24 20 */	mov eax, dword ptr [esp + 0x20]
/* 00000BBD 02CD  B9 A0 86 01 00 */	mov ecx, 0x186a0
/* 00000BC2 02D2  99 */	cdq
/* 00000BC3 02D3  F7 F9 */	idiv ecx
/* 00000BC5 02D5  52 */	push edx
/* 00000BC6 02D6  E8 00 00 00 00 */	call _foo
/* 00000BCB 02DB  8B 44 24 24 */	mov eax, dword ptr [esp + 0x24]
/* 00000BCF 02DF  B9 40 42 0F 00 */	mov ecx, 0xf4240
/* 00000BD4 02E4  99 */	cdq
/* 00000BD5 02E5  F7 F9 */	idiv ecx
/* 00000BD7 02E7  52 */	push edx
/* 00000BD8 02E8  E8 00 00 00 00 */	call _foo
/* 00000BDD 02ED  8B 44 24 28 */	mov eax, dword ptr [esp + 0x28]
/* 00000BE1 02F1  B9 80 96 98 00 */	mov ecx, 0x989680
/* 00000BE6 02F6  99 */	cdq
/* 00000BE7 02F7  F7 F9 */	idiv ecx
/* 00000BE9 02F9  52 */	push edx
/* 00000BEA 02FA  E8 00 00 00 00 */	call _foo
/* 00000BEF 02FF  8B 44 24 2C */	mov eax, dword ptr [esp + 0x2c]
/* 00000BF3 0303  B9 00 E1 F5 05 */	mov ecx, 0x5f5e100
/* 00000BF8 0308  99 */	cdq
/* 00000BF9 0309  F7 F9 */	idiv ecx
/* 00000BFB 030B  52 */	push edx
/* 00000BFC 030C  E8 00 00 00 00 */	call _foo
/* 00000C01 0311  8B 44 24 30 */	mov eax, dword ptr [esp + 0x30]
/* 00000C05 0315  B9 FE FF FF 3F */	mov ecx, 0x3ffffffe
/* 00000C0A 031A  99 */	cdq
/* 00000C0B 031B  F7 F9 */	idiv ecx
/* 00000C0D 031D  52 */	push edx
/* 00000C0E 031E  E8 00 00 00 00 */	call _foo
/* 00000C13 0323  8B 44 24 34 */	mov eax, dword ptr [esp + 0x34]
/* 00000C17 0327  B9 FF FF FF 3F */	mov ecx, 0x3fffffff
/* 00000C1C 032C  99 */	cdq
/* 00000C1D 032D  F7 F9 */	idiv ecx
/* 00000C1F 032F  52 */	push edx
/* 00000C20 0330  E8 00 00 00 00 */	call _foo
/* 00000C25 0335  8B 54 24 38 */	mov edx, dword ptr [esp + 0x38]
/* 00000C29 0339  81 E2 FF FF FF BF */	and edx, 0xbfffffff
/* 00000C2F 033F  79 08 */	jns .L00000C39
/* 00000C31 0341  4A */	dec edx
/* 00000C32 0342  81 CA 00 00 00 C0 */	or edx, 0xc0000000
/* 00000C38 0348  42 */	inc edx
.L00000C39:
/* 00000C39 0349  52 */	push edx
/* 00000C3A 034A  E8 00 00 00 00 */	call _foo
/* 00000C3F 034F  8B 44 24 3C */	mov eax, dword ptr [esp + 0x3c]
/* 00000C43 0353  B9 01 00 00 40 */	mov ecx, 0x40000001
/* 00000C48 0358  99 */	cdq
/* 00000C49 0359  F7 F9 */	idiv ecx
/* 00000C4B 035B  52 */	push edx
/* 00000C4C 035C  E8 00 00 00 00 */	call _foo
/* 00000C51 0361  8B 44 24 40 */	mov eax, dword ptr [esp + 0x40]
/* 00000C55 0365  B9 FD FF FF 7F */	mov ecx, 0x7ffffffd
/* 00000C5A 036A  99 */	cdq
/* 00000C5B 036B  F7 F9 */	idiv ecx
/* 00000C5D 036D  52 */	push edx
/* 00000C5E 036E  E8 00 00 00 00 */	call _foo
/* 00000C63 0373  8B 44 24 44 */	mov eax, dword ptr [esp + 0x44]
/* 00000C67 0377  B9 FE FF FF 7F */	mov ecx, 0x7ffffffe
/* 00000C6C 037C  99 */	cdq
/* 00000C6D 037D  F7 F9 */	idiv ecx
/* 00000C6F 037F  52 */	push edx
/* 00000C70 0380  E8 00 00 00 00 */	call _foo
/* 00000C75 0385  83 C4 40 */	add esp, 0x40
/* 00000C78 0388  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000C7C 038C  99 */	cdq
/* 00000C7D 038D  B9 FF FF FF 7F */	mov ecx, 0x7fffffff
/* 00000C82 0392  F7 F9 */	idiv ecx
/* 00000C84 0394  52 */	push edx
/* 00000C85 0395  E8 00 00 00 00 */	call _foo
/* 00000C8A 039A  8B 54 24 0C */	mov edx, dword ptr [esp + 0xc]
/* 00000C8E 039E  81 E2 FF FF FF 7F */	and edx, 0x7fffffff
/* 00000C94 03A4  52 */	push edx
/* 00000C95 03A5  E8 00 00 00 00 */	call _foo
/* 00000C9A 03AA  8B 44 24 10 */	mov eax, dword ptr [esp + 0x10]
/* 00000C9E 03AE  B9 01 00 00 80 */	mov ecx, 0x80000001
/* 00000CA3 03B3  99 */	cdq
/* 00000CA4 03B4  F7 F9 */	idiv ecx
/* 00000CA6 03B6  52 */	push edx
/* 00000CA7 03B7  E8 00 00 00 00 */	call _foo
/* 00000CAC 03BC  8B 44 24 14 */	mov eax, dword ptr [esp + 0x14]
/* 00000CB0 03C0  B9 02 00 00 80 */	mov ecx, 0x80000002
/* 00000CB5 03C5  99 */	cdq
/* 00000CB6 03C6  F7 F9 */	idiv ecx
/* 00000CB8 03C8  52 */	push edx
/* 00000CB9 03C9  E8 00 00 00 00 */	call _foo
/* 00000CBE 03CE  8B 44 24 18 */	mov eax, dword ptr [esp + 0x18]
/* 00000CC2 03D2  B9 F6 FF FF FF */	mov ecx, 0xfffffff6
/* 00000CC7 03D7  99 */	cdq
/* 00000CC8 03D8  F7 F9 */	idiv ecx
/* 00000CCA 03DA  52 */	push edx
/* 00000CCB 03DB  E8 00 00 00 00 */	call _foo
/* 00000CD0 03E0  8B 44 24 1C */	mov eax, dword ptr [esp + 0x1c]
/* 00000CD4 03E4  B9 F9 FF FF FF */	mov ecx, 0xfffffff9
/* 00000CD9 03E9  99 */	cdq
/* 00000CDA 03EA  F7 F9 */	idiv ecx
/* 00000CDC 03EC  52 */	push edx
/* 00000CDD 03ED  E8 00 00 00 00 */	call _foo
/* 00000CE2 03F2  8B 44 24 20 */	mov eax, dword ptr [esp + 0x20]
/* 00000CE6 03F6  B9 FB FF FF FF */	mov ecx, 0xfffffffb
/* 00000CEB 03FB  99 */	cdq
/* 00000CEC 03FC  F7 F9 */	idiv ecx
/* 00000CEE 03FE  52 */	push edx
/* 00000CEF 03FF  E8 00 00 00 00 */	call _foo
/* 00000CF4 0404  8B 44 24 24 */	mov eax, dword ptr [esp + 0x24]
/* 00000CF8 0408  99 */	cdq
/* 00000CF9 0409  33 C2 */	xor eax, edx
/* 00000CFB 040B  2B C2 */	sub eax, edx
/* 00000CFD 040D  83 E0 03 */	and eax, 3
/* 00000D00 0410  33 C2 */	xor eax, edx
/* 00000D02 0412  2B C2 */	sub eax, edx
/* 00000D04 0414  50 */	push eax
/* 00000D05 0415  E8 00 00 00 00 */	call _foo
/* 00000D0A 041A  8B 44 24 28 */	mov eax, dword ptr [esp + 0x28]
/* 00000D0E 041E  B9 FD FF FF FF */	mov ecx, 0xfffffffd
/* 00000D13 0423  99 */	cdq
/* 00000D14 0424  F7 F9 */	idiv ecx
/* 00000D16 0426  52 */	push edx
/* 00000D17 0427  E8 00 00 00 00 */	call _foo
/* 00000D1C 042C  8B 44 24 2C */	mov eax, dword ptr [esp + 0x2c]
/* 00000D20 0430  99 */	cdq
/* 00000D21 0431  33 C2 */	xor eax, edx
/* 00000D23 0433  2B C2 */	sub eax, edx
/* 00000D25 0435  83 E0 01 */	and eax, 1
/* 00000D28 0438  33 C2 */	xor eax, edx
/* 00000D2A 043A  2B C2 */	sub eax, edx
/* 00000D2C 043C  50 */	push eax
/* 00000D2D 043D  E8 00 00 00 00 */	call _foo
/* 00000D32 0442  8B 44 24 30 */	mov eax, dword ptr [esp + 0x30]
/* 00000D36 0446  99 */	cdq
/* 00000D37 0447  33 C2 */	xor eax, edx
/* 00000D39 0449  2B C2 */	sub eax, edx
/* 00000D3B 044B  83 E0 00 */	and eax, 0
/* 00000D3E 044E  33 C2 */	xor eax, edx
/* 00000D40 0450  2B C2 */	sub eax, edx
/* 00000D42 0452  50 */	push eax
/* 00000D43 0453  E8 00 00 00 00 */	call _foo
/* 00000D48 0458  83 C4 30 */	add esp, 0x30
/* 00000D4B 045B  C3 */	ret
/* 00000D4C 045C  90 */	nop
/* 00000D4D 045D  90 */	nop
/* 00000D4E 045E  90 */	nop
/* 00000D4F 045F  90 */	nop

test_u32_div:
/* 00000D50 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000D54 0004  50 */	push eax
/* 00000D55 0005  E8 00 00 00 00 */	call _foo
/* 00000D5A 000A  8B 4C 24 08 */	mov ecx, dword ptr [esp + 8]
/* 00000D5E 000E  D1 E9 */	shr ecx, 1
/* 00000D60 0010  51 */	push ecx
/* 00000D61 0011  E8 00 00 00 00 */	call _foo
/* 00000D66 0016  B8 AB AA AA AA */	mov eax, 0xaaaaaaab
/* 00000D6B 001B  F7 64 24 0C */	mul dword ptr [esp + 0xc]
/* 00000D6F 001F  D1 EA */	shr edx, 1
/* 00000D71 0021  52 */	push edx
/* 00000D72 0022  E8 00 00 00 00 */	call _foo
/* 00000D77 0027  8B 54 24 10 */	mov edx, dword ptr [esp + 0x10]
/* 00000D7B 002B  C1 EA 02 */	shr edx, 2
/* 00000D7E 002E  52 */	push edx
/* 00000D7F 002F  E8 00 00 00 00 */	call _foo
/* 00000D84 0034  B8 CD CC CC CC */	mov eax, 0xcccccccd
/* 00000D89 0039  F7 64 24 14 */	mul dword ptr [esp + 0x14]
/* 00000D8D 003D  C1 EA 02 */	shr edx, 2
/* 00000D90 0040  52 */	push edx
/* 00000D91 0041  E8 00 00 00 00 */	call _foo
/* 00000D96 0046  B8 AB AA AA AA */	mov eax, 0xaaaaaaab
/* 00000D9B 004B  F7 64 24 18 */	mul dword ptr [esp + 0x18]
/* 00000D9F 004F  C1 EA 02 */	shr edx, 2
/* 00000DA2 0052  52 */	push edx
/* 00000DA3 0053  E8 00 00 00 00 */	call _foo
/* 00000DA8 0058  8B 4C 24 1C */	mov ecx, dword ptr [esp + 0x1c]
/* 00000DAC 005C  B8 25 49 92 24 */	mov eax, 0x24924925
/* 00000DB1 0061  F7 E1 */	mul ecx
/* 00000DB3 0063  2B CA */	sub ecx, edx
/* 00000DB5 0065  D1 E9 */	shr ecx, 1
/* 00000DB7 0067  03 CA */	add ecx, edx
/* 00000DB9 0069  C1 E9 02 */	shr ecx, 2
/* 00000DBC 006C  51 */	push ecx
/* 00000DBD 006D  E8 00 00 00 00 */	call _foo
/* 00000DC2 0072  8B 44 24 20 */	mov eax, dword ptr [esp + 0x20]
/* 00000DC6 0076  C1 E8 03 */	shr eax, 3
/* 00000DC9 0079  50 */	push eax
/* 00000DCA 007A  E8 00 00 00 00 */	call _foo
/* 00000DCF 007F  B8 39 8E E3 38 */	mov eax, 0x38e38e39
/* 00000DD4 0084  F7 64 24 24 */	mul dword ptr [esp + 0x24]
/* 00000DD8 0088  D1 EA */	shr edx, 1
/* 00000DDA 008A  52 */	push edx
/* 00000DDB 008B  E8 00 00 00 00 */	call _foo
/* 00000DE0 0090  B8 CD CC CC CC */	mov eax, 0xcccccccd
/* 00000DE5 0095  F7 64 24 28 */	mul dword ptr [esp + 0x28]
/* 00000DE9 0099  C1 EA 03 */	shr edx, 3
/* 00000DEC 009C  52 */	push edx
/* 00000DED 009D  E8 00 00 00 00 */	call _foo
/* 00000DF2 00A2  B8 A3 8B 2E BA */	mov eax, 0xba2e8ba3
/* 00000DF7 00A7  F7 64 24 2C */	mul dword ptr [esp + 0x2c]
/* 00000DFB 00AB  C1 EA 03 */	shr edx, 3
/* 00000DFE 00AE  52 */	push edx
/* 00000DFF 00AF  E8 00 00 00 00 */	call _foo
/* 00000E04 00B4  B8 AB AA AA AA */	mov eax, 0xaaaaaaab
/* 00000E09 00B9  F7 64 24 30 */	mul dword ptr [esp + 0x30]
/* 00000E0D 00BD  C1 EA 03 */	shr edx, 3
/* 00000E10 00C0  52 */	push edx
/* 00000E11 00C1  E8 00 00 00 00 */	call _foo
/* 00000E16 00C6  B8 4F EC C4 4E */	mov eax, 0x4ec4ec4f
/* 00000E1B 00CB  F7 64 24 34 */	mul dword ptr [esp + 0x34]
/* 00000E1F 00CF  C1 EA 02 */	shr edx, 2
/* 00000E22 00D2  52 */	push edx
/* 00000E23 00D3  E8 00 00 00 00 */	call _foo
/* 00000E28 00D8  8B 4C 24 38 */	mov ecx, dword ptr [esp + 0x38]
/* 00000E2C 00DC  B8 25 49 92 24 */	mov eax, 0x24924925
/* 00000E31 00E1  F7 E1 */	mul ecx
/* 00000E33 00E3  2B CA */	sub ecx, edx
/* 00000E35 00E5  D1 E9 */	shr ecx, 1
/* 00000E37 00E7  03 CA */	add ecx, edx
/* 00000E39 00E9  C1 E9 03 */	shr ecx, 3
/* 00000E3C 00EC  51 */	push ecx
/* 00000E3D 00ED  E8 00 00 00 00 */	call _foo
/* 00000E42 00F2  B8 89 88 88 88 */	mov eax, 0x88888889
/* 00000E47 00F7  F7 64 24 3C */	mul dword ptr [esp + 0x3c]
/* 00000E4B 00FB  C1 EA 03 */	shr edx, 3
/* 00000E4E 00FE  52 */	push edx
/* 00000E4F 00FF  E8 00 00 00 00 */	call _foo
/* 00000E54 0104  8B 4C 24 40 */	mov ecx, dword ptr [esp + 0x40]
/* 00000E58 0108  C1 E9 04 */	shr ecx, 4
/* 00000E5B 010B  51 */	push ecx
/* 00000E5C 010C  E8 00 00 00 00 */	call _foo
/* 00000E61 0111  B8 F1 F0 F0 F0 */	mov eax, 0xf0f0f0f1
/* 00000E66 0116  83 C4 40 */	add esp, 0x40
/* 00000E69 0119  F7 64 24 04 */	mul dword ptr [esp + 4]
/* 00000E6D 011D  C1 EA 04 */	shr edx, 4
/* 00000E70 0120  52 */	push edx
/* 00000E71 0121  E8 00 00 00 00 */	call _foo
/* 00000E76 0126  B8 39 8E E3 38 */	mov eax, 0x38e38e39
/* 00000E7B 012B  F7 64 24 08 */	mul dword ptr [esp + 8]
/* 00000E7F 012F  C1 EA 02 */	shr edx, 2
/* 00000E82 0132  52 */	push edx
/* 00000E83 0133  E8 00 00 00 00 */	call _foo
/* 00000E88 0138  8B 4C 24 0C */	mov ecx, dword ptr [esp + 0xc]
/* 00000E8C 013C  B8 CB 6B 28 AF */	mov eax, 0xaf286bcb
/* 00000E91 0141  F7 E1 */	mul ecx
/* 00000E93 0143  2B CA */	sub ecx, edx
/* 00000E95 0145  D1 E9 */	shr ecx, 1
/* 00000E97 0147  03 CA */	add ecx, edx
/* 00000E99 0149  C1 E9 04 */	shr ecx, 4
/* 00000E9C 014C  51 */	push ecx
/* 00000E9D 014D  E8 00 00 00 00 */	call _foo
/* 00000EA2 0152  B8 CD CC CC CC */	mov eax, 0xcccccccd
/* 00000EA7 0157  F7 64 24 10 */	mul dword ptr [esp + 0x10]
/* 00000EAB 015B  C1 EA 04 */	shr edx, 4
/* 00000EAE 015E  52 */	push edx
/* 00000EAF 015F  E8 00 00 00 00 */	call _foo
/* 00000EB4 0164  8B 4C 24 14 */	mov ecx, dword ptr [esp + 0x14]
/* 00000EB8 0168  B8 87 61 18 86 */	mov eax, 0x86186187
/* 00000EBD 016D  F7 E1 */	mul ecx
/* 00000EBF 016F  2B CA */	sub ecx, edx
/* 00000EC1 0171  D1 E9 */	shr ecx, 1
/* 00000EC3 0173  03 CA */	add ecx, edx
/* 00000EC5 0175  C1 E9 04 */	shr ecx, 4
/* 00000EC8 0178  51 */	push ecx
/* 00000EC9 0179  E8 00 00 00 00 */	call _foo
/* 00000ECE 017E  B8 A3 8B 2E BA */	mov eax, 0xba2e8ba3
/* 00000ED3 0183  F7 64 24 18 */	mul dword ptr [esp + 0x18]
/* 00000ED7 0187  C1 EA 04 */	shr edx, 4
/* 00000EDA 018A  52 */	push edx
/* 00000EDB 018B  E8 00 00 00 00 */	call _foo
/* 00000EE0 0190  B8 C9 42 16 B2 */	mov eax, 0xb21642c9
/* 00000EE5 0195  F7 64 24 1C */	mul dword ptr [esp + 0x1c]
/* 00000EE9 0199  C1 EA 04 */	shr edx, 4
/* 00000EEC 019C  52 */	push edx
/* 00000EED 019D  E8 00 00 00 00 */	call _foo
/* 00000EF2 01A2  B8 AB AA AA AA */	mov eax, 0xaaaaaaab
/* 00000EF7 01A7  F7 64 24 20 */	mul dword ptr [esp + 0x20]
/* 00000EFB 01AB  C1 EA 04 */	shr edx, 4
/* 00000EFE 01AE  52 */	push edx
/* 00000EFF 01AF  E8 00 00 00 00 */	call _foo
/* 00000F04 01B4  B8 1F 85 EB 51 */	mov eax, 0x51eb851f
/* 00000F09 01B9  F7 64 24 24 */	mul dword ptr [esp + 0x24]
/* 00000F0D 01BD  C1 EA 03 */	shr edx, 3
/* 00000F10 01C0  52 */	push edx
/* 00000F11 01C1  E8 00 00 00 00 */	call _foo
/* 00000F16 01C6  B8 4F EC C4 4E */	mov eax, 0x4ec4ec4f
/* 00000F1B 01CB  F7 64 24 28 */	mul dword ptr [esp + 0x28]
/* 00000F1F 01CF  C1 EA 03 */	shr edx, 3
/* 00000F22 01D2  52 */	push edx
/* 00000F23 01D3  E8 00 00 00 00 */	call _foo
/* 00000F28 01D8  8B 4C 24 2C */	mov ecx, dword ptr [esp + 0x2c]
/* 00000F2C 01DC  B8 DB 4B 68 2F */	mov eax, 0x2f684bdb
/* 00000F31 01E1  F7 E1 */	mul ecx
/* 00000F33 01E3  2B CA */	sub ecx, edx
/* 00000F35 01E5  D1 E9 */	shr ecx, 1
/* 00000F37 01E7  03 CA */	add ecx, edx
/* 00000F39 01E9  C1 E9 04 */	shr ecx, 4
/* 00000F3C 01EC  51 */	push ecx
/* 00000F3D 01ED  E8 00 00 00 00 */	call _foo
/* 00000F42 01F2  8B 4C 24 30 */	mov ecx, dword ptr [esp + 0x30]
/* 00000F46 01F6  B8 25 49 92 24 */	mov eax, 0x24924925
/* 00000F4B 01FB  F7 E1 */	mul ecx
/* 00000F4D 01FD  2B CA */	sub ecx, edx
/* 00000F4F 01FF  D1 E9 */	shr ecx, 1
/* 00000F51 0201  03 CA */	add ecx, edx
/* 00000F53 0203  C1 E9 04 */	shr ecx, 4
/* 00000F56 0206  51 */	push ecx
/* 00000F57 0207  E8 00 00 00 00 */	call _foo
/* 00000F5C 020C  B8 09 CB 3D 8D */	mov eax, 0x8d3dcb09
/* 00000F61 0211  F7 64 24 34 */	mul dword ptr [esp + 0x34]
/* 00000F65 0215  C1 EA 04 */	shr edx, 4
/* 00000F68 0218  52 */	push edx
/* 00000F69 0219  E8 00 00 00 00 */	call _foo
/* 00000F6E 021E  B8 89 88 88 88 */	mov eax, 0x88888889
/* 00000F73 0223  F7 64 24 38 */	mul dword ptr [esp + 0x38]
/* 00000F77 0227  C1 EA 04 */	shr edx, 4
/* 00000F7A 022A  52 */	push edx
/* 00000F7B 022B  E8 00 00 00 00 */	call _foo
/* 00000F80 0230  8B 4C 24 3C */	mov ecx, dword ptr [esp + 0x3c]
/* 00000F84 0234  B8 85 10 42 08 */	mov eax, 0x8421085
/* 00000F89 0239  F7 E1 */	mul ecx
/* 00000F8B 023B  2B CA */	sub ecx, edx
/* 00000F8D 023D  D1 E9 */	shr ecx, 1
/* 00000F8F 023F  03 CA */	add ecx, edx
/* 00000F91 0241  C1 E9 04 */	shr ecx, 4
/* 00000F94 0244  51 */	push ecx
/* 00000F95 0245  E8 00 00 00 00 */	call _foo
/* 00000F9A 024A  8B 54 24 40 */	mov edx, dword ptr [esp + 0x40]
/* 00000F9E 024E  C1 EA 05 */	shr edx, 5
/* 00000FA1 0251  52 */	push edx
/* 00000FA2 0252  E8 00 00 00 00 */	call _foo
/* 00000FA7 0257  B8 E1 83 0F 3E */	mov eax, 0x3e0f83e1
/* 00000FAC 025C  83 C4 40 */	add esp, 0x40
/* 00000FAF 025F  F7 64 24 04 */	mul dword ptr [esp + 4]
/* 00000FB3 0263  C1 EA 03 */	shr edx, 3
/* 00000FB6 0266  52 */	push edx
/* 00000FB7 0267  E8 00 00 00 00 */	call _foo
/* 00000FBC 026C  B8 1F 85 EB 51 */	mov eax, 0x51eb851f
/* 00000FC1 0271  F7 64 24 08 */	mul dword ptr [esp + 8]
/* 00000FC5 0275  C1 EA 05 */	shr edx, 5
/* 00000FC8 0278  52 */	push edx
/* 00000FC9 0279  E8 00 00 00 00 */	call _foo
/* 00000FCE 027E  B8 81 80 80 80 */	mov eax, 0x80808081
/* 00000FD3 0283  F7 64 24 0C */	mul dword ptr [esp + 0xc]
/* 00000FD7 0287  C1 EA 07 */	shr edx, 7
/* 00000FDA 028A  52 */	push edx
/* 00000FDB 028B  E8 00 00 00 00 */	call _foo
/* 00000FE0 0290  8B 4C 24 10 */	mov ecx, dword ptr [esp + 0x10]
/* 00000FE4 0294  B8 6D C1 16 6C */	mov eax, 0x6c16c16d
/* 00000FE9 0299  F7 E1 */	mul ecx
/* 00000FEB 029B  2B CA */	sub ecx, edx
/* 00000FED 029D  D1 E9 */	shr ecx, 1
/* 00000FEF 029F  03 CA */	add ecx, edx
/* 00000FF1 02A1  C1 E9 08 */	shr ecx, 8
/* 00000FF4 02A4  51 */	push ecx
/* 00000FF5 02A5  E8 00 00 00 00 */	call _foo
/* 00000FFA 02AA  B8 D3 4D 62 10 */	mov eax, 0x10624dd3
/* 00000FFF 02AF  F7 64 24 14 */	mul dword ptr [esp + 0x14]
/* 00001003 02B3  C1 EA 06 */	shr edx, 6
/* 00001006 02B6  52 */	push edx
/* 00001007 02B7  E8 00 00 00 00 */	call _foo
/* 0000100C 02BC  B8 59 17 B7 D1 */	mov eax, 0xd1b71759
/* 00001011 02C1  F7 64 24 18 */	mul dword ptr [esp + 0x18]
/* 00001015 02C5  C1 EA 0D */	shr edx, 0xd
/* 00001018 02C8  52 */	push edx
/* 00001019 02C9  E8 00 00 00 00 */	call _foo
/* 0000101E 02CE  8B 4C 24 1C */	mov ecx, dword ptr [esp + 0x1c]
/* 00001022 02D2  B8 8F 58 8B 4F */	mov eax, 0x4f8b588f
/* 00001027 02D7  F7 E1 */	mul ecx
/* 00001029 02D9  2B CA */	sub ecx, edx
/* 0000102B 02DB  D1 E9 */	shr ecx, 1
/* 0000102D 02DD  03 CA */	add ecx, edx
/* 0000102F 02DF  C1 E9 10 */	shr ecx, 0x10
/* 00001032 02E2  51 */	push ecx
/* 00001033 02E3  E8 00 00 00 00 */	call _foo
/* 00001038 02E8  B8 83 DE 1B 43 */	mov eax, 0x431bde83
/* 0000103D 02ED  F7 64 24 20 */	mul dword ptr [esp + 0x20]
/* 00001041 02F1  C1 EA 12 */	shr edx, 0x12
/* 00001044 02F4  52 */	push edx
/* 00001045 02F5  E8 00 00 00 00 */	call _foo
/* 0000104A 02FA  B8 6B CA 5F 6B */	mov eax, 0x6b5fca6b
/* 0000104F 02FF  F7 64 24 24 */	mul dword ptr [esp + 0x24]
/* 00001053 0303  C1 EA 16 */	shr edx, 0x16
/* 00001056 0306  52 */	push edx
/* 00001057 0307  E8 00 00 00 00 */	call _foo
/* 0000105C 030C  B8 89 3B E6 55 */	mov eax, 0x55e63b89
/* 00001061 0311  F7 64 24 28 */	mul dword ptr [esp + 0x28]
/* 00001065 0315  C1 EA 19 */	shr edx, 0x19
/* 00001068 0318  52 */	push edx
/* 00001069 0319  E8 00 00 00 00 */	call _foo
/* 0000106E 031E  8B 44 24 2C */	mov eax, dword ptr [esp + 0x2c]
/* 00001072 0322  C1 E8 1E */	shr eax, 0x1e
/* 00001075 0325  50 */	push eax
/* 00001076 0326  E8 00 00 00 00 */	call _foo
/* 0000107B 032B  B8 FD FF FF FF */	mov eax, 0xfffffffd
/* 00001080 0330  F7 64 24 30 */	mul dword ptr [esp + 0x30]
/* 00001084 0334  C1 EA 1E */	shr edx, 0x1e
/* 00001087 0337  52 */	push edx
/* 00001088 0338  E8 00 00 00 00 */	call _foo
/* 0000108D 033D  8B 4C 24 34 */	mov ecx, dword ptr [esp + 0x34]
/* 00001091 0341  B8 05 00 00 00 */	mov eax, 5
/* 00001096 0346  F7 E1 */	mul ecx
/* 00001098 0348  2B CA */	sub ecx, edx
/* 0000109A 034A  D1 E9 */	shr ecx, 1
/* 0000109C 034C  03 CA */	add ecx, edx
/* 0000109E 034E  C1 E9 1E */	shr ecx, 0x1e
/* 000010A1 0351  51 */	push ecx
/* 000010A2 0352  E8 00 00 00 00 */	call _foo
/* 000010A7 0357  8B 4C 24 38 */	mov ecx, dword ptr [esp + 0x38]
/* 000010AB 035B  B8 03 00 00 00 */	mov eax, 3
/* 000010B0 0360  F7 E1 */	mul ecx
/* 000010B2 0362  2B CA */	sub ecx, edx
/* 000010B4 0364  D1 E9 */	shr ecx, 1
/* 000010B6 0366  03 CA */	add ecx, edx
/* 000010B8 0368  C1 E9 1E */	shr ecx, 0x1e
/* 000010BB 036B  51 */	push ecx
/* 000010BC 036C  E8 00 00 00 00 */	call _foo
/* 000010C1 0371  8B 4C 24 3C */	mov ecx, dword ptr [esp + 0x3c]
/* 000010C5 0375  C1 E9 1F */	shr ecx, 0x1f
/* 000010C8 0378  51 */	push ecx
/* 000010C9 0379  E8 00 00 00 00 */	call _foo
/* 000010CE 037E  8B 4C 24 40 */	mov ecx, dword ptr [esp + 0x40]
/* 000010D2 0382  B8 FD FF FF FF */	mov eax, 0xfffffffd
/* 000010D7 0387  F7 E1 */	mul ecx
/* 000010D9 0389  2B CA */	sub ecx, edx
/* 000010DB 038B  D1 E9 */	shr ecx, 1
/* 000010DD 038D  03 CA */	add ecx, edx
/* 000010DF 038F  C1 E9 1F */	shr ecx, 0x1f
/* 000010E2 0392  51 */	push ecx
/* 000010E3 0393  E8 00 00 00 00 */	call _foo
/* 000010E8 0398  83 C4 40 */	add esp, 0x40
/* 000010EB 039B  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 000010EF 039F  33 D2 */	xor edx, edx
/* 000010F1 03A1  B9 FE FF FF FF */	mov ecx, 0xfffffffe
/* 000010F6 03A6  F7 F1 */	div ecx
/* 000010F8 03A8  50 */	push eax
/* 000010F9 03A9  E8 00 00 00 00 */	call _foo
/* 000010FE 03AE  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00001102 03B2  33 D2 */	xor edx, edx
/* 00001104 03B4  83 C9 FF */	or ecx, 0xffffffff
/* 00001107 03B7  F7 F1 */	div ecx
/* 00001109 03B9  50 */	push eax
/* 0000110A 03BA  E8 00 00 00 00 */	call _foo
/* 0000110F 03BF  83 C4 08 */	add esp, 8
/* 00001112 03C2  C3 */	ret
/* 00001113 03C3  90 */	nop
/* 00001114 03C4  90 */	nop
/* 00001115 03C5  90 */	nop
/* 00001116 03C6  90 */	nop
/* 00001117 03C7  90 */	nop
/* 00001118 03C8  90 */	nop
/* 00001119 03C9  90 */	nop
/* 0000111A 03CA  90 */	nop
/* 0000111B 03CB  90 */	nop
/* 0000111C 03CC  90 */	nop
/* 0000111D 03CD  90 */	nop
/* 0000111E 03CE  90 */	nop
/* 0000111F 03CF  90 */	nop

test_u32_mod:
/* 00001120 0000  51 */	push ecx
/* 00001121 0001  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00001125 0005  6A 00 */	push 0
/* 00001127 0007  89 44 24 04 */	mov dword ptr [esp + 4], eax
/* 0000112B 000B  E8 00 00 00 00 */	call _foo
/* 00001130 0010  8B 4C 24 0C */	mov ecx, dword ptr [esp + 0xc]
/* 00001134 0014  83 E1 01 */	and ecx, 1
/* 00001137 0017  51 */	push ecx
/* 00001138 0018  E8 00 00 00 00 */	call _foo
/* 0000113D 001D  8B 44 24 10 */	mov eax, dword ptr [esp + 0x10]
/* 00001141 0021  33 D2 */	xor edx, edx
/* 00001143 0023  B9 03 00 00 00 */	mov ecx, 3
/* 00001148 0028  F7 F1 */	div ecx
/* 0000114A 002A  52 */	push edx
/* 0000114B 002B  E8 00 00 00 00 */	call _foo
/* 00001150 0030  8B 54 24 14 */	mov edx, dword ptr [esp + 0x14]
/* 00001154 0034  83 E2 03 */	and edx, 3
/* 00001157 0037  52 */	push edx
/* 00001158 0038  E8 00 00 00 00 */	call _foo
/* 0000115D 003D  8B 44 24 18 */	mov eax, dword ptr [esp + 0x18]
/* 00001161 0041  33 D2 */	xor edx, edx
/* 00001163 0043  B9 05 00 00 00 */	mov ecx, 5
/* 00001168 0048  F7 F1 */	div ecx
/* 0000116A 004A  52 */	push edx
/* 0000116B 004B  E8 00 00 00 00 */	call _foo
/* 00001170 0050  8B 44 24 1C */	mov eax, dword ptr [esp + 0x1c]
/* 00001174 0054  33 D2 */	xor edx, edx
/* 00001176 0056  B9 06 00 00 00 */	mov ecx, 6
/* 0000117B 005B  F7 F1 */	div ecx
/* 0000117D 005D  52 */	push edx
/* 0000117E 005E  E8 00 00 00 00 */	call _foo
/* 00001183 0063  8B 44 24 20 */	mov eax, dword ptr [esp + 0x20]
/* 00001187 0067  33 D2 */	xor edx, edx
/* 00001189 0069  B9 07 00 00 00 */	mov ecx, 7
/* 0000118E 006E  F7 F1 */	div ecx
/* 00001190 0070  52 */	push edx
/* 00001191 0071  E8 00 00 00 00 */	call _foo
/* 00001196 0076  8B 54 24 24 */	mov edx, dword ptr [esp + 0x24]
/* 0000119A 007A  83 E2 07 */	and edx, 7
/* 0000119D 007D  52 */	push edx
/* 0000119E 007E  E8 00 00 00 00 */	call _foo
/* 000011A3 0083  8B 44 24 28 */	mov eax, dword ptr [esp + 0x28]
/* 000011A7 0087  33 D2 */	xor edx, edx
/* 000011A9 0089  B9 09 00 00 00 */	mov ecx, 9
/* 000011AE 008E  F7 F1 */	div ecx
/* 000011B0 0090  52 */	push edx
/* 000011B1 0091  E8 00 00 00 00 */	call _foo
/* 000011B6 0096  8B 44 24 2C */	mov eax, dword ptr [esp + 0x2c]
/* 000011BA 009A  33 D2 */	xor edx, edx
/* 000011BC 009C  B9 0A 00 00 00 */	mov ecx, 0xa
/* 000011C1 00A1  F7 F1 */	div ecx
/* 000011C3 00A3  52 */	push edx
/* 000011C4 00A4  E8 00 00 00 00 */	call _foo
/* 000011C9 00A9  8B 44 24 30 */	mov eax, dword ptr [esp + 0x30]
/* 000011CD 00AD  33 D2 */	xor edx, edx
/* 000011CF 00AF  B9 0B 00 00 00 */	mov ecx, 0xb
/* 000011D4 00B4  F7 F1 */	div ecx
/* 000011D6 00B6  52 */	push edx
/* 000011D7 00B7  E8 00 00 00 00 */	call _foo
/* 000011DC 00BC  8B 44 24 34 */	mov eax, dword ptr [esp + 0x34]
/* 000011E0 00C0  33 D2 */	xor edx, edx
/* 000011E2 00C2  B9 0C 00 00 00 */	mov ecx, 0xc
/* 000011E7 00C7  F7 F1 */	div ecx
/* 000011E9 00C9  52 */	push edx
/* 000011EA 00CA  E8 00 00 00 00 */	call _foo
/* 000011EF 00CF  8B 44 24 38 */	mov eax, dword ptr [esp + 0x38]
/* 000011F3 00D3  33 D2 */	xor edx, edx
/* 000011F5 00D5  B9 0D 00 00 00 */	mov ecx, 0xd
/* 000011FA 00DA  F7 F1 */	div ecx
/* 000011FC 00DC  52 */	push edx
/* 000011FD 00DD  E8 00 00 00 00 */	call _foo
/* 00001202 00E2  8B 44 24 3C */	mov eax, dword ptr [esp + 0x3c]
/* 00001206 00E6  33 D2 */	xor edx, edx
/* 00001208 00E8  B9 0E 00 00 00 */	mov ecx, 0xe
/* 0000120D 00ED  F7 F1 */	div ecx
/* 0000120F 00EF  52 */	push edx
/* 00001210 00F0  E8 00 00 00 00 */	call _foo
/* 00001215 00F5  8B 44 24 40 */	mov eax, dword ptr [esp + 0x40]
/* 00001219 00F9  33 D2 */	xor edx, edx
/* 0000121B 00FB  B9 0F 00 00 00 */	mov ecx, 0xf
/* 00001220 0100  F7 F1 */	div ecx
/* 00001222 0102  52 */	push edx
/* 00001223 0103  E8 00 00 00 00 */	call _foo
/* 00001228 0108  8B 54 24 44 */	mov edx, dword ptr [esp + 0x44]
/* 0000122C 010C  83 E2 0F */	and edx, 0xf
/* 0000122F 010F  52 */	push edx
/* 00001230 0110  E8 00 00 00 00 */	call _foo
/* 00001235 0115  83 C4 40 */	add esp, 0x40
/* 00001238 0118  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 0000123C 011C  33 D2 */	xor edx, edx
/* 0000123E 011E  B9 11 00 00 00 */	mov ecx, 0x11
/* 00001243 0123  F7 F1 */	div ecx
/* 00001245 0125  52 */	push edx
/* 00001246 0126  E8 00 00 00 00 */	call _foo
/* 0000124B 012B  8B 44 24 0C */	mov eax, dword ptr [esp + 0xc]
/* 0000124F 012F  33 D2 */	xor edx, edx
/* 00001251 0131  B9 12 00 00 00 */	mov ecx, 0x12
/* 00001256 0136  F7 F1 */	div ecx
/* 00001258 0138  52 */	push edx
/* 00001259 0139  E8 00 00 00 00 */	call _foo
/* 0000125E 013E  8B 44 24 10 */	mov eax, dword ptr [esp + 0x10]
/* 00001262 0142  33 D2 */	xor edx, edx
/* 00001264 0144  B9 13 00 00 00 */	mov ecx, 0x13
/* 00001269 0149  F7 F1 */	div ecx
/* 0000126B 014B  52 */	push edx
/* 0000126C 014C  E8 00 00 00 00 */	call _foo
/* 00001271 0151  8B 44 24 14 */	mov eax, dword ptr [esp + 0x14]
/* 00001275 0155  33 D2 */	xor edx, edx
/* 00001277 0157  B9 14 00 00 00 */	mov ecx, 0x14
/* 0000127C 015C  F7 F1 */	div ecx
/* 0000127E 015E  52 */	push edx
/* 0000127F 015F  E8 00 00 00 00 */	call _foo
/* 00001284 0164  8B 44 24 18 */	mov eax, dword ptr [esp + 0x18]
/* 00001288 0168  33 D2 */	xor edx, edx
/* 0000128A 016A  B9 15 00 00 00 */	mov ecx, 0x15
/* 0000128F 016F  F7 F1 */	div ecx
/* 00001291 0171  52 */	push edx
/* 00001292 0172  E8 00 00 00 00 */	call _foo
/* 00001297 0177  8B 44 24 1C */	mov eax, dword ptr [esp + 0x1c]
/* 0000129B 017B  33 D2 */	xor edx, edx
/* 0000129D 017D  B9 16 00 00 00 */	mov ecx, 0x16
/* 000012A2 0182  F7 F1 */	div ecx
/* 000012A4 0184  52 */	push edx
/* 000012A5 0185  E8 00 00 00 00 */	call _foo
/* 000012AA 018A  8B 44 24 20 */	mov eax, dword ptr [esp + 0x20]
/* 000012AE 018E  33 D2 */	xor edx, edx
/* 000012B0 0190  B9 17 00 00 00 */	mov ecx, 0x17
/* 000012B5 0195  F7 F1 */	div ecx
/* 000012B7 0197  52 */	push edx
/* 000012B8 0198  E8 00 00 00 00 */	call _foo
/* 000012BD 019D  8B 44 24 24 */	mov eax, dword ptr [esp + 0x24]
/* 000012C1 01A1  33 D2 */	xor edx, edx
/* 000012C3 01A3  B9 18 00 00 00 */	mov ecx, 0x18
/* 000012C8 01A8  F7 F1 */	div ecx
/* 000012CA 01AA  52 */	push edx
/* 000012CB 01AB  E8 00 00 00 00 */	call _foo
/* 000012D0 01B0  8B 44 24 28 */	mov eax, dword ptr [esp + 0x28]
/* 000012D4 01B4  33 D2 */	xor edx, edx
/* 000012D6 01B6  B9 19 00 00 00 */	mov ecx, 0x19
/* 000012DB 01BB  F7 F1 */	div ecx
/* 000012DD 01BD  52 */	push edx
/* 000012DE 01BE  E8 00 00 00 00 */	call _foo
/* 000012E3 01C3  8B 44 24 2C */	mov eax, dword ptr [esp + 0x2c]
/* 000012E7 01C7  33 D2 */	xor edx, edx
/* 000012E9 01C9  B9 1A 00 00 00 */	mov ecx, 0x1a
/* 000012EE 01CE  F7 F1 */	div ecx
/* 000012F0 01D0  52 */	push edx
/* 000012F1 01D1  E8 00 00 00 00 */	call _foo
/* 000012F6 01D6  8B 44 24 30 */	mov eax, dword ptr [esp + 0x30]
/* 000012FA 01DA  33 D2 */	xor edx, edx
/* 000012FC 01DC  B9 1B 00 00 00 */	mov ecx, 0x1b
/* 00001301 01E1  F7 F1 */	div ecx
/* 00001303 01E3  52 */	push edx
/* 00001304 01E4  E8 00 00 00 00 */	call _foo
/* 00001309 01E9  8B 44 24 34 */	mov eax, dword ptr [esp + 0x34]
/* 0000130D 01ED  33 D2 */	xor edx, edx
/* 0000130F 01EF  B9 1C 00 00 00 */	mov ecx, 0x1c
/* 00001314 01F4  F7 F1 */	div ecx
/* 00001316 01F6  52 */	push edx
/* 00001317 01F7  E8 00 00 00 00 */	call _foo
/* 0000131C 01FC  8B 44 24 38 */	mov eax, dword ptr [esp + 0x38]
/* 00001320 0200  33 D2 */	xor edx, edx
/* 00001322 0202  B9 1D 00 00 00 */	mov ecx, 0x1d
/* 00001327 0207  F7 F1 */	div ecx
/* 00001329 0209  52 */	push edx
/* 0000132A 020A  E8 00 00 00 00 */	call _foo
/* 0000132F 020F  8B 44 24 3C */	mov eax, dword ptr [esp + 0x3c]
/* 00001333 0213  33 D2 */	xor edx, edx
/* 00001335 0215  B9 1E 00 00 00 */	mov ecx, 0x1e
/* 0000133A 021A  F7 F1 */	div ecx
/* 0000133C 021C  52 */	push edx
/* 0000133D 021D  E8 00 00 00 00 */	call _foo
/* 00001342 0222  8B 44 24 40 */	mov eax, dword ptr [esp + 0x40]
/* 00001346 0226  33 D2 */	xor edx, edx
/* 00001348 0228  B9 1F 00 00 00 */	mov ecx, 0x1f
/* 0000134D 022D  F7 F1 */	div ecx
/* 0000134F 022F  52 */	push edx
/* 00001350 0230  E8 00 00 00 00 */	call _foo
/* 00001355 0235  8B 54 24 44 */	mov edx, dword ptr [esp + 0x44]
/* 00001359 0239  83 E2 1F */	and edx, 0x1f
/* 0000135C 023C  52 */	push edx
/* 0000135D 023D  E8 00 00 00 00 */	call _foo
/* 00001362 0242  83 C4 40 */	add esp, 0x40
/* 00001365 0245  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00001369 0249  33 D2 */	xor edx, edx
/* 0000136B 024B  B9 21 00 00 00 */	mov ecx, 0x21
/* 00001370 0250  F7 F1 */	div ecx
/* 00001372 0252  52 */	push edx
/* 00001373 0253  E8 00 00 00 00 */	call _foo
/* 00001378 0258  8B 44 24 0C */	mov eax, dword ptr [esp + 0xc]
/* 0000137C 025C  33 D2 */	xor edx, edx
/* 0000137E 025E  B9 64 00 00 00 */	mov ecx, 0x64
/* 00001383 0263  F7 F1 */	div ecx
/* 00001385 0265  52 */	push edx
/* 00001386 0266  E8 00 00 00 00 */	call _foo
/* 0000138B 026B  8B 44 24 10 */	mov eax, dword ptr [esp + 0x10]
/* 0000138F 026F  33 D2 */	xor edx, edx
/* 00001391 0271  B9 FF 00 00 00 */	mov ecx, 0xff
/* 00001396 0276  F7 F1 */	div ecx
/* 00001398 0278  52 */	push edx
/* 00001399 0279  E8 00 00 00 00 */	call _foo
/* 0000139E 027E  8B 44 24 14 */	mov eax, dword ptr [esp + 0x14]
/* 000013A2 0282  33 D2 */	xor edx, edx
/* 000013A4 0284  B9 68 01 00 00 */	mov ecx, 0x168
/* 000013A9 0289  F7 F1 */	div ecx
/* 000013AB 028B  52 */	push edx
/* 000013AC 028C  E8 00 00 00 00 */	call _foo
/* 000013B1 0291  8B 44 24 18 */	mov eax, dword ptr [esp + 0x18]
/* 000013B5 0295  33 D2 */	xor edx, edx
/* 000013B7 0297  B9 E8 03 00 00 */	mov ecx, 0x3e8
/* 000013BC 029C  F7 F1 */	div ecx
/* 000013BE 029E  52 */	push edx
/* 000013BF 029F  E8 00 00 00 00 */	call _foo
/* 000013C4 02A4  8B 44 24 1C */	mov eax, dword ptr [esp + 0x1c]
/* 000013C8 02A8  33 D2 */	xor edx, edx
/* 000013CA 02AA  B9 10 27 00 00 */	mov ecx, 0x2710
/* 000013CF 02AF  F7 F1 */	div ecx
/* 000013D1 02B1  52 */	push edx
/* 000013D2 02B2  E8 00 00 00 00 */	call _foo
/* 000013D7 02B7  8B 44 24 20 */	mov eax, dword ptr [esp + 0x20]
/* 000013DB 02BB  33 D2 */	xor edx, edx
/* 000013DD 02BD  B9 A0 86 01 00 */	mov ecx, 0x186a0
/* 000013E2 02C2  F7 F1 */	div ecx
/* 000013E4 02C4  52 */	push edx
/* 000013E5 02C5  E8 00 00 00 00 */	call _foo
/* 000013EA 02CA  8B 44 24 24 */	mov eax, dword ptr [esp + 0x24]
/* 000013EE 02CE  33 D2 */	xor edx, edx
/* 000013F0 02D0  B9 40 42 0F 00 */	mov ecx, 0xf4240
/* 000013F5 02D5  F7 F1 */	div ecx
/* 000013F7 02D7  52 */	push edx
/* 000013F8 02D8  E8 00 00 00 00 */	call _foo
/* 000013FD 02DD  8B 44 24 28 */	mov eax, dword ptr [esp + 0x28]
/* 00001401 02E1  33 D2 */	xor edx, edx
/* 00001403 02E3  B9 80 96 98 00 */	mov ecx, 0x989680
/* 00001408 02E8  F7 F1 */	div ecx
/* 0000140A 02EA  52 */	push edx
/* 0000140B 02EB  E8 00 00 00 00 */	call _foo
/* 00001410 02F0  8B 44 24 2C */	mov eax, dword ptr [esp + 0x2c]
/* 00001414 02F4  33 D2 */	xor edx, edx
/* 00001416 02F6  B9 00 E1 F5 05 */	mov ecx, 0x5f5e100
/* 0000141B 02FB  F7 F1 */	div ecx
/* 0000141D 02FD  52 */	push edx
/* 0000141E 02FE  E8 00 00 00 00 */	call _foo
/* 00001423 0303  8B 54 24 30 */	mov edx, dword ptr [esp + 0x30]
/* 00001427 0307  81 E2 FF FF FF 3F */	and edx, 0x3fffffff
/* 0000142D 030D  52 */	push edx
/* 0000142E 030E  E8 00 00 00 00 */	call _foo
/* 00001433 0313  8B 44 24 34 */	mov eax, dword ptr [esp + 0x34]
/* 00001437 0317  33 D2 */	xor edx, edx
/* 00001439 0319  B9 01 00 00 40 */	mov ecx, 0x40000001
/* 0000143E 031E  F7 F1 */	div ecx
/* 00001440 0320  52 */	push edx
/* 00001441 0321  E8 00 00 00 00 */	call _foo
/* 00001446 0326  8B 44 24 38 */	mov eax, dword ptr [esp + 0x38]
/* 0000144A 032A  33 D2 */	xor edx, edx
/* 0000144C 032C  B9 FE FF FF 7F */	mov ecx, 0x7ffffffe
/* 00001451 0331  F7 F1 */	div ecx
/* 00001453 0333  52 */	push edx
/* 00001454 0334  E8 00 00 00 00 */	call _foo
/* 00001459 0339  8B 44 24 3C */	mov eax, dword ptr [esp + 0x3c]
/* 0000145D 033D  33 D2 */	xor edx, edx
/* 0000145F 033F  B9 FF FF FF 7F */	mov ecx, 0x7fffffff
/* 00001464 0344  F7 F1 */	div ecx
/* 00001466 0346  52 */	push edx
/* 00001467 0347  E8 00 00 00 00 */	call _foo
/* 0000146C 034C  8B 54 24 40 */	mov edx, dword ptr [esp + 0x40]
/* 00001470 0350  81 E2 FF FF FF 7F */	and edx, 0x7fffffff
/* 00001476 0356  52 */	push edx
/* 00001477 0357  E8 00 00 00 00 */	call _foo
/* 0000147C 035C  8B 44 24 44 */	mov eax, dword ptr [esp + 0x44]
/* 00001480 0360  33 D2 */	xor edx, edx
/* 00001482 0362  B9 01 00 00 80 */	mov ecx, 0x80000001
/* 00001487 0367  F7 F1 */	div ecx
/* 00001489 0369  52 */	push edx
/* 0000148A 036A  E8 00 00 00 00 */	call _foo
/* 0000148F 036F  83 C4 40 */	add esp, 0x40
/* 00001492 0372  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00001496 0376  33 D2 */	xor edx, edx
/* 00001498 0378  B9 FE FF FF FF */	mov ecx, 0xfffffffe
/* 0000149D 037D  F7 F1 */	div ecx
/* 0000149F 037F  52 */	push edx
/* 000014A0 0380  E8 00 00 00 00 */	call _foo
/* 000014A5 0385  8B 54 24 0C */	mov edx, dword ptr [esp + 0xc]
/* 000014A9 0389  52 */	push edx
/* 000014AA 038A  E8 00 00 00 00 */	call _foo
/* 000014AF 038F  83 C4 0C */	add esp, 0xc
/* 000014B2 0392  C3 */	ret
/* 000014B3 0393  90 */	nop
/* 000014B4 0394  90 */	nop
/* 000014B5 0395  90 */	nop
/* 000014B6 0396  90 */	nop
/* 000014B7 0397  90 */	nop
/* 000014B8 0398  90 */	nop
/* 000014B9 0399  90 */	nop
/* 000014BA 039A  90 */	nop
/* 000014BB 039B  90 */	nop
/* 000014BC 039C  90 */	nop
/* 000014BD 039D  90 */	nop
/* 000014BE 039E  90 */	nop
/* 000014BF 039F  90 */	nop

test:
/* 000014C0 0000  56 */	push esi
/* 000014C1 0001  8B 74 24 08 */	mov esi, dword ptr [esp + 8]
/* 000014C5 0005  56 */	push esi
/* 000014C6 0006  E8 00 00 00 00 */	call test_s8
/* 000014CB 000B  56 */	push esi
/* 000014CC 000C  E8 00 00 00 00 */	call test_s16
/* 000014D1 0011  56 */	push esi
/* 000014D2 0012  E8 00 00 00 00 */	call test_s32_div
/* 000014D7 0017  56 */	push esi
/* 000014D8 0018  E8 00 00 00 00 */	call test_s32_mod
/* 000014DD 001D  56 */	push esi
/* 000014DE 001E  E8 00 00 00 00 */	call test_u32_div
/* 000014E3 0023  56 */	push esi
/* 000014E4 0024  E8 00 00 00 00 */	call test_u32_mod
/* 000014E9 0029  83 C4 18 */	add esp, 0x18
/* 000014EC 002C  5E */	pop esi
/* 000014ED 002D  C3 */	ret
/* 000014EE 002E  90 */	nop
/* 000014EF 002F  90 */	nop

