.section .text
test_0:
/* 00000000 0000  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000004 0004  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000008 0008  8B 10 */	mov edx, dword ptr [eax]
/* 0000000A 000A  89 11 */	mov dword ptr [ecx], edx
/* 0000000C 000C  8B 40 04 */	mov eax, dword ptr [eax + 4]
/* 0000000F 000F  89 41 04 */	mov dword ptr [ecx + 4], eax
/* 00000012 0012  C3 */	ret
/* 00000013 0013  90 */	nop
/* 00000014 0014  90 */	nop
/* 00000015 0015  90 */	nop
/* 00000016 0016  90 */	nop
/* 00000017 0017  90 */	nop
/* 00000018 0018  90 */	nop
/* 00000019 0019  90 */	nop
/* 0000001A 001A  90 */	nop
/* 0000001B 001B  90 */	nop
/* 0000001C 001C  90 */	nop
/* 0000001D 001D  90 */	nop
/* 0000001E 001E  90 */	nop
/* 0000001F 001F  90 */	nop

test_1:
/* 00000020 0000  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000024 0004  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000028 0008  8B 10 */	mov edx, dword ptr [eax]
/* 0000002A 000A  89 11 */	mov dword ptr [ecx], edx
/* 0000002C 000C  8B 50 04 */	mov edx, dword ptr [eax + 4]
/* 0000002F 000F  89 51 04 */	mov dword ptr [ecx + 4], edx
/* 00000032 0012  8A 40 08 */	mov al, byte ptr [eax + 8]
/* 00000035 0015  88 41 08 */	mov byte ptr [ecx + 8], al
/* 00000038 0018  C3 */	ret
/* 00000039 0019  90 */	nop
/* 0000003A 001A  90 */	nop
/* 0000003B 001B  90 */	nop
/* 0000003C 001C  90 */	nop
/* 0000003D 001D  90 */	nop
/* 0000003E 001E  90 */	nop
/* 0000003F 001F  90 */	nop

test_2:
/* 00000040 0000  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000044 0004  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000048 0008  8B 10 */	mov edx, dword ptr [eax]
/* 0000004A 000A  89 11 */	mov dword ptr [ecx], edx
/* 0000004C 000C  8B 50 04 */	mov edx, dword ptr [eax + 4]
/* 0000004F 000F  89 51 04 */	mov dword ptr [ecx + 4], edx
/* 00000052 0012  66 8B 40 08 */	mov ax, word ptr [eax + 8]
/* 00000056 0016  66 89 41 08 */	mov word ptr [ecx + 8], ax
/* 0000005A 001A  C3 */	ret
/* 0000005B 001B  90 */	nop
/* 0000005C 001C  90 */	nop
/* 0000005D 001D  90 */	nop
/* 0000005E 001E  90 */	nop
/* 0000005F 001F  90 */	nop

test_3:
/* 00000060 0000  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000064 0004  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000068 0008  8B 10 */	mov edx, dword ptr [eax]
/* 0000006A 000A  89 11 */	mov dword ptr [ecx], edx
/* 0000006C 000C  8B 50 04 */	mov edx, dword ptr [eax + 4]
/* 0000006F 000F  89 51 04 */	mov dword ptr [ecx + 4], edx
/* 00000072 0012  66 8B 50 08 */	mov dx, word ptr [eax + 8]
/* 00000076 0016  66 89 51 08 */	mov word ptr [ecx + 8], dx
/* 0000007A 001A  8A 40 0A */	mov al, byte ptr [eax + 0xa]
/* 0000007D 001D  88 41 0A */	mov byte ptr [ecx + 0xa], al
/* 00000080 0020  C3 */	ret
/* 00000081 0021  90 */	nop
/* 00000082 0022  90 */	nop
/* 00000083 0023  90 */	nop
/* 00000084 0024  90 */	nop
/* 00000085 0025  90 */	nop
/* 00000086 0026  90 */	nop
/* 00000087 0027  90 */	nop
/* 00000088 0028  90 */	nop
/* 00000089 0029  90 */	nop
/* 0000008A 002A  90 */	nop
/* 0000008B 002B  90 */	nop
/* 0000008C 002C  90 */	nop
/* 0000008D 002D  90 */	nop
/* 0000008E 002E  90 */	nop
/* 0000008F 002F  90 */	nop

test_4:
/* 00000090 0000  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000094 0004  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000098 0008  8B 10 */	mov edx, dword ptr [eax]
/* 0000009A 000A  89 11 */	mov dword ptr [ecx], edx
/* 0000009C 000C  8B 50 04 */	mov edx, dword ptr [eax + 4]
/* 0000009F 000F  89 51 04 */	mov dword ptr [ecx + 4], edx
/* 000000A2 0012  8B 40 08 */	mov eax, dword ptr [eax + 8]
/* 000000A5 0015  89 41 08 */	mov dword ptr [ecx + 8], eax
/* 000000A8 0018  C3 */	ret
/* 000000A9 0019  90 */	nop
/* 000000AA 001A  90 */	nop
/* 000000AB 001B  90 */	nop
/* 000000AC 001C  90 */	nop
/* 000000AD 001D  90 */	nop
/* 000000AE 001E  90 */	nop
/* 000000AF 001F  90 */	nop

test_5:
/* 000000B0 0000  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 000000B4 0004  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 000000B8 0008  8B 10 */	mov edx, dword ptr [eax]
/* 000000BA 000A  89 11 */	mov dword ptr [ecx], edx
/* 000000BC 000C  8B 50 04 */	mov edx, dword ptr [eax + 4]
/* 000000BF 000F  89 51 04 */	mov dword ptr [ecx + 4], edx
/* 000000C2 0012  8B 50 08 */	mov edx, dword ptr [eax + 8]
/* 000000C5 0015  89 51 08 */	mov dword ptr [ecx + 8], edx
/* 000000C8 0018  8A 40 0C */	mov al, byte ptr [eax + 0xc]
/* 000000CB 001B  88 41 0C */	mov byte ptr [ecx + 0xc], al
/* 000000CE 001E  C3 */	ret
/* 000000CF 001F  90 */	nop

test_6:
/* 000000D0 0000  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 000000D4 0004  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 000000D8 0008  8B 10 */	mov edx, dword ptr [eax]
/* 000000DA 000A  89 11 */	mov dword ptr [ecx], edx
/* 000000DC 000C  8B 50 04 */	mov edx, dword ptr [eax + 4]
/* 000000DF 000F  89 51 04 */	mov dword ptr [ecx + 4], edx
/* 000000E2 0012  8B 50 08 */	mov edx, dword ptr [eax + 8]
/* 000000E5 0015  89 51 08 */	mov dword ptr [ecx + 8], edx
/* 000000E8 0018  66 8B 40 0C */	mov ax, word ptr [eax + 0xc]
/* 000000EC 001C  66 89 41 0C */	mov word ptr [ecx + 0xc], ax
/* 000000F0 0020  C3 */	ret
/* 000000F1 0021  90 */	nop
/* 000000F2 0022  90 */	nop
/* 000000F3 0023  90 */	nop
/* 000000F4 0024  90 */	nop
/* 000000F5 0025  90 */	nop
/* 000000F6 0026  90 */	nop
/* 000000F7 0027  90 */	nop
/* 000000F8 0028  90 */	nop
/* 000000F9 0029  90 */	nop
/* 000000FA 002A  90 */	nop
/* 000000FB 002B  90 */	nop
/* 000000FC 002C  90 */	nop
/* 000000FD 002D  90 */	nop
/* 000000FE 002E  90 */	nop
/* 000000FF 002F  90 */	nop

test_7:
/* 00000100 0000  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000104 0004  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000108 0008  8B 10 */	mov edx, dword ptr [eax]
/* 0000010A 000A  89 11 */	mov dword ptr [ecx], edx
/* 0000010C 000C  8B 50 04 */	mov edx, dword ptr [eax + 4]
/* 0000010F 000F  89 51 04 */	mov dword ptr [ecx + 4], edx
/* 00000112 0012  8B 50 08 */	mov edx, dword ptr [eax + 8]
/* 00000115 0015  89 51 08 */	mov dword ptr [ecx + 8], edx
/* 00000118 0018  66 8B 50 0C */	mov dx, word ptr [eax + 0xc]
/* 0000011C 001C  66 89 51 0C */	mov word ptr [ecx + 0xc], dx
/* 00000120 0020  8A 40 0E */	mov al, byte ptr [eax + 0xe]
/* 00000123 0023  88 41 0E */	mov byte ptr [ecx + 0xe], al
/* 00000126 0026  8B 0D 00 00 00 00 */	mov ecx, dword ptr [_s7]
/* 0000012C 002C  8B 15 04 00 00 00 */	mov edx, dword ptr [_s7 + 0x4]
/* 00000132 0032  A1 08 00 00 00 */	mov eax, dword ptr [_s7 + 0x8]
/* 00000137 0037  89 0D 00 00 00 00 */	mov dword ptr [_d7], ecx
/* 0000013D 003D  66 8B 0D 0C 00 00 00 */	mov cx, word ptr [_s7 + 0xc]
/* 00000144 0044  89 15 04 00 00 00 */	mov dword ptr [_d7 + 0x4], edx
/* 0000014A 004A  8A 15 0E 00 00 00 */	mov dl, byte ptr [_s7 + 0xe]
/* 00000150 0050  A3 08 00 00 00 */	mov dword ptr [_d7 + 0x8], eax
/* 00000155 0055  66 89 0D 0C 00 00 00 */	mov word ptr [_d7 + 0xc], cx
/* 0000015C 005C  88 15 0E 00 00 00 */	mov byte ptr [_d7 + 0xe], dl
/* 00000162 0062  C3 */	ret
/* 00000163 0063  90 */	nop
/* 00000164 0064  90 */	nop
/* 00000165 0065  90 */	nop
/* 00000166 0066  90 */	nop
/* 00000167 0067  90 */	nop
/* 00000168 0068  90 */	nop
/* 00000169 0069  90 */	nop
/* 0000016A 006A  90 */	nop
/* 0000016B 006B  90 */	nop
/* 0000016C 006C  90 */	nop
/* 0000016D 006D  90 */	nop
/* 0000016E 006E  90 */	nop
/* 0000016F 006F  90 */	nop

test:
/* 00000170 0000  C3 */	ret
/* 00000171 0001  90 */	nop
/* 00000172 0002  90 */	nop
/* 00000173 0003  90 */	nop
/* 00000174 0004  90 */	nop
/* 00000175 0005  90 */	nop
/* 00000176 0006  90 */	nop
/* 00000177 0007  90 */	nop
/* 00000178 0008  90 */	nop
/* 00000179 0009  90 */	nop
/* 0000017A 000A  90 */	nop
/* 0000017B 000B  90 */	nop
/* 0000017C 000C  90 */	nop
/* 0000017D 000D  90 */	nop
/* 0000017E 000E  90 */	nop
/* 0000017F 000F  90 */	nop
