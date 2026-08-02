u8 _sink(s32);                                      /* extern */

void test(void) {
    s32 var_r1;
    u8 temp_r0;

    temp_r0 = sink();
    switch (temp_r0) {                              /* irregular */
    case 0x0:
        var_r1 = 0x83;
block_247:
        sink(temp_r0 + var_r1);
        return;
    case 0x1:
        var_r1 = 0x184;
        goto block_247;
    case 0x2:
        var_r1 = 0x285;
        goto block_247;
    case 0x3:
        var_r1 = 0x386;
        goto block_247;
    case 0x4:
        var_r1 = 0x487;
        goto block_247;
    case 0x5:
        var_r1 = 0x588;
        goto block_247;
    case 0x6:
        var_r1 = 0x689;
        goto block_247;
    case 0x7:
        var_r1 = 0x78A;
        goto block_247;
    case 0x8:
        var_r1 = 0x88B;
        goto block_247;
    case 0x9:
        var_r1 = 0x98C;
        goto block_247;
    case 0xA:
        var_r1 = 0xA8D;
        goto block_247;
    case 0xB:
        var_r1 = 0xB8E;
        goto block_247;
    case 0xC:
        var_r1 = 0xC8F;
        goto block_247;
    case 0xD:
        var_r1 = 0xD90;
        goto block_247;
    case 0xE:
        var_r1 = 0xE91;
        goto block_247;
    case 0xF:
        var_r1 = 0xF92;
        goto block_247;
    case 0x10:
        var_r1 = 0x1093;
        goto block_247;
    case 0x11:
        var_r1 = 0x1194;
        goto block_247;
    case 0x12:
        var_r1 = 0x1295;
        goto block_247;
    case 0x13:
        var_r1 = 0x1396;
        goto block_247;
    case 0x14:
        var_r1 = 0x1497;
        goto block_247;
    case 0x15:
        var_r1 = 0x1598;
        goto block_247;
    case 0x16:
        var_r1 = 0x1699;
        goto block_247;
    case 0x17:
        var_r1 = 0x179A;
        goto block_247;
    case 0x18:
        var_r1 = 0x189B;
        goto block_247;
    case 0x19:
        var_r1 = 0x199C;
        goto block_247;
    case 0x1A:
        var_r1 = 0x1A9D;
        goto block_247;
    case 0x1B:
        var_r1 = 0x1B9E;
        goto block_247;
    case 0x1C:
        var_r1 = 0x1C9F;
        goto block_247;
    case 0x1D:
        var_r1 = 0x1DA0;
        goto block_247;
    case 0x1E:
        var_r1 = 0x1EA1;
        goto block_247;
    case 0x1F:
        var_r1 = 0x1FA2;
        goto block_247;
    case 0x20:
        var_r1 = 0x20A3;
        goto block_247;
    case 0x21:
        var_r1 = 0x21A4;
        goto block_247;
    case 0x22:
        var_r1 = 0x22A5;
        goto block_247;
    case 0x23:
        var_r1 = 0x23A6;
        goto block_247;
    case 0x24:
        var_r1 = 0x24A7;
        goto block_247;
    case 0x25:
        var_r1 = 0x25A8;
        goto block_247;
    case 0x26:
        var_r1 = 0x26A9;
        goto block_247;
    case 0x27:
        var_r1 = 0x27AA;
        goto block_247;
    case 0x28:
        var_r1 = 0x28AB;
        goto block_247;
    case 0x29:
        var_r1 = 0x29AC;
        goto block_247;
    case 0x2A:
        var_r1 = 0x2AAD;
        goto block_247;
    case 0x2B:
        var_r1 = 0x2BAE;
        goto block_247;
    case 0x2C:
        var_r1 = 0x2CAF;
        goto block_247;
    case 0x2D:
        var_r1 = 0x2DB0;
        goto block_247;
    case 0x2E:
        var_r1 = 0x2EB1;
        goto block_247;
    case 0x2F:
        var_r1 = 0x2FB2;
        goto block_247;
    case 0x30:
        var_r1 = 0x30B3;
        goto block_247;
    case 0x31:
        var_r1 = 0x31B4;
        goto block_247;
    case 0x32:
        var_r1 = 0x32B5;
        goto block_247;
    case 0x33:
        var_r1 = 0x33B6;
        goto block_247;
    case 0x34:
        var_r1 = 0x34B7;
        goto block_247;
    case 0x35:
        var_r1 = 0x35B8;
        goto block_247;
    case 0x36:
        var_r1 = 0x36B9;
        goto block_247;
    case 0x37:
        var_r1 = 0x37BA;
        goto block_247;
    case 0x38:
        var_r1 = 0x38BB;
        goto block_247;
    case 0x39:
        var_r1 = 0x39BC;
        goto block_247;
    case 0x3A:
        var_r1 = 0x3ABD;
        goto block_247;
    case 0x3B:
        var_r1 = 0x3BBE;
        goto block_247;
    case 0x3C:
        var_r1 = 0x3CBF;
        goto block_247;
    case 0x3D:
        var_r1 = 0x3DC0;
        goto block_247;
    case 0x3E:
        var_r1 = 0x3EC1;
        goto block_247;
    case 0x3F:
        var_r1 = 0x3FC2;
        goto block_247;
    case 0x40:
        var_r1 = 0x40C3;
        goto block_247;
    case 0x41:
        var_r1 = 0x41C4;
        goto block_247;
    case 0x42:
        var_r1 = 0x42C5;
        goto block_247;
    case 0x43:
        var_r1 = 0x43C6;
        goto block_247;
    case 0x44:
        var_r1 = 0x44C7;
        goto block_247;
    case 0x45:
        var_r1 = 0x45C8;
        goto block_247;
    case 0x46:
        var_r1 = 0x46C9;
        goto block_247;
    case 0x47:
        var_r1 = 0x47CA;
        goto block_247;
    case 0x48:
        var_r1 = 0x48CB;
        goto block_247;
    case 0x49:
        var_r1 = 0x49CC;
        goto block_247;
    case 0x4A:
        var_r1 = 0x4ACD;
        goto block_247;
    case 0x4B:
        var_r1 = 0x4BCE;
        goto block_247;
    case 0x4C:
        var_r1 = 0x4CCF;
        goto block_247;
    case 0x4D:
        var_r1 = 0x4DD0;
        goto block_247;
    case 0x4E:
        var_r1 = 0x4ED1;
        goto block_247;
    case 0x4F:
        var_r1 = 0x4FD2;
        goto block_247;
    case 0x50:
        var_r1 = 0x50D3;
        goto block_247;
    case 0x51:
        var_r1 = 0x51D4;
        goto block_247;
    case 0x52:
        var_r1 = 0x52D5;
        goto block_247;
    case 0x53:
        var_r1 = 0x53D6;
        goto block_247;
    case 0x54:
        var_r1 = 0x54D7;
        goto block_247;
    case 0x55:
        var_r1 = 0x55D8;
        goto block_247;
    case 0x56:
        var_r1 = 0x56D9;
        goto block_247;
    case 0x57:
        var_r1 = 0x57DA;
        goto block_247;
    case 0x58:
        var_r1 = 0x58DB;
        goto block_247;
    case 0x59:
        var_r1 = 0x59DC;
        goto block_247;
    case 0x5A:
        var_r1 = 0x5ADD;
        goto block_247;
    case 0x5B:
        var_r1 = 0x5BDE;
        goto block_247;
    case 0x5C:
        var_r1 = 0x5CDF;
        goto block_247;
    case 0x5D:
        var_r1 = 0x5DE0;
        goto block_247;
    case 0x5E:
        var_r1 = 0x5EE1;
        goto block_247;
    case 0x5F:
        var_r1 = 0x5FE2;
        goto block_247;
    case 0x60:
        var_r1 = 0x60E3;
        goto block_247;
    case 0x61:
        var_r1 = 0x61E4;
        goto block_247;
    case 0x62:
        var_r1 = 0x62E5;
        goto block_247;
    case 0x63:
        var_r1 = 0x63E6;
        goto block_247;
    case 0x64:
        var_r1 = 0x64E7;
        goto block_247;
    case 0x65:
        var_r1 = 0x65E8;
        goto block_247;
    case 0x66:
        var_r1 = 0x66E9;
        goto block_247;
    case 0x67:
        var_r1 = 0x67EA;
        goto block_247;
    case 0x68:
        var_r1 = 0x68EB;
        goto block_247;
    case 0x69:
        var_r1 = 0x69EC;
        goto block_247;
    case 0x6A:
        var_r1 = 0x6AED;
        goto block_247;
    case 0x6B:
        var_r1 = 0x6BEE;
        goto block_247;
    case 0x6C:
        var_r1 = 0x6CEF;
        goto block_247;
    case 0x6D:
        var_r1 = 0x6DF0;
        goto block_247;
    case 0x6E:
        var_r1 = 0x6EF1;
        goto block_247;
    case 0x6F:
        var_r1 = 0x6FF2;
        goto block_247;
    case 0x70:
        var_r1 = 0x70F3;
        goto block_247;
    case 0x71:
        var_r1 = 0x71F4;
        goto block_247;
    case 0x72:
        var_r1 = 0x72F5;
        goto block_247;
    case 0x73:
        var_r1 = 0x73F6;
        goto block_247;
    case 0x74:
        var_r1 = 0x74F7;
        goto block_247;
    case 0x75:
        var_r1 = 0x75F8;
        goto block_247;
    case 0x76:
        var_r1 = 0x76F9;
        goto block_247;
    case 0x77:
        var_r1 = 0x77FA;
        goto block_247;
    case 0x78:
        var_r1 = 0x78FB;
        goto block_247;
    case 0x79:
        var_r1 = 0x79FC;
        goto block_247;
    case 0x7A:
        var_r1 = 0x7AFD;
        goto block_247;
    case 0x7B:
        var_r1 = 0x7BFE;
        goto block_247;
    case 0x7C:
        var_r1 = 0x7CFF;
        goto block_247;
    case 0x7D:
        var_r1 = 0x7E00;
        goto block_247;
    case 0x7E:
        var_r1 = 0x7F01;
        goto block_247;
    case 0x7F:
        var_r1 = 0x8002;
        goto block_247;
    case 0x80:
        var_r1 = 0x8103;
        goto block_247;
    case 0x81:
        var_r1 = 0x8204;
        goto block_247;
    case 0x82:
        var_r1 = 0x8305;
        goto block_247;
    case 0x83:
        var_r1 = 0x8406;
        goto block_247;
    case 0x84:
        var_r1 = 0x8507;
        goto block_247;
    case 0x85:
        var_r1 = 0x8608;
        goto block_247;
    case 0x86:
        var_r1 = 0x8709;
        goto block_247;
    case 0x87:
        var_r1 = 0x880A;
        goto block_247;
    case 0x88:
        var_r1 = 0x890B;
        goto block_247;
    case 0x89:
        var_r1 = 0x8A0C;
        goto block_247;
    case 0x8A:
        var_r1 = 0x8B0D;
        goto block_247;
    case 0x8B:
        var_r1 = 0x8C0E;
        goto block_247;
    case 0x8C:
        var_r1 = 0x8D0F;
        goto block_247;
    case 0x8D:
        var_r1 = 0x8E10;
        goto block_247;
    case 0x8E:
        var_r1 = 0x8F11;
        goto block_247;
    case 0x8F:
        var_r1 = 0x9012;
        goto block_247;
    case 0x90:
        var_r1 = 0x9113;
        goto block_247;
    case 0x91:
        var_r1 = 0x9214;
        goto block_247;
    case 0x92:
        var_r1 = 0x9315;
        goto block_247;
    case 0x93:
        var_r1 = 0x9416;
        goto block_247;
    case 0x94:
        var_r1 = 0x9517;
        goto block_247;
    case 0x95:
        var_r1 = 0x9618;
        goto block_247;
    case 0x96:
        var_r1 = 0x9719;
        goto block_247;
    case 0x97:
        var_r1 = 0x981A;
        goto block_247;
    case 0x98:
        var_r1 = 0x991B;
        goto block_247;
    case 0x99:
        var_r1 = 0x9A1C;
        goto block_247;
    case 0x9A:
        var_r1 = 0x9B1D;
        goto block_247;
    case 0x9B:
        var_r1 = 0x9C1E;
        goto block_247;
    case 0x9C:
        var_r1 = 0x9D1F;
        goto block_247;
    case 0x9D:
        var_r1 = 0x9E20;
        goto block_247;
    case 0x9E:
        var_r1 = 0x9F21;
        goto block_247;
    case 0x9F:
        var_r1 = 0xA022;
        goto block_247;
    case 0xA0:
        var_r1 = 0xA123;
        goto block_247;
    case 0xA1:
        var_r1 = 0xA224;
        goto block_247;
    case 0xA2:
        var_r1 = 0xA325;
        goto block_247;
    case 0xA3:
        var_r1 = 0xA426;
        goto block_247;
    case 0xA4:
        var_r1 = 0xA527;
        goto block_247;
    case 0xA5:
        var_r1 = 0xA628;
        goto block_247;
    case 0xA6:
        var_r1 = 0xA729;
        goto block_247;
    case 0xA7:
        var_r1 = 0xA82A;
        goto block_247;
    case 0xA8:
        var_r1 = 0xA92B;
        goto block_247;
    case 0xA9:
        var_r1 = 0xAA2C;
        goto block_247;
    case 0xAA:
        var_r1 = 0xAB2D;
        goto block_247;
    case 0xAB:
        var_r1 = 0xAC2E;
        goto block_247;
    case 0xAC:
        var_r1 = 0xAD2F;
        goto block_247;
    case 0xAD:
        var_r1 = 0xAE30;
        goto block_247;
    case 0xAE:
        var_r1 = 0xAF31;
        goto block_247;
    case 0xAF:
        var_r1 = 0xB032;
        goto block_247;
    case 0xB0:
        var_r1 = 0xB133;
        goto block_247;
    case 0xB1:
        var_r1 = 0xB234;
        goto block_247;
    case 0xB2:
        var_r1 = 0xB335;
        goto block_247;
    case 0xB3:
        var_r1 = 0xB436;
        goto block_247;
    case 0xB4:
        var_r1 = 0xB537;
        goto block_247;
    case 0xB5:
        var_r1 = 0xB638;
        goto block_247;
    case 0xB6:
        var_r1 = 0xB739;
        goto block_247;
    case 0xB7:
        var_r1 = 0xB83A;
        goto block_247;
    case 0xB8:
        var_r1 = 0xB93B;
        goto block_247;
    case 0xB9:
        var_r1 = 0xBA3C;
        goto block_247;
    case 0xBA:
        var_r1 = 0xBB3D;
        goto block_247;
    case 0xBB:
        var_r1 = 0xBC3E;
        goto block_247;
    case 0xBC:
        var_r1 = 0xBD3F;
        goto block_247;
    case 0xBD:
        var_r1 = 0xBE40;
        goto block_247;
    case 0xBE:
        var_r1 = 0xBF41;
        goto block_247;
    case 0xBF:
        var_r1 = 0xC042;
        goto block_247;
    case 0xC0:
        var_r1 = 0xC143;
        goto block_247;
    case 0xC1:
        var_r1 = 0xC244;
        goto block_247;
    case 0xC2:
        var_r1 = 0xC345;
        goto block_247;
    case 0xC3:
        var_r1 = 0xC446;
        goto block_247;
    case 0xC4:
        var_r1 = 0xC547;
        goto block_247;
    case 0xC5:
        var_r1 = 0xC648;
        goto block_247;
    case 0xC6:
        var_r1 = 0xC749;
        goto block_247;
    case 0xC7:
        var_r1 = 0xC84A;
        goto block_247;
    case 0xC8:
        var_r1 = 0xC94B;
        goto block_247;
    case 0xC9:
        var_r1 = 0xCA4C;
        goto block_247;
    case 0xCA:
        var_r1 = 0xCB4D;
        goto block_247;
    case 0xCB:
        var_r1 = 0xCC4E;
        goto block_247;
    case 0xCC:
        var_r1 = 0xCD4F;
        goto block_247;
    case 0xCD:
        var_r1 = 0xCE50;
        goto block_247;
    case 0xCE:
        var_r1 = 0xCF51;
        goto block_247;
    case 0xCF:
        var_r1 = 0xD052;
        goto block_247;
    case 0xD0:
        var_r1 = 0xD153;
        goto block_247;
    case 0xD1:
        var_r1 = 0xD254;
        goto block_247;
    case 0xD2:
        var_r1 = 0xD355;
        goto block_247;
    case 0xD3:
        var_r1 = 0xD456;
        goto block_247;
    case 0xD4:
        var_r1 = 0xD557;
        goto block_247;
    case 0xD5:
        var_r1 = 0xD658;
        goto block_247;
    case 0xD6:
        var_r1 = 0xD759;
        goto block_247;
    case 0xD7:
        var_r1 = 0xD85A;
        goto block_247;
    case 0xD8:
        var_r1 = 0xD95B;
        goto block_247;
    case 0xD9:
        var_r1 = 0xDA5C;
        goto block_247;
    case 0xDA:
        var_r1 = 0xDB5D;
        goto block_247;
    case 0xDB:
        var_r1 = 0xDC5E;
        goto block_247;
    case 0xDC:
        var_r1 = 0xDD5F;
        goto block_247;
    case 0xDD:
        var_r1 = 0xDE60;
        goto block_247;
    case 0xDE:
        var_r1 = 0xDF61;
        goto block_247;
    case 0xDF:
        var_r1 = 0xE062;
        goto block_247;
    case 0xE0:
        var_r1 = 0xE163;
        goto block_247;
    case 0xE1:
        var_r1 = 0xE264;
        goto block_247;
    case 0xE2:
        var_r1 = 0xE365;
        goto block_247;
    case 0xE3:
        var_r1 = 0xE466;
        goto block_247;
    case 0xE4:
        var_r1 = 0xE567;
        goto block_247;
    case 0xE5:
        var_r1 = 0xE668;
        goto block_247;
    case 0xE6:
        var_r1 = 0xE769;
        goto block_247;
    case 0xE7:
        var_r1 = 0xE86A;
        goto block_247;
    case 0xE8:
        var_r1 = 0xE96B;
        goto block_247;
    case 0xE9:
        var_r1 = 0xEA6C;
        goto block_247;
    case 0xEA:
        var_r1 = 0xEB6D;
        goto block_247;
    case 0xEB:
        var_r1 = 0xEC6E;
        goto block_247;
    case 0xEC:
        var_r1 = 0xED6F;
        goto block_247;
    case 0xED:
        var_r1 = 0xEE70;
        goto block_247;
    case 0xEE:
        var_r1 = 0xEF71;
        goto block_247;
    case 0xEF:
        var_r1 = 0xF072;
        goto block_247;
    case 0xF0:
        var_r1 = 0xF173;
        goto block_247;
    case 0xF1:
        var_r1 = 0xF274;
        goto block_247;
    case 0xF2:
        var_r1 = 0xF375;
        goto block_247;
    case 0xF3:
        var_r1 = 0xF476;
        goto block_247;
    default:
        sink(temp_r0 - 1);
        return;
    }
}
