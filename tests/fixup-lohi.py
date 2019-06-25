#!/usr/bin/env python3
import sys


def replace_last(s, a, b):
    return b.join(s.rsplit(a, 1))


filename = sys.argv[1]
entry = sys.argv[2][2:].upper()
output = []
waiting_hi = {}
with open(filename) as fin:
    for line in fin:
        line = line.strip()
        if entry + " " in line:
            output.append("glabel test")
        index = len(output)
        if line.startswith("func"):
            line = "glabel " + line[:-1]
        elif "lui" in line:
            reg, imm_hex = line.split(" ")[-2::]
            imm = int(imm_hex, 0)
            if 0x10 <= imm < 0x1000:  # modestly large lui is probably a pointer
                waiting_hi[reg[:-1]] = (index, imm, imm_hex)
        else:
            lo_regs = [reg for reg in waiting_hi.keys() if reg in line]
            if lo_regs and "0x" in line:
                lo_reg = lo_regs[0]
                hi_ind, hi_imm, hi_imm_hex = waiting_hi[lo_reg]
                lo_imm_hex = line.split(" ")[-1].split("(")[0]
                lo_imm = int(lo_imm_hex, 0)
                if lo_imm >= 0x8000:
                    lo_imm -= 0x10000
                sym = "D_" + hex((hi_imm << 16) + lo_imm)[2:].upper()
                line = replace_last(line, lo_imm_hex, f"%lo({sym})")
                output[hi_ind] = replace_last(output[hi_ind], hi_imm_hex, f"%hi({sym})")
                del waiting_hi[lo_reg]
        output.append(line)

with open(filename, "w") as fout:
    for line in output:
        print(line, file=fout)
