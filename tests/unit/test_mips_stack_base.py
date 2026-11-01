import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from m2c.main import parse_flags, run


class TestMipsStackBase(unittest.TestCase):
    def decompile(self, assembly: str, context: str, target: str) -> str:
        with tempfile.TemporaryDirectory() as directory:
            asm_path = Path(directory) / "test.s"
            ctx_path = Path(directory) / "context.c"
            asm_path.write_text(assembly)
            ctx_path.write_text(context)
            options = parse_flags(
                [
                    "-t",
                    target,
                    "--valid-syntax",
                    "--no-cache",
                    "--context",
                    str(ctx_path),
                    str(asm_path),
                ]
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = run(options)
            self.assertEqual(status, 0, output.getvalue())
            return output.getvalue()

    def test_stack_base_address(self) -> None:
        # All these instructions copy the address of the local at offset zero.
        copies = [
            "move $a0, {base}",
            "addu $a0, {base}, $zero",
            "daddu $a0, {base}, $zero",
            "or $a0, {base}, $zero",
            "addi $a0, {base}, 0",
            "addiu $a0, {base}, 0",
            "daddi $a0, {base}, 0",
            "daddiu $a0, {base}, 0",
            "paddub $a0, {base}, $zero",
        ]
        types = [
            ("unsigned short", "s32", "u16", "lhu $v0, 0($sp)", "(s32) sp0"),
            ("unsigned int", "u32", "u32", "lw $v0, 0($sp)", "sp0"),
            ("float", "f32", "f32", "lwc1 $f0, 0($sp)", "sp0"),
            # No direct stack load: taking its address must bound the local region.
            ("unsigned int", "void", "u32", "", ""),
        ]
        for c_type, result_type, local_type, load, result in types:
            for base in ["$sp", "$fp"]:
                for copy in copies:
                    with self.subTest(type=c_type, base=base, copy=copy):
                        setup = ""
                        restore = ""
                        if base == "$fp":
                            setup = "sd $fp, 0x10($sp)\nmove $fp, $sp"
                            restore = "ld $fp, 0x10($sp)"
                        assembly = f"""glabel test
addiu $sp, $sp, -0x30
sd $ra, 0x20($sp)
{setup}
move $t9, $a0
jalr $t9
 {copy.format(base=base)}
{load}
{restore}
ld $ra, 0x20($sp)
jr $ra
 addiu $sp, $sp, 0x30
"""
                        return_type = (
                            "void"
                            if result_type == "void"
                            else "int" if c_type == "unsigned short" else c_type
                        )
                        context = f"{return_type} test(void (*writer)({c_type} *));"
                        output = self.decompile(assembly, context, "mipsee-gcc-c")
                        self.assertEqual(
                            output,
                            f"{result_type} test(void (*writer)({local_type} *)) {{\n"
                            f"    {local_type} sp0;\n\n"
                            f"    writer(&sp0);\n"
                            + (f"    return {result};\n" if result else "")
                            + "}\n",
                        )

    def test_home_space_and_stack_argument(self) -> None:
        # Ordinary moves must preserve each ABI's outgoing argument area.
        cases = [
            ("mips-gcc-c", "sw", "lw", 0x20, 0x1C, 0x18, 0x10, 4),
            ("mipsee-gcc-c", "sd", "ld", 0x30, 0x20, 0x10, 0, 8),
        ]
        for target, save, restore, size, ra, local, outgoing, arg_count in cases:
            with self.subTest(target=target):
                register_args = "\n".join(
                    f"li $a{i - 1}, {i}" for i in range(3, arg_count + 1)
                )
                assembly = f"""glabel test
addiu $sp, $sp, -{size}
{save} $ra, {ra}($sp)
move $a1, $a0
li $v0, {arg_count + 1}
sw $v0, {outgoing}($sp)
{register_args}
jal writer
 addiu $a0, $sp, {local}
lw $v0, {local}($sp)
{restore} $ra, {ra}($sp)
jr $ra
 addiu $sp, $sp, {size}
"""
                arg_types = ", int" * arg_count
                output = self.decompile(
                    assembly,
                    "unsigned int test(int arg); "
                    f"void writer(unsigned int *{arg_types});",
                    target,
                )
                arg_values = ", ".join(str(i) for i in range(3, arg_count + 2))
                name = f"sp{local:X}"
                self.assertEqual(
                    output,
                    "u32 test(s32 arg) {\n"
                    f"    u32 {name};\n\n"
                    f"    writer(&{name}, arg, {arg_values});\n"
                    f"    return {name};\n}}\n",
                )
