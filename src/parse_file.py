import re
import typing
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import attr

from options import Options
from parse_instruction import *


@attr.s(frozen=True)
class Label:
    name: str = attr.ib()

    def __str__(self) -> str:
        return f'  .{self.name}:'

@attr.s
class Function:
    name: str = attr.ib()
    body: List[Union[Instruction, Label]] = attr.ib(factory=list)
    jumptable_labels: List[Label] = attr.ib(factory=list)

    def new_label(self, name: str) -> None:
        self.body.append(Label(name))

    def new_jumptable_label(self, name: str) -> None:
        self.body.append(Label(name))
        self.jumptable_labels.append(Label(name))

    def new_instruction(self, instruction: Instruction) -> None:
        self.body.append(instruction)

    def __str__(self) -> str:
        body = "\n".join(str(item) for item in self.body)
        return f'glabel {self.name}\n{body}'

@attr.s
class MIPSFile:
    filename: str = attr.ib()
    functions: List[Function] = attr.ib(factory=list)
    current_function: Optional[Function] = attr.ib(default=None, repr=False)

    def new_function(self, name: str) -> None:
        self.current_function = Function(name=name)
        self.functions.append(self.current_function)

    def new_instruction(self, instruction: Instruction) -> None:
        assert self.current_function is not None
        self.current_function.new_instruction(instruction)

    def new_label(self, label_name: str) -> None:
        assert self.current_function is not None
        self.current_function.new_label(label_name)

    def new_jumptable_label(self, label_name: str) -> None:
        assert self.current_function is not None
        self.current_function.new_jumptable_label(label_name)

    def __str__(self) -> str:
        functions_str = '\n\n'.join(str(function) for function in self.functions)
        return f'# {self.filename}\n{functions_str}'


def parse_file(f: typing.TextIO, options: Options) -> MIPSFile:
    mips_file: MIPSFile = MIPSFile(options.filename)
    defines: Dict[str, int] = options.preproc_defines.copy()
    ifdef_level: int = 0
    ifdef_levels: List[int] = []

    for line in f:
        # Strip comments and whitespace
        line = re.sub(r'/\*.*?\*/', '', line)
        line = re.sub(r'#.*$', '', line)
        line = line.strip()

        if line == '':
            pass
        elif line.startswith('.') and not line.endswith(':'):
            # Assembler directive.
            if line.startswith('.ifdef') or line.startswith('.ifndef'):
                macro_name = line.split()[1]
                if macro_name not in defines:
                    defines[macro_name] = 0
                    print(f"Note: assuming {macro_name} is unset for .ifdef, "
                        f"pass -D{macro_name}/-U{macro_name} to set/unset explicitly.")
                level = defines[macro_name]
                if line.startswith('.ifdef'):
                    level = 1 - level
                ifdef_level += level
                ifdef_levels.append(level)
            elif line.startswith('.else'):
                level = ifdef_levels.pop()
                ifdef_level -= level
                level = 1 - level
                ifdef_level += level
                ifdef_levels.append(level)
            elif line.startswith('.endif'):
                ifdef_level -= ifdef_levels.pop()
        elif ifdef_level == 0:
            if line.startswith('.'):
                # Label.
                if ifdef_level == 0:
                    label_name: str = line.strip('.:')
                    mips_file.new_label(label_name)
            elif line.startswith('glabel'):
                # Function label.
                function_name: str = line.split(' ')[1]
                if re.match('L[0-9A-F]{8}', function_name):
                    mips_file.new_jumptable_label(function_name)
                else:
                    mips_file.new_function(function_name)
            else:
                # Instruction.
                instruction: Instruction = parse_instruction(line)
                mips_file.new_instruction(instruction)

    return mips_file
