import attr
import re

import typing
from typing import List, Union, Iterator, Optional, Dict, Callable, Any

from parse_instruction import *

@attr.s(frozen=True)
class Label:
    name: str = attr.ib()

    def __str__(self):
        return f'  .{self.name}:'

@attr.s
class Function:
    name: str = attr.ib()
    body: List[Union[Instruction, Label]] = attr.ib(factory=list)

    def new_label(self, name: str):
        self.body.append(Label(name))

    def new_instruction(self, instruction):
        self.body.append(instruction)

    def __str__(self):
        body = "\n".join(str(item) for item in self.body)
        return f'glabel {self.name}\n{body}'

@attr.s
class MIPSFile:
    filename: str = attr.ib()
    functions: List[Function] = attr.ib(factory=list)
    current_function: Optional[Function] = attr.ib(default=None, repr=False)

    def new_function(self, name: str):
        self.current_function = Function(name=name)
        self.functions.append(self.current_function)

    def new_instruction(self, instruction: Instruction):
        assert self.current_function is not None
        self.current_function.new_instruction(instruction)

    def new_label(self, label_name):
        self.current_function.new_label(label_name)

    def __str__(self):
        functions_str = '\n\n'.join(str(function) for function in self.functions)
        return f'# {self.filename}\n{functions_str}'


def parse_file(filename: str, f: typing.TextIO) -> MIPSFile:
    mips_file: MIPSFile = MIPSFile(filename)

    for line in f:
        # Strip comments and whitespace
        line = re.sub(r'/\*.*\*/', '', line)
        line = re.sub(r'#.*$', '', line)
        line = line.strip()

        if line == '':
            continue
        elif line.startswith('.') and line.endswith(':'):
            # Label.
            label_name: str = line.strip('.:')
            mips_file.new_label(label_name)
        elif line.startswith('.'):
            # Assembler directive.
            pass
        elif line.startswith('glabel'):
            # Function label.
            function_name: str = line.split(' ')[1]
            mips_file.new_function(function_name)
        else:
            # Instruction.
            instruction: Instruction = parse_instruction(line)
            mips_file.new_instruction(instruction)

    return mips_file
