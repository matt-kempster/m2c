#!/usr/bin/env python
#
# CodeWarrior demangler, supporting operators, templates (including literals), and arrays.
#
# This can be included as a library, or used as a CLI application directly:
#
#   $ ./demangle_cw.py 'check__3FooCFUlR3Bar'
#   Foo::check (long unsigned, Bar &) const
#
# Based on `demangler.cs` by arookas, which was provided under the 3-clause BSD License below:
#
#   https://gist.github.com/arookas/0c0cd842c9ca836d7a9ed06a655b1a22
#
# Copyright (c) 2017, arookas
# All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   1. Redistributions of source code must retain the above copyright notice, this
#      list of conditions and the following disclaimer.
#
#   2. Redistributions in binary form must reproduce the above copyright notice,
#      this list of conditions and the following disclaimer in the documentation
#      and/or other materials provided with the distribution.
#
#   3. Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from io import StringIO, TextIOBase
from typing import Any, ClassVar, Iterator, List, Optional, Set, Tuple

__all__ = [
    "CxxName",
    "CxxQualifiedName",
    "CxxSymbol",
    "CxxTerm",
    "CxxType",
    "demangle",
    "parse",
]


# These substitutions are performed by the doldisasm.py decomp tool, and not by
# the CodeWarrior compiler directly. Undo them for convience, even though they
# aren't technically part of CW's mangling scheme.
DOLDISASM_SUBSTITUTIONS = [
    ("<", "$$0"),
    (">", "$$1"),
    ("@", "$$2"),
    ("\\", "$$3"),
    (",", "$$4"),
    ("-", "$$5"),
]


@contextmanager
def peeking(src: TextIOBase) -> Iterator[None]:
    # Store the current offset in `src`, and restore it at the end of the context
    ptr = src.tell()
    try:
        yield
    finally:
        src.seek(ptr)


def peek(src: TextIOBase, n: int = 1) -> str:
    # Read `n` bytes from `src` without advancing the offset
    with peeking(src):
        return src.read(n)


@contextmanager
def as_stringio(src: str) -> Iterator[StringIO]:
    # Wrap `src` in a `StringIO`, and assert it was fully consumed at the end of the context
    buf = StringIO(src)
    yield buf
    leftover = buf.read()
    if leftover:
        raise ValueError(f"Unable to parse full input, leftover chars: {leftover!r}")


@dataclass
class CxxName:
    NUMBER_CHARS: ClassVar[Set[str]] = set("-0123456789")

    name: str
    template_params: Optional[List["CxxType"]] = None

    @staticmethod
    def parse(src: TextIOBase) -> "CxxName":
        # Numbers: either a template literal, or a length prefix
        number_str = ""
        while peek(src) in CxxName.NUMBER_CHARS:
            number_str += src.read(1)
        if number_str == "":
            raise ValueError(
                "Unable to parse CxxName, input did not start with a number"
            )
        number = int(number_str)

        # Check if the number represents a template literal
        if peek(src) in (",", ">"):
            return CxxName(number_str)

        # Otherwise, it is the length of the name
        assert number > 0, ValueError
        name = src.read(number)

        # Simple case: plain identifier
        if "<" not in name:
            return CxxName(name)

        # Otherwise, parse the template parameters between the `<...>` brackets
        assert name[-1] == ">", ValueError
        base_name, sep, param_strs = name.partition("<")

        with as_stringio(param_strs) as buf:
            template_params = []
            while True:
                template_params.append(CxxType.parse(buf))
                sep = buf.read(1)
                if sep == ">":
                    break
                if sep == ",":
                    continue
                raise ValueError(f"Unexpected character while parsing CxxName: {sep}")

        return CxxName(name=base_name, template_params=template_params)

    def __str__(self) -> str:
        if self.template_params is None:
            return self.name
        return f"{self.name}<{', '.join(str(p) for p in self.template_params)}>"


CxxQualifiedName = List["CxxName"]


@dataclass
class CxxTerm:
    class Kind(Enum):
        CONST = "C"
        POINTER = "P"
        REFERENCE = "R"
        SIGNED = "S"
        UNSIGNED = "U"
        ELLIPSIS = "e"
        VOID = "v"
        BOOL = "b"
        CHAR = "c"
        SHORT = "s"
        INT = "i"
        LONG = "l"
        LONG_LONG = "x"
        WIDE_CHAR = "w"
        FLOAT = "f"
        DOUBLE = "d"
        LONG_DOUBLE = "r"

        ARRAY = "A"
        QUALIFIED = "Q"
        FUNCTION = "F"
        SYMBOL_REFERENCE = "&"

    NONTERMINATING_KINDS = {
        Kind.CONST,
        Kind.POINTER,
        Kind.REFERENCE,
        Kind.SIGNED,
        Kind.UNSIGNED,
        Kind.ARRAY,
    }

    OPS = {
        "nw": " new",
        "nwa": " new[]",
        "dl": " delete",
        "dla": " delete[]",
        "pl": "+",
        "mi": "-",
        "ml": "*",
        "dv": "/",
        "md": "%",
        "er": "^",
        "adv": "/=",
        "ad": "&",
        "or": "|",
        "co": "~",
        "nt": "!",
        "as": "=",
        "lt": "<",
        "gt": ">",
        "apl": "+=",
        "ami": "-=",
        "amu": "*=",
        "amd": "%=",
        "aer": "^=",
        "aad": "&=",
        "aor": "|=",
        "ls": "<<",
        "rs": ">>",
        "ars": ">>=",
        "als": "<<=",
        "eq": "==",
        "ne": "!=",
        "le": "<=",
        "ge": ">=",
        "aa": "&&",
        "oo": "||",
        "pp": "++",
        "mm": "--",
        "cl": "()",
        "vc": "[]",
        "rf": "->",
        "cm": ",",
        "rm": "->*",
    }

    kind: Kind

    # The following fields are only populated for the corresponding `kind`s
    array_dim: Optional[int] = None
    function_params: Optional[List["CxxType"]] = None
    function_return: Optional["CxxType"] = None
    qualified_name: Optional[CxxQualifiedName] = None
    symbol_reference: Optional["CxxSymbol"] = None

    @staticmethod
    def parse(src: TextIOBase) -> "CxxTerm":
        if peek(src) in CxxName.NUMBER_CHARS:
            return CxxTerm(
                kind=CxxTerm.Kind.QUALIFIED, qualified_name=[CxxName.parse(src)]
            )

        kind = CxxTerm.Kind(src.read(1))
        if kind == CxxTerm.Kind.QUALIFIED:
            count = int(src.read(1))
            qualified_name = []
            for _ in range(count):
                qualified_name.append(CxxName.parse(src))
            return CxxTerm(kind=kind, qualified_name=qualified_name)

        if kind == CxxTerm.Kind.FUNCTION:
            function_params = []
            function_return = None
            while peek(src) not in ("", "_", ",", ">"):
                function_params.append(CxxType.parse(src))
            if peek(src, 1) == "_":
                src.read(1)
                function_return = CxxType.parse(src)
            return CxxTerm(
                kind=kind,
                function_params=function_params,
                function_return=function_return,
            )

        if kind == CxxTerm.Kind.ARRAY:
            array_dim = 0
            while True:
                c = src.read(1)
                if c == "_":
                    break
                array_dim = (array_dim * 10) + int(c)
            return CxxTerm(kind=kind, array_dim=array_dim)

        if kind == CxxTerm.Kind.SYMBOL_REFERENCE:
            return CxxTerm(
                kind=kind,
                symbol_reference=CxxSymbol.parse(src),
            )

        return CxxTerm(kind=kind)

    def __str__(self) -> str:
        if self.kind == CxxTerm.Kind.ARRAY:
            assert self.array_dim is not None
            return f"[{self.array_dim}]"

        if self.kind == CxxTerm.Kind.QUALIFIED:
            assert self.qualified_name is not None
            names = [str(q) for q in self.qualified_name]
            if names[-1] in CxxTerm.OPS:
                names[-1] = "operator" + CxxTerm.OPS[names[-1]]
            elif names[-1] == "__ct":
                assert len(names) >= 2
                names[-1] = names[-2]
            elif names[-1] == "__dt":
                assert len(names) >= 2
                names[-1] = f"~{names[-2]}"
            return "::".join(names)

        if self.kind == CxxTerm.Kind.FUNCTION:
            assert self.function_params is not None
            prefix = ""
            if self.function_return is not None:
                prefix = f"{self.function_return} (*) "
            args = ", ".join(str(p) for p in self.function_params)
            return f"{prefix}({args})"

        if self.kind == CxxTerm.Kind.SYMBOL_REFERENCE:
            assert self.symbol_reference is not None
            return f"&{self.symbol_reference}"

        if self.kind == CxxTerm.Kind.POINTER:
            return "*"
        if self.kind == CxxTerm.Kind.REFERENCE:
            return "&"
        if self.kind == CxxTerm.Kind.LONG_LONG:
            return "long long"
        if self.kind == CxxTerm.Kind.LONG_DOUBLE:
            return "long double"
        if self.kind == CxxTerm.Kind.WIDE_CHAR:
            return "wchar_t"
        if self.kind == CxxTerm.Kind.ELLIPSIS:
            return "..."
        return self.kind.name.lower()


@dataclass
class CxxType:
    terms: List[CxxTerm] = field(default_factory=list)

    @staticmethod
    def parse(src: TextIOBase) -> "CxxType":
        terms = []
        while True:
            if peek(src) in (",", ">"):
                break
            term = CxxTerm.parse(src)
            terms.append(term)
            if term.kind not in CxxTerm.NONTERMINATING_KINDS:
                break
        return CxxType(terms=terms)

    def __str__(self) -> str:
        return " ".join(str(a) for a in self.terms[::-1]).strip()


@dataclass
class CxxSymbol:
    qualified_name: CxxQualifiedName
    type: CxxType

    @staticmethod
    def parse(src: TextIOBase) -> "CxxSymbol":
        with peeking(src):
            base_name = src.read()

        strip_underscores = False
        template_depth = 0
        with peeking(src):
            chars = ""
            while True:
                c = src.read(1)
                if not c:
                    break

                if c in (",", ">") and template_depth == 0:
                    break
                elif c == "<":
                    template_depth += 1
                elif c == ">":
                    template_depth -= 1

                chars += c
                if not (c == "_" and peek(src) == "_"):
                    continue
                strip_underscores = True
                lookahead = peek(src, 2)
                if len(lookahead) == 2:
                    if lookahead[1] not in "QFC0123456789":
                        continue
                base_name = chars[:-1]

        if strip_underscores:
            src.read(len(base_name) + 2)
        else:
            src.read(len(base_name))

        class_name: Optional[CxxType] = CxxType.parse(src)
        if peek(src) not in ("", ",", ">"):
            type = CxxType.parse(src)
        else:
            assert class_name is not None
            type = class_name
            class_name = None

        qualified_name: CxxQualifiedName = []
        if class_name is not None:
            assert len(class_name.terms) == 1
            assert class_name.terms[0].kind == CxxTerm.Kind.QUALIFIED
            assert class_name.terms[0].qualified_name is not None
            qualified_name.extend(class_name.terms[0].qualified_name)

        with as_stringio(str(len(base_name)) + base_name) as buf:
            qualified_name.append(CxxName.parse(buf))

        return CxxSymbol(qualified_name=qualified_name, type=type)

    def __str__(self) -> str:
        name = CxxTerm(CxxTerm.Kind.QUALIFIED, qualified_name=self.qualified_name)
        return f"{name} {self.type}"


def parse(mangled: str) -> CxxSymbol:
    for old, new in DOLDISASM_SUBSTITUTIONS:
        mangled = mangled.replace(new, old)
    with as_stringio(mangled) as buf:
        return CxxSymbol.parse(buf)


def demangle(mangled: str) -> str:
    try:
        return str(parse(mangled))
    except ValueError:
        return mangled


def main() -> None:
    import sys

    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <mangled_name>")
        sys.exit(1)

    print(demangle(sys.argv[1]))


if __name__ == "__main__":
    main()
