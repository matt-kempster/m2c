from __future__ import annotations
from dataclasses import dataclass
from typing import NoReturn


@dataclass
class DecompFailure(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def assert_never(x: NoReturn) -> NoReturn:
    raise Exception(f"Unreachable: {repr(x)}")
