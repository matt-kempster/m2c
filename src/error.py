from dataclasses import dataclass


@dataclass
class DecompFailure(Exception):
    message: str

    def __str__(self) -> str:
        return self.message
