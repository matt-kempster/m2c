import attr

@attr.s
class DecompFailure(Exception):
    message: str = attr.ib()

    def __str__(self) -> str:
        return self.message
