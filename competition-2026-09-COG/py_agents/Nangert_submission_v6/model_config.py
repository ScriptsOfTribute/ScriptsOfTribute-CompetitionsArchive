from typing import Any

class SubmissionModelConfig:

    def __init__(self, **fields: Any) -> None:
        for name, value in fields.items():
            setattr(self, name, value)

    def __repr__(self) -> str:
        inner = ', '.join((f'{k}={v!r}' for k, v in sorted(vars(self).items())))
        return f'SubmissionModelConfig({inner})'
