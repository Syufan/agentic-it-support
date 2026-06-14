from dataclasses import dataclass

@dataclass(frozen=True)
class Allow:
    pass

@dataclass(frozen=True)
class Retry:
    correction: str

@dataclass(frozen=True)
class Terminate:
    message: str

@dataclass(frozen=True)
class Escalate:
    reason: str

@dataclass(frozen=True)
class Continue:
    pass
