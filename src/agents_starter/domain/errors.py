class DomainError(Exception):
    """Base class for errors that inbound adapters translate into user-facing responses."""


class UnknownAgentError(DomainError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Unknown agent: {name!r}")
        self.name = name


class UnknownWorkflowError(DomainError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Unknown workflow: {name!r}")
        self.name = name


class MaxTurnsExceededError(DomainError):
    def __init__(self, agent: str, max_turns: int) -> None:
        super().__init__(f"Agent {agent!r} did not finish within {max_turns} turns")


class AgentRunError(DomainError):
    """A remote agent runtime reported a failed, cancelled or expired run."""


class ReadOnlyIndexError(DomainError):
    def __init__(self, index: str) -> None:
        super().__init__(f"Index {index!r} is owned by another system and is read-only")
