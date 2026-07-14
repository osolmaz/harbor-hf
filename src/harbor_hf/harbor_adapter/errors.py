class WorkerError(RuntimeError):
    """Raised when a remote benchmark run cannot complete correctly."""


class HarborTrialFailure(WorkerError):
    """A Harbor result reported a typed trial or step exception."""

    def __init__(self, message: str, exception_type: str) -> None:
        super().__init__(message)
        self.exception_type = exception_type


class HarborVerificationFailure(WorkerError):
    """Harbor evidence does not satisfy the immutable benchmark contract."""
