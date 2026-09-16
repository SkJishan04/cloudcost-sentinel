"""Domain-specific exceptions and their HTTP mappings."""

from fastapi import Request, status
from fastapi.responses import JSONResponse


class FinOpsError(Exception):
    """Base class for all domain errors raised by the application."""

    status_code: int = status.HTTP_400_BAD_REQUEST

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class InstanceNotFoundError(FinOpsError):
    status_code = status.HTTP_404_NOT_FOUND


class InsufficientUsageDataError(FinOpsError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY


class GuardrailViolationError(FinOpsError):
    """Raised when an agent action would violate a safety guardrail."""

    status_code = status.HTTP_403_FORBIDDEN


class ForecastingError(FinOpsError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


async def finops_exception_handler(request: Request, exc: FinOpsError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.__class__.__name__, "message": exc.message},
    )
