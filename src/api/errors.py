from fastapi import Request
from fastapi.responses import JSONResponse
from src.loader.base import FileValidationError


class EmptyQueryError(Exception):
    pass


class APITimeoutError(Exception):
    pass


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """全局异常处理器，统一返回格式"""
    status_code = 500
    error_type = "internal_error"
    message = str(exc) or "服务器内部错误"

    if isinstance(exc, FileValidationError):
        status_code = 400
        error_type = "invalid_file"
    elif isinstance(exc, EmptyQueryError):
        status_code = 400
        error_type = "empty_query"
    elif isinstance(exc, TimeoutError):
        status_code = 504
        error_type = "api_timeout"
    elif isinstance(exc, APITimeoutError):
        status_code = 504
        error_type = "api_timeout"
    elif isinstance(exc, ValueError):
        status_code = 400
        error_type = "invalid_request"

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": error_type,
                "message": message,
            }
        },
    )
