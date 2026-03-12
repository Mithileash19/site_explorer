"""
core/middleware.py — Request ID injection + request timing middleware.
"""
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from core.logging import get_logger

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        duration = round((time.perf_counter() - start) * 1000, 1)

        logger.info(
            "[%s] %s %s → %d  (%s ms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )
        response.headers["X-Request-ID"] = request_id
        return response
