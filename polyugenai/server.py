"""Local, authenticated OpenAI-compatible HTTP endpoints."""

import json
import os
import secrets
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from pathlib import Path

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .adapter import (
    ORIGIN,
    STATE_PATH,
    APIError,
    PolyUClient,
    content_chunks,
    load_cookies,
    prepare_messages,
)

MAX_BODY = 16 * 1024 * 1024


async def close_upstream(
    chunks: AsyncGenerator[str, None] | None, client: httpx.AsyncClient
) -> None:
    with anyio.CancelScope(shield=True):
        try:
            if chunks is not None:
                await chunks.aclose()
        finally:
            await client.aclose()


def create_client() -> httpx.AsyncClient:
    state_path = Path(os.getenv("POLYU_STATE_PATH", str(STATE_PATH)))
    return httpx.AsyncClient(
        base_url=ORIGIN,
        cookies=load_cookies(state_path),
        follow_redirects=False,
        timeout=httpx.Timeout(120, connect=15),
        headers={"Origin": ORIGIN, "Referer": ORIGIN + "/", "User-Agent": "PolyUGenAI-OpenAI/0.1"},
    )


def validate_request(body: object) -> tuple[dict, str, list[str]]:
    if not isinstance(body, dict):
        raise APIError(400, "Request body must be a JSON object.", "invalid_request")
    allowed = {"model", "messages", "stream", "reasoning_effort", "n", "stream_options"}
    if set(body) - allowed:
        raise APIError(
            400,
            "Unsupported parameters: " + ", ".join(sorted(set(body) - allowed)),
            "unsupported_parameter",
        )
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise APIError(400, "model must be a non-empty string.")
    if "stream" in body and type(body["stream"]) is not bool:
        raise APIError(400, "stream must be a boolean.")
    if "n" in body and (type(body["n"]) is not int or body["n"] != 1):
        raise APIError(400, "Only n=1 is supported.")
    if body.get("reasoning_effort") not in (
        None,
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    ):
        raise APIError(400, "Invalid reasoning_effort.")
    options = body.get("stream_options")
    if options is not None and (not isinstance(options, dict) or options or not body.get("stream")):
        raise APIError(
            400,
            "stream_options supports only {} with stream=true. "
            "The school stream does not expose per-request token usage.",
        )
    prompt, images = prepare_messages(body.get("messages"))
    return body, prompt, images


def create_app(
    api_key: str | None = None, client_factory: Callable[[], httpx.AsyncClient] = create_client
) -> FastAPI:
    app = FastAPI(
        title="PolyU GenAI OpenAI adapter", docs_url=None, redoc_url=None, openapi_url=None
    )

    @app.exception_handler(APIError)
    async def api_error(_request: Request, exc: APIError):
        return JSONResponse(exc.payload(), status_code=exc.status)

    @app.exception_handler(httpx.HTTPError)
    async def transport_error(_request: Request, exc: httpx.HTTPError):
        error = APIError(
            504 if isinstance(exc, httpx.TimeoutException) else 502,
            "School API timed out."
            if isinstance(exc, httpx.TimeoutException)
            else "Unable to connect to the school API.",
        )
        return JSONResponse(error.payload(), status_code=error.status)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        expected = api_key or os.getenv("POLYUGENAI_API_KEY")
        if not expected:
            error = APIError(503, "Set POLYUGENAI_API_KEY or start with polyugenai serve.")
            return JSONResponse(error.payload(), status_code=503)
        supplied = request.headers.get("authorization", "")
        if not secrets.compare_digest(supplied.encode(), ("Bearer " + expected).encode()):
            error = APIError(401, "Invalid local API key.", "invalid_api_key")
            return JSONResponse(error.payload(), status_code=401)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        async with client_factory() as client:
            details = await PolyUClient(client).models()
        return {
            "object": "list",
            "data": [
                {"id": model["id"], "object": "model", "created": 0, "owned_by": "polyu"}
                for model in details
            ],
        }

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        raw = bytearray()
        async for part in request.stream():
            raw.extend(part)
            if len(raw) > MAX_BODY:
                raise APIError(413, "Request body exceeds 16 MiB.", "request_too_large")
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise APIError(400, "Invalid JSON.", "invalid_json") from exc
        body, prompt, images = validate_request(body)
        client = client_factory()
        chunks = None
        handed_off = False
        try:
            upstream = PolyUClient(client)
            details = await upstream.models()
            model = next(
                (m for m in details if m["id"].casefold() == body["model"].casefold()), None
            )
            if model is None:
                raise APIError(404, "Unknown model. See GET /v1/models.", "model_not_found")
            effort = body.get("reasoning_effort")
            settings = model.get("reasoning_effort")
            supported = settings.get("options") if isinstance(settings, dict) else None
            if effort is not None and isinstance(supported, list) and effort not in supported:
                raise APIError(400, "reasoning_effort is not supported by this school model.")
            response = await upstream.start(model, prompt, images, effort)
            chunks = content_chunks(response)
            first = await anext(chunks)
            completion_id, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())

            def chunk(delta: dict, finish: str | None = None) -> dict:
                return {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model["id"],
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }

            if not body.get("stream", False):
                parts = [first]
                async for value in chunks:
                    parts.append(value)
                return {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model["id"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "".join(parts),
                                "refusal": None,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": None,
                }

            async def events():
                def event(value: dict) -> str:
                    return "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"

                try:
                    yield event(chunk({"role": "assistant", "content": ""}))
                    yield event(chunk({"content": first}))
                    async for value in chunks:
                        yield event(chunk({"content": value}))
                    yield event(chunk({}, "stop"))
                    yield "data: [DONE]\n\n"
                except (APIError, httpx.HTTPError) as exc:
                    error = (
                        exc
                        if isinstance(exc, APIError)
                        else APIError(502, "School stream interrupted.")
                    )
                    yield event(error.payload())
                finally:
                    await close_upstream(chunks, client)

            result = StreamingResponse(
                events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
            # StreamingResponse now owns the open upstream resources.
            handed_off = True
            return result
        finally:
            if not handed_off:
                await close_upstream(chunks, client)

    return app


app = create_app()
