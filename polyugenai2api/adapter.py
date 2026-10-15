"""Translate the protocol observed in the PolyU HAR into Chat Completions."""

import base64
import binascii
import json
import os
import re
from collections.abc import AsyncGenerator
from pathlib import Path

import anyio
import httpx

ORIGIN = "https://genai.polyu.edu.hk"
STATE_PATH = Path(".auth/storage-state.json")


class APIError(Exception):
    def __init__(self, status: int, message: str, code: str = "upstream_error"):
        super().__init__(message)
        self.status = status
        self.code = code

    def payload(self) -> dict:
        return {
            "error": {
                "message": str(self),
                "type": "invalid_request_error" if self.status < 500 else "api_error",
                "param": None,
                "code": self.code,
            }
        }


def prepare_messages(messages: object) -> tuple[str, list[str]]:
    if not isinstance(messages, list) or not messages or len(messages) > 200:
        raise APIError(400, "messages must contain 1–200 messages.", "invalid_messages")
    transcript, images = [], []
    for message in messages:
        if not isinstance(message, dict) or set(message) - {"role", "content", "name"}:
            raise APIError(400, "Unsupported message fields.", "invalid_messages")
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "developer", "user", "assistant"):
            raise APIError(400, "Only system/developer/user/assistant roles are supported.")
        parts = []
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise APIError(400, "Invalid content part.")
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif part.get("type") == "image_url" and role == "user":
                    image = part.get("image_url")
                    url = image.get("url") if isinstance(image, dict) else None
                    if not isinstance(url, str) or not re.fullmatch(
                        r"data:image/(png|jpeg|webp|gif);base64,[A-Za-z0-9+/=]+", url
                    ):
                        raise APIError(400, "Images must be base64 data URLs (png/jpeg/webp/gif).")
                    try:
                        base64.b64decode(url.split(",", 1)[1], validate=True)
                    except (ValueError, binascii.Error) as exc:
                        raise APIError(400, "Invalid image base64.") from exc
                    images.append(url)
                else:
                    raise APIError(400, "Unsupported message content part.")
        else:
            raise APIError(400, "content must be a string or a list of content parts.")
        transcript.append((role, "\n".join(parts)))
    if transcript[-1][0] != "user":
        raise APIError(400, "The last message must be a user message.")
    if not any(text.strip() for _, text in transcript) and not images:
        raise APIError(400, "Empty prompt.")
    if len(transcript) == 1:
        return transcript[0][1], images
    # The upstream accepts one input field, not a native role-based messages array.
    prompt = (
        "Continue the following conversation, answering only the final user message.\n"
        + json.dumps([{"role": r, "content": t} for r, t in transcript], ensure_ascii=False)
    )
    return prompt, images


def load_cookies(state_path: Path) -> httpx.Cookies:
    cookies = httpx.Cookies()
    raw_cookie = os.getenv("POLYUGENAI2API_COOKIE")
    if raw_cookie:
        for part in raw_cookie.split(";"):
            name, sep, value = part.strip().partition("=")
            if sep and name:
                cookies.set(name, value, domain="genai.polyu.edu.hk", path="/")
    elif state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
            for cookie in state["cookies"]:
                domain = cookie["domain"].lstrip(".")
                if "genai.polyu.edu.hk" == domain or "genai.polyu.edu.hk".endswith("." + domain):
                    cookies.set(
                        cookie["name"],
                        cookie["value"],
                        domain=cookie["domain"],
                        path=cookie.get("path", "/"),
                    )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise APIError(
                503, "Invalid login state. Run polyugenai2api login again.", "login_required"
            ) from exc
    if not cookies:
        raise APIError(
            503, "Run polyugenai2api login or set POLYUGENAI2API_COOKIE.", "login_required"
        )
    return cookies


class PolyUClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    @staticmethod
    def check(response: httpx.Response) -> None:
        if response.status_code in (401, 403) or response.is_redirect:
            raise APIError(
                401,
                "School session expired or access denied. Run polyugenai2api login.",
                "upstream_authentication_error",
            )
        if response.status_code == 429:
            raise APIError(429, "School quota or rate limit reached.", "rate_limit_exceeded")
        if not response.is_success:
            raise APIError(502, f"School API returned HTTP {response.status_code}.")

    async def post(self, path: str, **kwargs) -> dict:
        response = await self.client.post(path, **kwargs)
        self.check(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise APIError(502, "School API returned non-JSON data; check login.") from exc
        if not isinstance(data, dict) or data.get("error"):
            raise APIError(502, "School API reported an error.")
        return data

    async def models(self) -> list[dict]:
        data = await self.post("/api/llm/list")
        details = data.get("details")
        if not isinstance(details, list):
            raise APIError(502, "Unexpected school model-list response.")
        result = [
            m
            for m in details
            if isinstance(m, dict)
            and m.get("is_authorized") is True
            and m.get("streaming_enabled") is True
            and isinstance(m.get("id"), str)
            and isinstance(m.get("endpoint_key"), str)
            and re.fullmatch(r"[A-Za-z0-9_-]+", m["endpoint_key"])
        ]
        return result

    async def start(
        self, model: dict, prompt: str, images: list[str], effort: str | None
    ) -> httpx.Response:
        # Always use a fresh topic: no cross-request history or concurrent context mixing.
        topic = await self.post("/api/topics/create", json={"title": "OpenAI adapter"})
        topic_id = topic.get("topic_id")
        if type(topic_id) is not int or topic_id <= 0:
            raise APIError(502, "School did not return a valid topic_id.")
        form = [
            ("genai_model", (None, model["id"])),
            ("input", (None, prompt)),
            ("topic_id", (None, str(topic_id))),
        ]
        form.extend(("images", (None, image)) for image in images)
        chat = await self.post("/api/chat/create", files=form)
        chat_id = chat.get("chat_id")
        if type(chat_id) is not int or chat_id <= 0:
            raise APIError(502, "School did not return a valid chat_id.")
        think = model.get("think", {})
        default_effort = model.get("reasoning_effort", {})
        if not isinstance(think, dict) or not isinstance(default_effort, dict):
            raise APIError(502, "Unexpected school model settings.")
        body = {
            "genai_model": model["id"],
            "chat_id": chat_id,
            "think_enabled": effort not in (None, "none")
            if effort is not None
            else think.get("default", False),
            "reasoning_effort": effort if effort is not None else default_effort.get("default"),
            "web_search": False,
            "deep_research": False,
            "params": {
                "guidance": 0,
                "steps": 0,
                "image_strength": 0,
                "seed": "0",
                "width": 1024,
                "height": 1024,
                "ratioW": 1,
                "ratioH": 1,
                "controlnet": "none",
                "orientation": "landscape",
                "resolution": "512",
                "ratio": "1:1",
            },
            "seed": "0",
            "is_stream": True,
        }
        request = self.client.build_request(
            "POST",
            f"/api/chat/send/{model['endpoint_key']}",
            json=body,
            headers={"Accept": "text/event-stream"},
        )
        response = await self.client.send(request, stream=True)
        try:
            self.check(response)
            if "text/event-stream" not in response.headers.get("content-type", ""):
                raise APIError(502, "School did not return an event stream; check login.")
        except BaseException:
            with anyio.CancelScope(shield=True):
                await response.aclose()
            raise
        return response


async def content_chunks(response: httpx.Response) -> AsyncGenerator[str, None]:
    data_lines = []
    seen_content = False

    def decode(lines: list[str]) -> str | None:
        data = "\n".join(lines)
        if data == "[DONE]":
            return None
        try:
            payload = json.loads(data)
        except ValueError as exc:
            raise APIError(502, "Malformed school SSE event.") from exc
        if not isinstance(payload, dict) or payload.get("error"):
            raise APIError(502, "School stream reported an error.")
        value = payload.get("content")
        if value is not None and not isinstance(value, str):
            raise APIError(502, "Unexpected school content event.")
        return value

    try:
        async for line in response.aiter_lines():
            if len(line) > 1_000_000:
                raise APIError(502, "School SSE event exceeds size limit.")
            if line.startswith("data:"):
                data_lines.append(line[5:].removeprefix(" "))
            elif line == "" and data_lines:
                if data_lines == ["[DONE]"]:
                    break
                value = decode(data_lines)
                data_lines = []
                if value is not None:
                    seen_content = True
                    yield value
        if data_lines and data_lines != ["[DONE]"]:
            value = decode(data_lines)
            if value is not None:
                seen_content = True
                yield value
        if not seen_content:
            raise APIError(502, "School stream ended without content.")
    finally:
        with anyio.CancelScope(shield=True):
            await response.aclose()
