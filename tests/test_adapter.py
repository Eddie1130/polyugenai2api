import json

import anyio
import anyio.lowlevel
import httpx
import pytest
from openai import AsyncOpenAI

from polyugenai2api.adapter import (
    APIError,
    PolyUClient,
    content_chunks,
    load_cookies,
    prepare_messages,
)
from polyugenai2api.cli import private_write
from polyugenai2api.server import close_upstream, create_app

KEY = "test-local-key"
MODEL = {
    "id": "GPT56Sol",
    "endpoint_key": "gpt56sol",
    "is_authorized": True,
    "streaming_enabled": True,
    "think": {"default": True},
    "reasoning_effort": {"default": "medium", "options": ["low", "medium", "high"]},
}
PROMPT = {"model": "GPT56Sol", "messages": [{"role": "user", "content": "Hello"}]}


class FragmentedStream(httpx.AsyncByteStream):
    def __init__(self, data):
        self.data = data
        self.closed = False

    async def __aiter__(self):
        for byte in self.data:
            yield bytes([byte])

    async def aclose(self):
        self.closed = True


def fixture_app(
    sse=b'data: {"content":"Hello"}\n\ndata: {"content":" again!"}\n\n', fail_path=None, status=401
):
    calls, clients, streams = [], [], []
    topics = 0

    def handler(request):
        nonlocal topics
        calls.append(request)
        if request.url.path == fail_path:
            return httpx.Response(status, text="private upstream error must not leak")
        if request.url.path == "/api/llm/list":
            return httpx.Response(200, json={"details": [MODEL]})
        if request.url.path == "/api/topics/create":
            topics += 1
            return httpx.Response(200, json={"topic_id": topics})
        if request.url.path == "/api/chat/create":
            return httpx.Response(200, json={"chat_id": 100 + topics})
        if request.url.path == "/api/chat/send/gpt56sol":
            stream = FragmentedStream(sse)
            streams.append(stream)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        raise AssertionError(request.url)

    def factory():
        client = httpx.AsyncClient(
            base_url="https://genai.polyu.edu.hk", transport=httpx.MockTransport(handler)
        )
        clients.append(client)
        return client

    return create_app(KEY, factory), calls, clients, streams


async def request_app(app, body=PROMPT, key=KEY):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as c:
        return await c.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer " + key}
        )


async def test_sdk_models_and_completion_and_stream():
    app, calls, clients, streams = fixture_app()
    async with AsyncOpenAI(
        api_key=KEY,
        base_url="http://test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app)),
    ) as sdk:
        models = await sdk.models.list()
        assert models.data[0].id == "GPT56Sol"
        result = await sdk.chat.completions.create(**PROMPT)
        assert result.choices[0].message.content == "Hello again!"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage is None
        stream = await sdk.chat.completions.create(**PROMPT, stream=True)
        events = [chunk async for chunk in stream]
        assert events[0].choices[0].delta.role == "assistant"
        assert "".join(e.choices[0].delta.content or "" for e in events) == "Hello again!"
        assert events[-1].choices[0].finish_reason == "stop"
        assert len({e.id for e in events}) == 1
    created = [r for r in calls if r.url.path == "/api/chat/create"]
    assert b'name="topic_id"\r\n\r\n1' in created[0].content
    assert b'name="topic_id"\r\n\r\n2' in created[1].content
    sent = [json.loads(r.content) for r in calls if "/send/" in r.url.path]
    assert sent[0]["reasoning_effort"] == "medium"
    assert all(c.is_closed for c in clients)
    assert all(s.closed for s in streams)


@pytest.mark.parametrize(
    "change",
    [
        {"temperature": 0.3},
        {"tools": []},
        {"stream": "true"},
        {"n": True},
        {"n": 2},
        {"stream": True, "stream_options": {"include_usage": True}},
        {"messages": []},
        {"messages": [{"role": "tool", "content": "x"}]},
        {"model": None},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://127.0.0.1/private"}}
                    ],
                }
            ]
        },
    ],
)
async def test_invalid_input_never_contacts_school(change):
    app, calls, _, _ = fixture_app()
    response = await request_app(app, PROMPT | change)
    assert response.status_code == 400
    assert response.json()["error"]["message"]
    assert not calls


async def test_auth_before_upstream_and_validation():
    app, calls, _, _ = fixture_app()
    response = await request_app(app, key="wrong")
    assert response.status_code == 401
    assert not calls


@pytest.mark.parametrize(
    "path", ["/api/llm/list", "/api/topics/create", "/api/chat/create", "/api/chat/send/gpt56sol"]
)
@pytest.mark.parametrize("status,expected", [(401, 401), (302, 401), (429, 429), (500, 502)])
async def test_upstream_errors_are_safe_and_close_clients(path, status, expected):
    app, _, clients, _ = fixture_app(fail_path=path, status=status)
    response = await request_app(app)
    assert response.status_code == expected
    assert "private upstream" not in response.text
    assert all(c.is_closed for c in clients)


async def test_partial_stream_error_does_not_signal_success():
    app, _, clients, streams = fixture_app(
        sse=b'data: {"content":"Hi"}\n\ndata: {"error":"private"}\n\n'
    )
    response = await request_app(app, PROMPT | {"stream": True})
    assert '"error"' in response.text
    assert "[DONE]" not in response.text
    assert '"finish_reason": "stop"' not in response.text
    assert "private" not in response.text
    assert all(c.is_closed for c in clients)
    assert all(s.closed for s in streams)


@pytest.mark.parametrize(
    "sse", [b"", b"data: bad-json\n\n", b"data: [DONE]\n\n", b'data: {"content":42}\n\n']
)
async def test_malformed_stream_is_502(sse):
    app, _, clients, streams = fixture_app(sse=sse)
    response = await request_app(app, PROMPT | {"stream": True})
    assert response.status_code == 502
    assert all(c.is_closed for c in clients)
    assert all(s.closed for s in streams)


async def test_sse_unicode_comments_crlf_and_final_unterminated_event():
    stream = FragmentedStream(
        ':keepalive\r\ndata: {"content":"你好"}\r\n\r\ndata: {"content":"!"}'.encode()
    )
    response = httpx.Response(200, stream=stream)
    assert [v async for v in content_chunks(response)] == ["你好", "!"]
    assert stream.closed


def test_multiturn_and_images():
    prompt, images = prepare_messages(
        [
            {"role": "system", "content": "Use Traditional Chinese"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
                ],
            },
        ]
    )
    assert '"role": "system"' in prompt
    assert "first answer" in prompt
    assert images == ["data:image/png;base64,aGk="]


def test_storage_state_filters_domains_and_uses_private_permissions(tmp_path, monkeypatch):
    monkeypatch.delenv("POLYUGENAI2API_COOKIE", raising=False)
    state = tmp_path / "auth" / "state.json"
    private_write(
        state,
        json.dumps(
            {
                "cookies": [
                    {
                        "name": "school",
                        "value": "local",
                        "domain": "genai.polyu.edu.hk",
                        "path": "/",
                    },
                    {"name": "sso", "value": "secret", "domain": "login.microsoftonline.com"},
                ]
            }
        ),
    )
    cookies = load_cookies(state)
    assert cookies.get("school") == "local"
    assert cookies.get("sso") is None
    assert state.stat().st_mode & 0o777 == 0o600


def test_missing_login_is_actionable(tmp_path, monkeypatch):
    monkeypatch.delenv("POLYUGENAI2API_COOKIE", raising=False)
    with pytest.raises(APIError, match="polyugenai2api login"):
        load_cookies(tmp_path / "missing.json")


async def test_unknown_model_and_unsupported_reasoning_have_no_side_effects():
    app, calls, _, _ = fixture_app()
    assert (await request_app(app, PROMPT | {"model": "invented"})).status_code == 404
    assert (await request_app(app, PROMPT | {"reasoning_effort": "xhigh"})).status_code == 400
    assert all(r.url.path == "/api/llm/list" for r in calls)


async def test_only_authorized_streaming_models_are_exposed():
    models = [
        MODEL,
        MODEL | {"id": "Forbidden", "is_authorized": False},
        MODEL | {"id": "NonStreaming", "streaming_enabled": False},
        MODEL | {"id": "MissingFlag", "streaming_enabled": None},
        MODEL | {"id": "BadEndpoint", "endpoint_key": "../../other"},
    ]
    async with httpx.AsyncClient(
        base_url="https://genai.polyu.edu.hk",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"details": models})),
    ) as client:
        assert await PolyUClient(client).models() == [MODEL]


async def test_cancellation_does_not_interrupt_stream_or_client_cleanup():
    class CheckpointStream(FragmentedStream):
        async def aclose(self):
            await anyio.lowlevel.checkpoint()
            self.closed = True

    class CheckpointClient(httpx.AsyncClient):
        cleanup_completed = False

        async def aclose(self):
            await anyio.lowlevel.checkpoint()
            await super().aclose()
            self.cleanup_completed = True

    stream = CheckpointStream(b'data: {"content":"Hi"}\n\n')
    chunks = content_chunks(httpx.Response(200, stream=stream))
    assert await anext(chunks) == "Hi"
    client = CheckpointClient()
    with anyio.CancelScope() as scope:
        scope.cancel()
        await close_upstream(chunks, client)
    assert stream.closed
    assert client.cleanup_completed


async def test_stream_stops_after_done():
    stream = FragmentedStream(b'data: {"content":"Hi"}\n\ndata: [DONE]\n\ndata: bad\n\n')
    response = httpx.Response(200, stream=stream)
    assert [v async for v in content_chunks(response)] == ["Hi"]
