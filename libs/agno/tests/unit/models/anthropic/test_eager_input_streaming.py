"""``eager_input_streaming`` marks client tools on streamed requests only.

Without the field the API buffers a tool input until it is complete and sends no event while
it does, so one long argument looks like a dead connection to a read timeout. The field goes
on every tool the caller defines, on the two streaming methods; a non-streamed request, a
strict tool and a server tool are left as they are.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agno.models.anthropic.claude import Claude
from agno.models.message import Message

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "add_entry",
            "description": "Add one entry.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "strict_entry",
            "description": "Add one entry, validated.",
            "strict": True,
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
    },
    {"type": "web_search_20250305", "name": "web_search"},
]


def _by_name(tools: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {tool["name"]: tool for tool in tools}


def _model(**kwargs: Any) -> Claude:
    return Claude(id="claude-haiku-5-5", api_key="test", **kwargs)


class _Stream:
    def __enter__(self) -> "_Stream":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def __iter__(self) -> Any:
        return iter(())

    async def __aenter__(self) -> "_Stream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def __aiter__(self) -> "_Stream":
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


def _messages() -> List[Message]:
    return [Message(role="user", content="add it")]


def test_off_by_default() -> None:
    model = _model()
    request_kwargs = model._prepare_request_kwargs("system", tools=TOOLS)
    before = [dict(tool) for tool in request_kwargs["tools"]]
    model._apply_eager_input_streaming(request_kwargs)
    assert request_kwargs["tools"] == before


def test_marks_only_client_tools_and_leaves_the_input_untouched() -> None:
    model = _model(eager_input_streaming=True)
    request_kwargs = model._prepare_request_kwargs("system", tools=TOOLS)
    original = request_kwargs["tools"]
    model._apply_eager_input_streaming(request_kwargs)
    tools = _by_name(request_kwargs["tools"])
    assert tools["add_entry"]["eager_input_streaming"] is True
    assert "eager_input_streaming" not in tools["strict_entry"]
    assert "eager_input_streaming" not in tools["web_search"]
    assert all("eager_input_streaming" not in tool for tool in original)


def test_streamed_request_carries_the_field_and_a_plain_request_does_not() -> None:
    model = _model(eager_input_streaming=True)
    client = MagicMock()
    client.messages.stream.return_value = _Stream()
    client.beta.messages.stream.return_value = _Stream()
    model.get_client = lambda: client  # type: ignore[method-assign]

    list(model.invoke_stream(messages=_messages(), assistant_message=Message(role="assistant"), tools=TOOLS[:1]))
    sent = client.messages.stream.call_args.kwargs["tools"]
    assert _by_name(sent)["add_entry"]["eager_input_streaming"] is True

    try:
        model.invoke(messages=_messages(), assistant_message=Message(role="assistant"), tools=TOOLS[:1])
    except Exception:
        pass  # the mocked response is not a real message; only the request matters here
    sent = client.messages.create.call_args.kwargs["tools"]
    assert "eager_input_streaming" not in _by_name(sent)["add_entry"]


@pytest.mark.asyncio
async def test_async_streamed_request_carries_the_field() -> None:
    model = _model(eager_input_streaming=True)
    client = MagicMock()
    client.messages.stream.return_value = _Stream()
    model.get_async_client = lambda: client  # type: ignore[method-assign]

    async for _ in model.ainvoke_stream(
        messages=_messages(), assistant_message=Message(role="assistant"), tools=TOOLS[:1]
    ):
        pass
    sent = client.messages.stream.call_args.kwargs["tools"]
    assert _by_name(sent)["add_entry"]["eager_input_streaming"] is True
