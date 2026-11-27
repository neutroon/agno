"""A generator tool with show_result that yields run content events (a member run streamed through
a delegation tool, as in route mode) must produce the same events on the sync and async paths: the
ModelResponse carrying the content and the content event itself."""

from typing import Any, AsyncIterator, Callable, Iterator, List

import pytest

from agno.models.base import Model
from agno.models.message import Message
from agno.models.response import ModelResponse
from agno.run.agent import RunContentEvent
from agno.tools.function import Function, FunctionCall

_CHUNKS = ["first ", "second"]


class _StubModel(Model):
    def __init__(self):
        super().__init__(id="stub", name="stub", provider="stub")

    def invoke(self, *args, **kwargs) -> ModelResponse:
        raise NotImplementedError

    async def ainvoke(self, *args, **kwargs) -> ModelResponse:
        raise NotImplementedError

    def invoke_stream(self, *args, **kwargs) -> Iterator[ModelResponse]:
        raise NotImplementedError

    async def ainvoke_stream(self, *args, **kwargs) -> AsyncIterator[ModelResponse]:
        raise NotImplementedError

    def _parse_provider_response(self, response: Any, **kwargs) -> ModelResponse:
        raise NotImplementedError

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        raise NotImplementedError


def sync_member_tool() -> Iterator[RunContentEvent]:
    """Stream a member's content."""
    for chunk in _CHUNKS:
        yield RunContentEvent(agent_id="member", content=chunk)


async def async_member_tool() -> AsyncIterator[RunContentEvent]:
    """Stream a member's content."""
    for chunk in _CHUNKS:
        yield RunContentEvent(agent_id="member", content=chunk)


def _function_call(tool: Callable) -> FunctionCall:
    function = Function.from_callable(tool)
    function.process_entrypoint()
    function.show_result = True
    return FunctionCall(function=function, arguments={}, call_id="call_1")


def _content(responses: List[Any]) -> List[Any]:
    """The content responses and content events, in order, as (kind, content)."""
    out = []
    for r in responses:
        if isinstance(r, RunContentEvent):
            out.append(("event", r.content))
        elif isinstance(r, ModelResponse) and r.content is not None and not r.tool_executions:
            out.append(("response", r.content))
    return out


_EXPECTED = [(kind, chunk) for chunk in _CHUNKS for kind in ("response", "event")]


def test_sync_path_yields_response_and_content_event():
    """Verify the sync path's behavior, which the async path must match."""
    results: List[Message] = []
    responses = list(_StubModel().run_function_call(_function_call(sync_member_tool), function_call_results=results))
    assert _content(responses) == _EXPECTED


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", [sync_member_tool, async_member_tool])
async def test_async_path_yields_response_and_content_event(tool):
    """Verify the async path also bubbles up the content event, not only the ModelResponse."""
    results: List[Message] = []
    responses = [
        r async for r in _StubModel().arun_function_calls([_function_call(tool)], function_call_results=results)
    ]
    assert _content(responses) == _EXPECTED
    assert results[0].content == "".join(_CHUNKS)
