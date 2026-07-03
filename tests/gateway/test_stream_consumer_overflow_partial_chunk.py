"""Regression coverage for the first-send overflow chunk loop.

Covers the branch in ``GatewayStreamConsumer.run`` where the accumulated
final text exceeds the platform limit AND ``_message_id is None`` (first
send / post-segment-break).  The text is split by ``truncate_message``
and delivered chunk-by-chunk.

The historical bug: delivery was marked complete when *any* single chunk
landed.  If chunk 1 sent and chunk 2 was throttled/rejected, the turn was
marked delivered and the tail was silently dropped — invisible to the
streaming on/off toggle because both paths share this accounting.

Contract asserted here:
  * ``final_response_sent`` becomes True only when EVERY chunk lands.
  * On partial failure the final flags stay False so the outer gateway
    does NOT suppress its authoritative final send (``run.py`` delivers
    from ``response["final_response"]``, not from consumer state).
  * The consumer's ``_accumulated`` buffer is preserved on partial
    failure as internal state hygiene (not the outer send source).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _ok(message_id: str) -> SimpleNamespace:
    return SimpleNamespace(success=True, message_id=message_id, raw_response=None)


def _fail(*, retryable: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        success=False, message_id=None, raw_response=None, retryable=retryable,
    )


def _make_adapter(send_side_effect) -> SimpleNamespace:
    """Strict fake adapter.

    SimpleNamespace (not MagicMock) so no phantom truthy attributes can
    silently enable unrelated branches (draft streaming, requires-finalize).
    ``truncate_message`` is deterministic: always two chunks.
    """
    adapter = SimpleNamespace()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 650
    adapter.send = AsyncMock(side_effect=send_side_effect)
    adapter.edit_message = AsyncMock(return_value=_ok("should-not-edit"))
    adapter.delete_message = AsyncMock(return_value=True)
    adapter.truncate_message = lambda text, limit, len_fn=len: ["HEAD-chunk", "TAIL-chunk"]
    return adapter


def _consumer(adapter) -> GatewayStreamConsumer:
    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="chat",
        config=StreamConsumerConfig(
            edit_interval=0.0, buffer_threshold=0, cursor=" ▉",
        ),
    )
    consumer._chunk_retry_delay = 0.0  # no real backoff in tests
    return consumer


# Oversize text so _len_fn(accumulated) > _safe_limit on the first send,
# with _message_id still None -> enters the target overflow branch.
_BIG = "word " * 200


@pytest.mark.asyncio
async def test_partial_chunk_failure_not_marked_delivered_and_buffer_preserved():
    """Chunk 1 lands, chunk 2 fails -> turn must NOT be marked delivered.

    The final flags staying False is what prevents the outer gateway from
    suppressing its authoritative final send (run.py delivers from
    response["final_response"]). _accumulated preservation is internal
    state hygiene, not the outer send source."""
    adapter = _make_adapter([_ok("m1"), _fail(retryable=True), _fail(retryable=True)])
    consumer = _consumer(adapter)
    # Prime stale True flags from a hypothetical prior segment. The partial
    # failure MUST reset both to False, or run.py would suppress the
    # authoritative final send and the truncation would survive the fix.
    consumer._final_response_sent = True
    consumer._final_content_delivered = True
    consumer.on_delta(_BIG)
    consumer.finish()
    await consumer.run()

    assert consumer.final_response_sent is False, (
        "a failed tail chunk must not be reported as final delivery"
    )
    assert consumer.final_content_delivered is False
    # The outer gateway suppresses its final send only when the FULL final
    # text was delivered (exact match). A visible head chunk must not fool it.
    assert consumer.has_delivered_text(_BIG) is False
    assert consumer._accumulated == _BIG, (
        "authoritative consumer buffer preserved as internal state hygiene "
        "on partial failure"
    )
    # Exactly one retry of the failed chunk: chunk1 ok + chunk2 fail + 1 retry.
    assert adapter.send.await_count == 3
    # The retry must resend the SAME tail chunk, never restart from the head
    # (which would duplicate already-visible content).
    sent = [c.kwargs["content"] for c in adapter.send.await_args_list]
    assert sent == ["HEAD-chunk", "TAIL-chunk", "TAIL-chunk"]


@pytest.mark.asyncio
async def test_non_retryable_chunk_failure_is_not_retried():
    """A non-retryable failure must NOT trigger a blind resend (would risk
    duplicate delivery). chunk1 ok + chunk2 fail(non-retryable) = 2 sends."""
    adapter = _make_adapter([_ok("m1"), _fail(retryable=False)])
    consumer = _consumer(adapter)
    consumer.on_delta(_BIG)
    consumer.finish()
    await consumer.run()

    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False
    assert consumer.has_delivered_text(_BIG) is False
    assert consumer._accumulated == _BIG
    assert adapter.send.await_count == 2


@pytest.mark.asyncio
async def test_all_chunks_land_marks_delivered_and_clears_buffer():
    """Both chunks land -> delivered flags set, buffer cleared (unchanged
    happy path)."""
    adapter = _make_adapter([_ok("m1"), _ok("m2")])
    consumer = _consumer(adapter)
    consumer.on_delta(_BIG)
    consumer.finish()
    await consumer.run()

    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True
    assert consumer._accumulated == ""
    assert adapter.send.await_count == 2


@pytest.mark.asyncio
async def test_failed_chunk_retried_once_then_succeeds():
    """Chunk 2 fails once, retry succeeds -> delivered, buffer cleared."""
    adapter = _make_adapter([_ok("m1"), _fail(retryable=True), _ok("m2")])
    consumer = _consumer(adapter)
    consumer.on_delta(_BIG)
    consumer.finish()
    await consumer.run()

    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True
    assert consumer._accumulated == ""
    assert adapter.send.await_count == 3
    # The successful retry must resend the SAME tail chunk, not the head.
    sent = [c.kwargs["content"] for c in adapter.send.await_args_list]
    assert sent == ["HEAD-chunk", "TAIL-chunk", "TAIL-chunk"]
