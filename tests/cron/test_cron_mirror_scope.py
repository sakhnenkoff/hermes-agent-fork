"""Cron delivery context-attach: scope resolution + seed/mirror routing."""
from typing import Any
from unittest.mock import MagicMock, patch

from cron.scheduler import (
    _cron_mirror_delivery_scope,
    _attach_cron_delivery_to_context,
)


# ---- scope resolution ----

def test_scope_default_origin():
    assert _cron_mirror_delivery_scope({}, {}) == "origin"


def test_scope_global_target():
    assert _cron_mirror_delivery_scope({}, {"cron": {"mirror_delivery_scope": "target"}}) == "target"


def test_scope_per_job_beats_global():
    job = {"mirror_delivery_scope": "target"}
    assert _cron_mirror_delivery_scope(job, {"cron": {"mirror_delivery_scope": "origin"}}) == "target"


def test_scope_alias_delivered_target_normalizes():
    assert _cron_mirror_delivery_scope({"mirror_delivery_scope": "delivered_target"}, {}) == "target"


def test_scope_garbage_falls_back_to_origin():
    assert _cron_mirror_delivery_scope({"mirror_delivery_scope": "bogus"}, {}) == "origin"


# ---- context attach routing ----

_BASE: dict[str, Any] = dict(
    platform_name="telegram", chat_id="-100123", mirror_text="hi",
    mirror_context_target=True, mirror_origin_target=False,
    mirror_delivered_target=True, in_channel_surface=False,
)


@patch("cron.scheduler._seed_cron_thread_session", return_value=True)
@patch("cron.scheduler._maybe_mirror_cron_delivery")
def test_thread_target_seeds_and_skips_plain_mirror(mock_mirror, mock_seed):
    ts, ic = _attach_cron_delivery_to_context(
        {"id": "j"}, thread_id="55", user_id="u1", adapter=MagicMock(),
        is_dm_target=False, **_BASE,
    )
    assert ts is True
    mock_seed.assert_called_once()
    mock_mirror.assert_called_once()  # called, but enabled=False since seeded
    assert mock_mirror.call_args.kwargs["enabled"] is False


@patch("cron.scheduler._seed_cron_channel_session", return_value=True)
@patch("cron.scheduler._maybe_mirror_cron_delivery")
def test_flat_dm_target_seeds_channel(mock_mirror, mock_seed):
    ts, ic = _attach_cron_delivery_to_context(
        {"id": "j"}, thread_id=None, user_id="u1", adapter=MagicMock(),
        is_dm_target=True, **_BASE,
    )
    assert ic is True
    mock_seed.assert_called_once()


@patch("cron.scheduler._seed_cron_thread_session")
@patch("cron.scheduler._seed_cron_channel_session")
@patch("cron.scheduler._maybe_mirror_cron_delivery")
def test_flat_group_without_user_id_does_nothing(mock_mirror, mock_chan, mock_thread):
    ts, ic = _attach_cron_delivery_to_context(
        {"id": "j"}, thread_id=None, user_id=None, adapter=MagicMock(),
        is_dm_target=False, **_BASE,
    )
    assert (ts, ic) == (False, False)
    mock_thread.assert_not_called()
    mock_chan.assert_not_called()
    mock_mirror.assert_not_called()


@patch("cron.scheduler._seed_cron_thread_session", return_value=False)
@patch("cron.scheduler._maybe_mirror_cron_delivery")
def test_seed_failure_falls_back_to_plain_mirror(mock_mirror, mock_seed):
    _attach_cron_delivery_to_context(
        {"id": "j"}, thread_id="55", user_id="u1", adapter=MagicMock(),
        is_dm_target=False, **_BASE,
    )
    # seed failed -> not seeded -> plain mirror enabled
    assert mock_mirror.call_args.kwargs["enabled"] is True


@patch("cron.scheduler._maybe_mirror_cron_delivery")
def test_context_disabled_is_noop(mock_mirror):
    base = dict(_BASE)
    base["mirror_context_target"] = False
    ts, ic = _attach_cron_delivery_to_context(
        {"id": "j"}, thread_id="55", user_id="u1", adapter=MagicMock(),
        is_dm_target=False, **base,
    )
    assert (ts, ic) == (False, False)
    mock_mirror.assert_not_called()
