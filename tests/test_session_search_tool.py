"""Regression tests for tools/session_search_tool.py::_discover.

Covers the "current-lineage starvation" bug: when the CURRENT session's
lineage dominates the top FTS rows (and is filtered out post-fetch), a
single-page fetch returned count=0 / sessions_searched=0. _discover must
now page deeper (bounded) until it finds enough distinct non-current
lineages, without ever surfacing the current session itself.
"""

import json

from tools import session_search_tool as sst


class FakeDB:
    """Minimal SessionDB stand-in. Rows 1-4 belong to the current session
    lineage (filtered out); row 5 is an older matching session."""

    def __init__(self, rows=None):
        self.calls = []
        self._rows = rows if rows is not None else [
            {"id": 1, "session_id": "current", "role": "assistant", "snippet": "match"},
            {"id": 2, "session_id": "current", "role": "user", "snippet": "match"},
            {"id": 3, "session_id": "current", "role": "assistant", "snippet": "match"},
            {"id": 4, "session_id": "current", "role": "user", "snippet": "match"},
            {"id": 5, "session_id": "old", "role": "assistant", "snippet": "historic match"},
        ]

    def search_messages(self, *, query, role_filter, exclude_sources, limit, offset, sort):
        self.calls.append((limit, offset))
        return [dict(r) for r in self._rows[offset:offset + limit]]

    def get_session(self, sid):
        return {
            "id": sid,
            "parent_session_id": None,
            "started_at": 0,
            "source": "cli",
            "model": "test",
            "title": sid,
        }

    def get_anchored_view(self, sid, msg_id, window=5, bookend=3):
        return {
            "bookend_start": [],
            "window": [{"id": msg_id, "role": "assistant", "content": "historic match", "timestamp": 0}],
            "bookend_end": [],
            "messages_before": 0,
            "messages_after": 0,
        }


def _small_scan(monkeypatch, scan_limit=2, max_rows=10):
    monkeypatch.setattr(sst, "_DISCOVER_SCAN_LIMIT", scan_limit)
    monkeypatch.setattr(sst, "_DISCOVER_MAX_SCAN_ROWS", max_rows)
    monkeypatch.setattr(sst, "_title_match_result", lambda *a, **k: None)


def test_discover_pages_past_current_lineage(monkeypatch):
    """The historical session must surface even though the current session
    fills the first two pages — the exact bug that made recall return 0."""
    db = FakeDB()
    _small_scan(monkeypatch)

    out = json.loads(
        sst._discover(
            db,
            query="book repo",
            role_filter=None,
            limit=1,
            sort=None,
            current_session_id="current",
        )
    )

    assert out["success"] is True
    assert out["count"] == 1
    assert out["results"][0]["session_id"] == "old"
    # Paged through both current-only pages before reaching the old-session row.
    assert db.calls[:3] == [(2, 0), (2, 2), (2, 4)]


def test_discover_never_returns_current_session(monkeypatch):
    """If EVERY match is the current lineage, recall returns empty rather
    than surfacing self-references as 'memory'."""
    only_current = [
        {"id": i, "session_id": "current", "role": "assistant", "snippet": "m"}
        for i in range(1, 8)
    ]
    db = FakeDB(rows=only_current)
    _small_scan(monkeypatch, scan_limit=2, max_rows=6)

    out = json.loads(
        sst._discover(
            db,
            query="anything",
            role_filter=None,
            limit=3,
            sort=None,
            current_session_id="current",
        )
    )

    assert out["success"] is True
    assert out["count"] == 0
    # Scan is bounded — must not page forever chasing rows that never qualify.
    assert sum(n for n, _ in db.calls) <= 6


def test_discover_stops_early_when_enough_found(monkeypatch):
    """When the first page already has enough distinct non-current lineages,
    it must not keep paging."""
    rows = [
        {"id": 1, "session_id": "a", "role": "assistant", "snippet": "m"},
        {"id": 2, "session_id": "b", "role": "assistant", "snippet": "m"},
        {"id": 3, "session_id": "c", "role": "assistant", "snippet": "m"},
        {"id": 4, "session_id": "d", "role": "assistant", "snippet": "m"},
    ]
    db = FakeDB(rows=rows)
    _small_scan(monkeypatch, scan_limit=10, max_rows=100)

    out = json.loads(
        sst._discover(
            db, query="q", role_filter=None, limit=2, sort=None, current_session_id="current",
        )
    )

    assert out["count"] == 2
    # Only one page fetched — enough lineages on the first page.
    assert db.calls == [(10, 0)]


def test_discover_relaxed_fallback_surfaces_old_session(monkeypatch):
    """Strict-AND matches only the current session (filtered); relaxed OR
    fallback surfaces an old session sharing most terms, re-ranked by overlap."""
    strict = [{"id": 1, "session_id": "current", "role": "assistant",
               "snippet": "book github repo extracted agents architecture",
               "content": "book github repo extracted agents architecture"}]
    relaxed = strict + [
        {"id": 9, "session_id": "old", "role": "assistant",
         "snippet": "the agents architecture book repo we extracted",
         "content": "the agents architecture book repo we extracted"},
        {"id": 10, "session_id": "weak", "role": "assistant",
         "snippet": "architecture only", "content": "architecture only"},
    ]

    class TwoModeDB(FakeDB):
        def search_messages(self, *, query, role_filter, exclude_sources, limit, offset, sort):
            self.calls.append((limit, offset))
            data = relaxed if " OR " in query else strict
            return [dict(r) for r in data[offset:offset + limit]]

    db = TwoModeDB()
    _small_scan(monkeypatch, scan_limit=50, max_rows=500)
    out = json.loads(sst._discover(
        db, query="book github repo extracted agents architecture",
        role_filter=None, limit=3, sort=None, current_session_id="current"))
    ids = [r["session_id"] for r in out["results"]]
    assert out["success"] is True and out["count"] >= 1
    assert "old" in ids and "current" not in ids
    assert ids[0] == "old"
