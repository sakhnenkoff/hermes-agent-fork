"""Tests for FTS5 natural-language query sanitization in holographic memory.

Regression coverage for the recall-killing bug where raw natural-language
prose was handed straight to FTS5's MATCH operator. FTS5 AND-joins tokens by
default, so a multi-word query like "what happened with the deployment
rollback" required EVERY token to co-occur in a single fact — dropping recall
to zero on exactly the kind of queries the agent issues.

The fix routes both retrieval paths through FactRetriever._build_fts_match,
which drops stopwords and OR-joins quoted content tokens. Precision is restored
downstream by the Jaccard + trust + HRR rerank in FactRetriever.search().
``_sanitize_fts_query`` is the upstream-compatible alias delegating to the
same implementation.

Two paths share the bug class and are both covered here:
  1. FactRetriever._fts_candidates  (used by FactRetriever.search)
  2. MemoryStore.search_facts       (store-level sibling path)
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("numpy")  # retrieval module imports numpy indirectly

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "memory_store.db"))
    try:
        yield s
    finally:
        s.close()


def _quoted_terms(match: str) -> set[str]:
    """Extract the OR-joined quoted tokens from an FTS5 MATCH string."""
    return set(re.findall(r'"([^"]+)"', match))


# ── Sanitizer unit tests ─────────────────────────────────────────────────


class TestBuildFtsMatch:
    def test_prose_becomes_or_joined_quoted_tokens(self):
        match = FactRetriever._build_fts_match(
            "what happened with the deployment rollback?"
        )
        assert '"deployment"' in match
        assert '"rollback"' in match
        assert '"happened"' in match
        assert " OR " in match
        # Stopwords must be dropped.
        assert '"what"' not in match
        assert '"the"' not in match
        assert '"with"' not in match

    def test_operator_only_input_returns_empty(self):
        # Pure punctuation has no word tokens -> empty (caller must guard).
        assert FactRetriever._build_fts_match("!!!") == ""
        assert FactRetriever._build_fts_match("") == ""
        assert FactRetriever._build_fts_match("   ") == ""
        assert FactRetriever._build_fts_match("():") == ""

    def test_hyphens_and_punctuation_are_split_safely(self):
        # Hyphenated terms would error a raw MATCH; here they split cleanly.
        match = FactRetriever._build_fts_match("finn-jobs user-activation")
        assert match == '"finn" OR "jobs" OR "user" OR "activation"'

    def test_embedded_quotes_and_parens_neutralized(self):
        # \w+ tokenization means FTS operator chars never survive into MATCH.
        # ("now" is a stopword, so we use "launch" to keep this focused on
        # quote/paren neutralization rather than stopword filtering.)
        match = FactRetriever._build_fts_match('deploy "prod" (launch)')
        assert match == '"deploy" OR "prod" OR "launch"'

    def test_all_stopword_query_falls_back_to_raw_tokens(self):
        # If stopword filtering nukes everything, we still search (never
        # silently return zero) — the fallback re-adds word tokens as quoted
        # literals (NOT the raw AND-joined string).
        match = FactRetriever._build_fts_match("the and of")
        assert match  # non-empty
        assert " OR " in match
        assert _quoted_terms(match) == {"the", "and", "of"}

    def test_dedupes_and_caps_terms(self):
        match = FactRetriever._build_fts_match("deploy deploy deploy rollback")
        # Deduped: "deploy" appears once.
        assert match.count('"deploy"') == 1
        # Cap respected.
        many = " ".join(f"term{i}" for i in range(40))
        capped = FactRetriever._build_fts_match(many, max_terms=5)
        assert capped.count(" OR ") == 4  # 5 terms -> 4 separators

    def test_upstream_alias_delegates_to_build_fts_match(self):
        # Compatibility alias must produce identical output.
        q = "what happened with the deployment rollback?"
        assert FactRetriever._sanitize_fts_query(q) == FactRetriever._build_fts_match(q)


# ── Sanitizer: content-token extraction table (upstream-derived) ──────────
# Folds in upstream v2026.7.1's parametrized coverage, with expectations
# corrected for THIS implementation: \w+ tokenization SPLITS on hyphens
# (so "length-probe" -> {"length","probe"}), and all-stopword input falls
# back to quoted word tokens rather than the raw AND-joined string.


@pytest.mark.parametrize(
    "query,expected",
    [
        ("what happened with the deployment rollback", {"happened", "deployment", "rollback"}),
        ("compaction", {"compaction"}),
        ("the and of", {"the", "and", "of"}),
        ("", set()),
        ("context: length-probe", {"context", "length", "probe"}),
        ("hello, world!", {"hello", "world"}),
    ],
)
def test_sanitize_fts_query_extracts_content_tokens(query, expected):
    result = FactRetriever._sanitize_fts_query(query)
    if not expected:
        assert result == ""
    else:
        assert _quoted_terms(result) == expected


@pytest.mark.parametrize(
    "query",
    [
        'context: length-probe',
        '"""',
        "foo -bar",
        "(alpha OR beta)",
        "test^2 query",
        "test:colon query",
        "!!!",
        "a" * 1000,
    ],
)
def test_sanitize_fts_query_never_crashes_on_fts5_specials(store, query):
    """Sanitizer + the store search path must never raise on FTS5 specials."""
    store.add_fact("context length probe alpha beta foo bar", category="test")
    assert isinstance(FactRetriever._sanitize_fts_query(query), str)
    # store.search_facts uses the sanitizer and must never raise.
    assert isinstance(store.search_facts(query, limit=5), list)


# ── Integration: FactRetriever.search (the _fts_candidates path) ──────────


class TestRetrieverSearchRecall:
    def test_prose_query_finds_partial_matches(self, store):
        store.add_fact("Deployment finished successfully", category="work")
        store.add_fact("Rollback was cancelled after checks", category="work")
        store.add_fact("Breakfast anchor works in the morning", category="health")

        retriever = FactRetriever(store, hrr_weight=0)
        # No single fact contains all of these words — raw AND-MATCH would
        # return zero. OR-sanitized MATCH must surface the relevant ones.
        results = retriever.search("what happened with deployment rollback", limit=10)

        contents = [r["content"] for r in results]
        assert any("Deployment" in c for c in contents)
        assert any("Rollback" in c for c in contents)

    def test_operator_only_query_returns_empty_no_error(self, store):
        store.add_fact("Deployment finished successfully", category="work")
        retriever = FactRetriever(store, hrr_weight=0)
        # Must not raise an FTS5 syntax error.
        assert retriever.search("!!!", limit=10) == []

    def test_single_keyword_still_works(self, store):
        store.add_fact("Compaction settings tuned to 0.85 threshold.", category="tool")
        retriever = FactRetriever(store, hrr_weight=0)
        results = retriever.search("compaction", limit=10)
        assert results
        assert any("compaction" in r["content"].lower() for r in results)

    def test_stopword_only_query_does_not_crash(self, store):
        store.add_fact("Deployment finished successfully", category="work")
        retriever = FactRetriever(store, hrr_weight=0)
        # Stopword-only falls back to quoted tokens; search stays a list.
        assert _quoted_terms(FactRetriever._sanitize_fts_query("the and of")) == {
            "the", "and", "of"
        }
        assert isinstance(retriever.search("the and of", limit=10), list)


# ── Integration: MemoryStore.search_facts (the sibling path) ──────────────


class TestStoreSearchFactsRecall:
    def test_prose_query_finds_partial_matches(self, store):
        store.add_fact("Deployment finished successfully", category="work")
        store.add_fact("Rollback was cancelled after checks", category="work")

        results = store.search_facts(
            "what happened with deployment rollback", limit=10
        )
        contents = [r["content"] for r in results]
        assert any("Deployment" in c for c in contents)
        assert any("Rollback" in c for c in contents)

    def test_operator_only_query_returns_empty_no_error(self, store):
        store.add_fact("Deployment finished successfully", category="work")
        # Previously this handed FTS5 an empty/garbage MATCH and raised.
        assert store.search_facts("!!!", limit=10) == []


# ── Learning signal: retrieval_count bump on returned facts ───────────────


class TestMarkRetrieved:
    def test_search_increments_retrieval_count_on_returned_facts(self, store):
        store.add_fact("Unique deployment rollback marker", category="work")

        retriever = FactRetriever(store, hrr_weight=0)
        before = store.search_facts("deployment", limit=5)
        # search_facts itself bumps count; capture, then use retriever.search.
        base = before[0]["retrieval_count"] if before else 0

        results = retriever.search("deployment rollback marker", limit=5)
        assert results  # found it

        after = store.search_facts("deployment", limit=5)
        # retriever.search's _mark_retrieved plus the two search_facts calls
        # should have advanced the counter beyond the initial baseline.
        assert after[0]["retrieval_count"] > base
