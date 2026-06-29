"""
Tests for performance fixes:
  1. clean_context() truncates to MAX_CONTEXT_CHARS
  2. llm_answer() uses max_tokens=512
  3. fetch_articles_for_query() fetches in parallel via ThreadPoolExecutor
"""
import pytest
from unittest.mock import patch, MagicMock, call
from concurrent.futures import Future


# ─── Test 1: clean_context truncates to MAX_CONTEXT_CHARS ───────────────────

def test_clean_context_truncates_long_text():
    """Context longer than MAX_CONTEXT_CHARS must be truncated."""
    from app.utils import clean_context

    long_text = "A" * 20000  # longer than default 12000

    with patch("app.config.Config.MAX_CONTEXT_CHARS", 12000):
        result = clean_context(long_text)

    assert len(result) <= 12000, f"Expected <=12000 chars, got {len(result)}"


def test_clean_context_short_text_not_truncated():
    """Context shorter than MAX_CONTEXT_CHARS must not be modified."""
    from app.utils import clean_context

    short_text = "Line one\nLine two\nLine three"

    with patch("app.config.Config.MAX_CONTEXT_CHARS", 12000):
        result = clean_context(short_text)

    assert "Line one" in result
    assert "Line two" in result


def test_clean_context_deduplicates_lines():
    """Duplicate lines must still be removed before truncation."""
    from app.utils import clean_context

    text = "same line\nsame line\nsame line\ndifferent"

    with patch("app.config.Config.MAX_CONTEXT_CHARS", 12000):
        result = clean_context(text)

    assert result.count("same line") == 1


# ─── Test 2: llm_answer uses max_tokens=512 ─────────────────────────────────

def test_llm_answer_uses_max_tokens_512():
    """llm_answer must call llama with max_tokens=512."""
    from app import utils

    mock_llama = MagicMock()
    mock_llama.return_value = {"choices": [{"text": "Step 1: do something"}]}

    mock_app = MagicMock()
    mock_app.llama = mock_llama

    with patch("app.utils.current_app", mock_app), \
         patch("app.utils.clean_context", side_effect=lambda x: x):
        utils.llm_answer("some context", "some question")

    call_kwargs = mock_llama.call_args
    assert call_kwargs.kwargs.get("max_tokens") == 512, (
        f"Expected max_tokens=512, got {call_kwargs.kwargs.get('max_tokens')}"
    )


# ─── Test 3: fetch_articles_for_query fetches in parallel ───────────────────

def test_fetch_articles_parallel():
    """fetch_articles_for_query must use ThreadPoolExecutor, not sequential loop."""
    from app.services.sfdc_knowledge import fetch_articles_for_query

    mock_sf = MagicMock()

    # SOSL returns 3 article stubs
    mock_records = [
        {"Id": "id1", "Title": "NODE ALARM CORE PRESENT", "Summary": "core dump issue"},
        {"Id": "id2", "Title": "Core files present alarm", "Summary": "core alarm"},
        {"Id": "id3", "Title": "MFS core alarm", "Summary": "mfs core"},
    ]

    mock_article = {
        "id": "id1", "title": "NODE ALARM CORE PRESENT",
        "summary": "core dump", "body": "Resolution: run gdb",
        "article_number": "000001", "url_name": "node-alarm",
        "publish_status": "Online", "last_modified": "2026-01-01", "version": "1",
    }

    with patch("app.services.sfdc_knowledge.search_knowledge_articles", return_value=mock_records), \
         patch("app.services.sfdc_knowledge.fetch_article_body", return_value=mock_article) as mock_fetch, \
         patch("app.services.sfdc_knowledge.ThreadPoolExecutor", wraps=__import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor) as mock_executor:

        articles = fetch_articles_for_query(mock_sf, "NODE ALARM CORE PRESENT", limit=5)

    # All 3 articles should be fetched
    assert len(articles) == 3
    # fetch_article_body should have been called 3 times (once per article)
    assert mock_fetch.call_count == 3
    # ThreadPoolExecutor should have been used (not a sequential loop)
    assert mock_executor.called, "ThreadPoolExecutor must be used for parallel fetch"


def test_fetch_articles_handles_fetch_failure_gracefully():
    """If one article fetch fails, others should still be returned."""
    from app.services.sfdc_knowledge import fetch_articles_for_query

    mock_sf = MagicMock()
    mock_records = [
        {"Id": "id1", "Title": "Good article", "Summary": "ok"},
        {"Id": "id2", "Title": "Bad article", "Summary": "fails"},
    ]

    def side_effect(sf, article_id):
        if article_id == "id2":
            raise Exception("Network timeout")
        return {"id": "id1", "title": "Good article", "summary": "ok",
                "body": "Some body", "article_number": "1",
                "url_name": "good", "publish_status": "Online",
                "last_modified": "2026-01-01", "version": "1"}

    with patch("app.services.sfdc_knowledge.search_knowledge_articles", return_value=mock_records), \
         patch("app.services.sfdc_knowledge.fetch_article_body", side_effect=side_effect):

        articles = fetch_articles_for_query(mock_sf, "test query", limit=5)

    # Only the successful article should be returned
    assert len(articles) == 1
    assert articles[0]["id"] == "id1"
