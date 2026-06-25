"""
SFDC Knowledge Article Service for RAG App.

Searches and retrieves Knowledge Articles from Salesforce,
processes them into chunks, and indexes them into pgvector
for RAG retrieval.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import html2text

from app.services.sfdc_client import SalesforceClient, SalesforceError

logger = logging.getLogger(__name__)

# html2text converter for cleaning SFDC rich-text fields
_h2t = html2text.HTML2Text()
_h2t.ignore_links = False
_h2t.ignore_images = True
_h2t.body_width = 0  # no wrapping


def html_to_text(html_content: str) -> str:
    """Convert HTML to clean markdown/plain text."""
    if not html_content:
        return ""
    text = _h2t.handle(html_content)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def search_knowledge_articles(
    sf: SalesforceClient,
    search_query: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Search published Knowledge Articles by keyword using SOSL.

    Returns list of article summaries (Id, Title, ArticleNumber, Summary).
    """
    # Escape special SOSL characters
    safe_query = re.sub(r"[&|!{}[\]()^~*?:\\\"']", " ", search_query).strip()
    if not safe_query:
        return []

    sosl = (
        f"FIND {{{safe_query}}} IN ALL FIELDS "
        f"RETURNING Knowledge__kav("
        f"Id, KnowledgeArticleId, ArticleNumber, Title, Summary, "
        f"UrlName, PublishStatus, LastModifiedDate, VersionNumber "
        f"WHERE PublishStatus = 'Online' "
        f"ORDER BY LastModifiedDate DESC "
        f"LIMIT {int(limit)})"
    )

    try:
        results = sf.search(sosl)
        logger.info("[SFDC-KA] SOSL search returned %d results for '%s'", len(results), safe_query)
        return results
    except SalesforceError as e:
        logger.error("[SFDC-KA] SOSL search failed: %s", e)
        return []


def search_knowledge_articles_soql(
    sf: SalesforceClient,
    search_terms: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Search published Knowledge Articles using SOQL LIKE matching.
    Fallback when SOSL is unavailable or for more targeted queries.
    """
    # Split into keywords and build LIKE clauses
    keywords = [w.strip() for w in search_terms.split() if len(w.strip()) >= 3]
    if not keywords:
        return []

    like_clauses = " AND ".join(
        f"(Title LIKE '%{kw}%' OR Summary LIKE '%{kw}%')" for kw in keywords[:5]
    )

    soql = (
        f"SELECT Id, KnowledgeArticleId, ArticleNumber, Title, Summary, "
        f"UrlName, PublishStatus, LastModifiedDate, VersionNumber "
        f"FROM Knowledge__kav "
        f"WHERE PublishStatus = 'Online' AND {like_clauses} "
        f"ORDER BY LastModifiedDate DESC "
        f"LIMIT {int(limit)}"
    )

    try:
        result = sf.query_all(soql)
        records = result.get("records", [])
        logger.info("[SFDC-KA] SOQL search returned %d results", len(records))
        return records
    except SalesforceError as e:
        logger.error("[SFDC-KA] SOQL search failed: %s", e)
        return []


def fetch_article_body(sf: SalesforceClient, article_id: str) -> Dict[str, Any]:
    """
    Fetch full Knowledge Article content by record Id.

    Returns dict with title, summary, body (as plain text), and metadata.
    """
    try:
        record = sf.get_sobject("Knowledge__kav", article_id)
    except SalesforceError as e:
        logger.error("[SFDC-KA] Failed to fetch article %s: %s", article_id, e)
        return {}

    # Extract body - SFDC KA objects may use different body field names
    body_html = (
        record.get("Answer__c")
        or record.get("ArticleBody__c")
        or record.get("Content__c")
        or record.get("Solution__c")
        or ""
    )

    body_text = html_to_text(body_html)
    summary_text = html_to_text(record.get("Summary", "") or "")

    return {
        "id": record.get("Id", ""),
        "article_number": record.get("ArticleNumber", ""),
        "title": record.get("Title", ""),
        "summary": summary_text,
        "body": body_text,
        "url_name": record.get("UrlName", ""),
        "publish_status": record.get("PublishStatus", ""),
        "last_modified": record.get("LastModifiedDate", ""),
        "version": record.get("VersionNumber", ""),
    }


def fetch_articles_for_query(
    sf: SalesforceClient,
    user_query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search SFDC for Knowledge Articles matching user_query and fetch their full content.
    Tries SOSL first, falls back to SOQL LIKE.
    """
    # Try SOSL first (full-text search)
    results = search_knowledge_articles(sf, user_query, limit=limit)

    # Fallback to SOQL LIKE if SOSL returns nothing
    if not results:
        results = search_knowledge_articles_soql(sf, user_query, limit=limit)

    if not results:
        logger.info("[SFDC-KA] No articles found for query: %s", user_query)
        return []

    articles = []
    for record in results:
        article_id = record.get("Id", "")
        if not article_id:
            continue
        article = fetch_article_body(sf, article_id)
        if article and (article.get("body") or article.get("summary")):
            articles.append(article)

    logger.info("[SFDC-KA] Fetched %d articles with content", len(articles))
    return articles


def articles_to_chunks(articles: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert fetched SFDC articles into text chunks suitable for RAG.

    Each chunk includes metadata about the source article.
    Returns list of dicts with 'text' and 'source' keys.
    """
    chunks = []
    for article in articles:
        title = article.get("title", "Untitled")
        article_num = article.get("article_number", "")
        summary = article.get("summary", "")
        body = article.get("body", "")

        source_label = f"SFDC_KA_{article_num}" if article_num else f"SFDC_KA_{article.get('id', 'unknown')}"

        # Build full text with clear structure
        parts = [f"## {title}"]
        if article_num:
            parts.append(f"Knowledge Article: {article_num}")
        if summary:
            parts.append(f"\n### Summary\n{summary}")
        if body:
            parts.append(f"\n### Content\n{body}")

        full_text = "\n".join(parts)

        # Split into manageable chunks (reuse similar logic to PDF chunking)
        text_chunks = _split_article_text(full_text, chunk_size=900)

        for chunk_text in text_chunks:
            if len(chunk_text.strip()) >= 10:
                chunks.append({"text": chunk_text, "source": source_label})

    return chunks


def _split_article_text(text: str, chunk_size: int = 900) -> List[str]:
    """Split article text into chunks, respecting section boundaries."""
    lines = text.strip().split("\n")
    chunks, buf, in_code = [], "", False

    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
        if line.startswith("## ") and not in_code:
            if buf.strip():
                chunks.append(buf.strip())
            buf = line + "\n"
        elif len(buf) + len(line) < chunk_size or in_code:
            buf += line + "\n"
        else:
            if buf.strip():
                chunks.append(buf.strip())
            buf = line + "\n"

    if buf.strip():
        chunks.append(buf.strip())

    return [c for c in chunks if len(c.strip()) >= 10]
