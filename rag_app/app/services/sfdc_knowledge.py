"""
SFDC Knowledge Article Service for RAG App.

Searches and retrieves Knowledge Articles from Salesforce,
processes them into chunks, and indexes them into pgvector
for RAG retrieval.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# Product line presets for quick switching
PRODUCT_LINE_PRESETS = {
    "ezmeral": {"queue": "HPE Ezmeral", "line": "CONT PLT SW (RM)"},
    "datafabric": {"queue": "HPE Ezmeral", "line": "DATA FABRIC SW (PU)"},
    "all": {"queue": "HPE Ezmeral", "line": ""},
}


def search_knowledge_articles(
    sf: SalesforceClient,
    search_query: str,
    limit: int = 20,
    product_line: str = "",
) -> List[Dict[str, Any]]:
    """
    Search published Knowledge Articles by keyword using SOSL.
    Filters by GSD_KM_Issue_Solution_Product_Queue__c to only return HPE Ezmeral articles.

    Args:
        product_line: preset key ("ezmeral", "datafabric", "all") or raw product line value.
    Returns list of article summaries (Id, Title, ArticleNumber, Summary).
    """
    from app.config import Config

    # Escape special SOSL characters
    safe_query = re.sub(r"[&|!{}[\]()^~*?:\\\"']", " ", search_query).strip()
    if not safe_query:
        return []

    # Resolve product line preset or use config defaults
    preset = PRODUCT_LINE_PRESETS.get(product_line)
    preset_key = product_line if preset else ""
    if preset:
        p_queue = preset["queue"]
        p_line = preset["line"]
    else:
        p_queue = Config.SFDC_PRODUCT_QUEUE
        p_line = product_line if product_line else Config.SFDC_PRODUCT_LINE

    # Build product filter clauses
    product_filter = ""
    if p_queue:
        product_filter += f" AND GSD_KM_Issue_Solution_Product_Queue__c = '{p_queue}'"
    if p_line:
        # Data Fabric has legacy KAs with Product_Line__c = NULL.
        # Include NULL only for datafabric preset while still keeping Ezmeral queue filter.
        if preset_key == "datafabric":
            product_filter += (
                f" AND (Product_Line__c = '{p_line}' OR Product_Line__c = NULL)"
            )
        else:
            product_filter += f" AND Product_Line__c = '{p_line}'"

    sosl = (
        f"FIND {{{safe_query}}} IN ALL FIELDS "
        f"RETURNING Knowledge__kav("
        f"Id, KnowledgeArticleId, ArticleNumber, Title, Summary, "
        f"UrlName, PublishStatus, LastModifiedDate, VersionNumber "
        f"WHERE PublishStatus = 'Online'{product_filter} "
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
    product_line: str = "",
) -> List[Dict[str, Any]]:
    """
    Search published Knowledge Articles using SOQL LIKE matching.
    Fallback when SOSL is unavailable or for more targeted queries.
    Filters by GSD_KM_Issue_Solution_Product_Queue__c to only return HPE Ezmeral articles.
    """
    from app.config import Config

    # Split into keywords and build LIKE clauses
    keywords = [w.strip() for w in search_terms.split() if len(w.strip()) >= 3]
    if not keywords:
        return []

    like_clauses = " AND ".join(
        f"(Title LIKE '%{kw}%' OR Summary LIKE '%{kw}%')" for kw in keywords[:5]
    )

    # Resolve product line preset or use config defaults
    preset = PRODUCT_LINE_PRESETS.get(product_line)
    preset_key = product_line if preset else ""
    if preset:
        p_queue = preset["queue"]
        p_line = preset["line"]
    else:
        p_queue = Config.SFDC_PRODUCT_QUEUE
        p_line = product_line if product_line else Config.SFDC_PRODUCT_LINE

    # Build product filter clauses
    product_filter = ""
    if p_queue:
        product_filter += f" AND GSD_KM_Issue_Solution_Product_Queue__c = '{p_queue}'"
    if p_line:
        if preset_key == "datafabric":
            product_filter += (
                f" AND (Product_Line__c = '{p_line}' OR Product_Line__c = NULL)"
            )
        else:
            product_filter += f" AND Product_Line__c = '{p_line}'"

    soql = (
        f"SELECT Id, KnowledgeArticleId, ArticleNumber, Title, Summary, "
        f"UrlName, PublishStatus, LastModifiedDate, VersionNumber "
        f"FROM Knowledge__kav "
        f"WHERE PublishStatus = 'Online'{product_filter} AND {like_clauses} "
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

    # Extract body - SFDC KA objects use different fields per article type:
    #   "How To" articles:       Procedure__c, Procedure_Overview__c
    #   "Troubleshooting" articles: GSD_KM_Issue_Solution_Issue__c,
    #                               GSD_KM_Issue_Solution_Cause__c,
    #                               GSD_KM_Issue_Solution_Resolution__c
    body_html = (
        record.get("Procedure__c")
        or record.get("ArticleBody__c")
        or record.get("Content__c")
        or record.get("Solution__c")
        or ""
    )

    # Also grab the procedure overview if available
    overview_html = record.get("Procedure_Overview__c") or ""
    environment_html = record.get("GSD_KM_Issue_Solution_Environment__c") or ""

    # Troubleshooting article fields (Issue / Cause / Resolution)
    issue_html = record.get("GSD_KM_Issue_Solution_Issue__c") or ""
    cause_html = record.get("GSD_KM_Issue_Solution_Cause__c") or ""
    resolution_html = record.get("GSD_KM_Issue_Solution_Resolution__c") or ""

    body_text = html_to_text(body_html)
    overview_text = html_to_text(overview_html)
    environment_text = html_to_text(environment_html)
    issue_text = html_to_text(issue_html)
    cause_text = html_to_text(cause_html)
    resolution_text = html_to_text(resolution_html)
    summary_text = html_to_text(record.get("Summary", "") or "")

    # Combine all available content for complete article body
    full_body = ""
    if overview_text:
        full_body += overview_text + "\n\n"
    if environment_text:
        full_body += "Environment: " + environment_text + "\n\n"
    if issue_text:
        full_body += "Issue:\n" + issue_text + "\n\n"
    if cause_text:
        full_body += "Cause:\n" + cause_text + "\n\n"
    if resolution_text:
        full_body += "Resolution:\n" + resolution_text + "\n\n"
    if body_text:
        full_body += body_text

    return {
        "id": record.get("Id", ""),
        "article_number": record.get("ArticleNumber", ""),
        "title": record.get("Title", ""),
        "summary": summary_text,
        "body": full_body or body_text,
        "url_name": record.get("UrlName", ""),
        "publish_status": record.get("PublishStatus", ""),
        "last_modified": record.get("LastModifiedDate", ""),
        "version": record.get("VersionNumber", ""),
    }


def fetch_articles_for_query(
    sf: SalesforceClient,
    user_query: str,
    limit: int = 10,
    product_line: str = "",
) -> List[Dict[str, Any]]:
    """
    Search SFDC for Knowledge Articles matching user_query and fetch their full content.
    Tries SOSL first, falls back to SOQL LIKE.
    Article bodies are fetched in parallel to reduce latency.
    """
    # Try SOSL first (full-text search)
    results = search_knowledge_articles(sf, user_query, limit=limit, product_line=product_line)

    # Fallback to SOQL LIKE if SOSL returns nothing
    if not results:
        results = search_knowledge_articles_soql(sf, user_query, limit=limit, product_line=product_line)

    if not results:
        logger.info("[SFDC-KA] No articles found for query: %s", user_query)
        return []

    # Fetch all article bodies in parallel (max 5 workers to avoid rate-limiting)
    article_ids = [r.get("Id", "") for r in results if r.get("Id")]
    articles = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_id = {executor.submit(fetch_article_body, sf, aid): aid for aid in article_ids}
        for future in as_completed(future_to_id):
            try:
                article = future.result()
                if article and (article.get("body") or article.get("summary")):
                    articles.append(article)
            except Exception as e:
                logger.warning("[SFDC-KA] Failed to fetch article %s: %s", future_to_id[future], e)

    # Re-rank articles by keyword relevance to the user's query
    # so the most relevant articles come first in the context
    query_words = set(user_query.lower().split())
    def _relevance(art):
        title = (art.get("title") or "").lower()
        summary = (art.get("summary") or "").lower()
        text = title + " " + summary
        return sum(1 for w in query_words if w in text)
    articles.sort(key=_relevance, reverse=True)

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
