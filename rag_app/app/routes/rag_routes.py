import os
import logging
from flask import Blueprint, request, render_template, current_app, jsonify
from werkzeug.utils import secure_filename
from app.utils import (
    extract_text_from_pdfs_single, split_text,
    retrieve_relevant_chunks_pg, load_embeddings_to_pg,
    llm_answer, overlapping_chunks
)
from app.models import KBChunk, QAHist, SFDCArticle, db
from app.config import Config

logger = logging.getLogger(__name__)



rag_bp = Blueprint("rag", __name__)
PDF_FOLDER = Config.PDF_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'


@rag_bp.route("/", methods=["GET","POST"])
def index():
    answer = ""
    html_response = ""
    source_used = ""

    embedder = current_app.embedder
    llama = current_app.llama

    sfdc_available = getattr(current_app, "sfdc_client", None) is not None

    if request.method == "POST":
        question = request.form["question"]
        source_mode = request.form.get("source", "both")  # "pdf", "sfdc", "both"

        chunks = []

        # --- PDF source (existing pgvector retrieval) ---
        if source_mode in ("pdf", "both"):
            pdf_chunks = retrieve_relevant_chunks_pg(question, top_k=Config.MAX_RESULTS)
            chunks.extend(pdf_chunks)

        # --- SFDC live source ---
        sfdc_articles = []
        if source_mode in ("sfdc", "both") and sfdc_available:
            try:
                from app.services.sfdc_knowledge import fetch_articles_for_query, articles_to_chunks
                sf = current_app.sfdc_client
                product_line = request.form.get("product_line", "")
                sfdc_articles = fetch_articles_for_query(sf, question, limit=Config.SFDC_SEARCH_LIMIT, product_line=product_line)
                if sfdc_articles:
                    sfdc_chunks = articles_to_chunks(sfdc_articles)
                    chunks.extend([c["text"] for c in sfdc_chunks])
                    source_used = "sfdc" if source_mode == "sfdc" else "both"
            except Exception as e:
                logger.error("[SFDC] Live search failed: %s", e)

        if not source_used:
            source_used = source_mode if source_mode == "pdf" else "pdf"

        # --- SFDC-only: return formatted articles directly, skip LLM ---
        if source_mode == "sfdc":
            answer = _format_sfdc_articles(sfdc_articles, question)
        else:
            full_context = "".join(chunks)
            print(f"\n [CONTEXT USED]\n {full_context}")
            answer = llm_answer(full_context, question)

        qa = QAHist(question=question, answer=answer, source=source_used)
        db.session.add(qa)
        db.session.commit()

        print(f"The Answer to be rendered {answer}\n\n")
    return render_template("index.html", answer=answer, sfdc_available=sfdc_available)



@rag_bp.route("/upload", methods=["GET","POST"])
def upload():
    if request.method == "POST":

        embedder = current_app.embedder

        files = request.files.getlist('pdfs')
        all_chunks = []
        for file in files:
            if file and allowed_file(file.filename):
                fname = secure_filename(file.filename)
                fpath = os.path.join(PDF_FOLDER, fname)
                file.save(fpath)
                raw_chunks = extract_text_from_pdfs_single(fpath)
                for doc in raw_chunks:
                    split_chunks = split_text(doc)
                    overlapped = overlapping_chunks(split_chunks, overlap=1)
                    all_chunks.extend(overlapped)
                if all_chunks:
                   load_embeddings_to_pg(all_chunks, fname)
        return "PDF uploaded and processed successfully.", 200
    return render_template("upload.html")



@rag_bp.route("/history")
def history():
    all_qa = QAHist.query.order_by(QAHist.id.desc()).all()
    return render_template("history.html", history=all_qa)


# ============================================================================
# SFDC Knowledge Article Routes
# ============================================================================

@rag_bp.route("/sfdc/status")
def sfdc_status():
    """Check SFDC connection status."""
    sf = getattr(current_app, "sfdc_client", None)
    if sf is None:
        return jsonify({"connected": False, "message": "SFDC not configured"}), 200

    try:
        connected = sf.test_connection()
        return jsonify({
            "connected": connected,
            "message": "Connected" if connected else "Connection failed (session may have expired)",
            "product_filter": Config.SFDC_PRODUCT_QUEUE or "(none - all products)",
            "product_line_filter": Config.SFDC_PRODUCT_LINE or "(none - all lines)",
        }), 200
    except Exception as e:
        return jsonify({"connected": False, "message": str(e)}), 200


@rag_bp.route("/sfdc/describe")
def sfdc_describe():
    """Describe Knowledge__kav fields to help diagnose filter field names."""
    sf = getattr(current_app, "sfdc_client", None)
    if sf is None:
        return jsonify({"error": "SFDC not configured"}), 400

    try:
        desc = sf.describe_sobject("Knowledge__kav")
        # Return only field names/labels containing 'product' or 'queue' for relevance
        fields = []
        for f in desc.get("fields", []):
            name = f.get("name", "")
            label = f.get("label", "")
            if any(kw in name.lower() or kw in label.lower()
                   for kw in ("product", "queue", "group", "line", "category")):
                fields.append({
                    "api_name": name,
                    "label": label,
                    "type": f.get("type", ""),
                    "picklistValues": [
                        v.get("value") for v in f.get("picklistValues", []) if v.get("active")
                    ] if f.get("type") == "picklist" else [],
                })
        return jsonify({
            "total_fields": len(desc.get("fields", [])),
            "product_related_fields": fields,
        }), 200
    except Exception as e:
        logger.error("[SFDC] Describe error: %s", e)
        return jsonify({"error": str(e)}), 500


@rag_bp.route("/sfdc/search", methods=["POST"])
def sfdc_search():
    """Search SFDC Knowledge Articles by keyword or article ID/number."""
    sf = getattr(current_app, "sfdc_client", None)
    if sf is None:
        return jsonify({"error": "SFDC not configured"}), 400

    data = request.get_json() or {}
    article_id_input = data.get("article_id", "").strip()
    query = data.get("query", "").strip()
    product_line = data.get("product_line", "").strip()

    if not query and not article_id_input:
        return jsonify({"error": "query or article_id is required"}), 400

    try:
        # --- Search by Article ID / Number ---
        if article_id_input:
            from app.services.sfdc_knowledge import fetch_article_body
            # Try as ArticleNumber first (e.g. 000114533)
            soql = (
                "SELECT Id, ArticleNumber, Title, Summary, PublishStatus, LastModifiedDate "
                "FROM Knowledge__kav WHERE PublishStatus='Online' AND Language='en_US'"
            )
            if article_id_input.startswith("ka"):
                soql += f" AND Id = '{article_id_input}'"
            else:
                soql += f" AND ArticleNumber = '{article_id_input}'"
            soql += " LIMIT 5"

            resp = sf.query(soql)
            results = resp.get("records", []) if resp else []

            articles = []
            for r in results:
                articles.append({
                    "id": r.get("Id", ""),
                    "article_number": r.get("ArticleNumber", ""),
                    "title": r.get("Title", ""),
                    "summary": r.get("Summary", ""),
                    "publish_status": r.get("PublishStatus", ""),
                    "last_modified": r.get("LastModifiedDate", ""),
                })
            return jsonify({"articles": articles, "count": len(articles)}), 200

        # --- Search by keyword ---
        from app.services.sfdc_knowledge import search_knowledge_articles, search_knowledge_articles_soql
        results = search_knowledge_articles(sf, query, limit=20, product_line=product_line)
        if not results:
            results = search_knowledge_articles_soql(sf, query, limit=20, product_line=product_line)

        articles = []
        for r in results:
            articles.append({
                "id": r.get("Id", ""),
                "article_number": r.get("ArticleNumber", ""),
                "title": r.get("Title", ""),
                "summary": r.get("Summary", ""),
                "publish_status": r.get("PublishStatus", ""),
                "last_modified": r.get("LastModifiedDate", ""),
            })
        return jsonify({"articles": articles, "count": len(articles)}), 200
    except Exception as e:
        logger.error("[SFDC] Search error: %s", e)
        return jsonify({"error": str(e)}), 500


@rag_bp.route("/sfdc/ingest", methods=["POST"])
def sfdc_ingest():
    """
    Ingest selected SFDC Knowledge Articles into the pgvector store.
    Accepts JSON body: {"article_ids": ["id1", "id2", ...]}
    """
    sf = getattr(current_app, "sfdc_client", None)
    if sf is None:
        return jsonify({"error": "SFDC not configured"}), 400

    data = request.get_json() or {}
    article_ids = data.get("article_ids", [])
    if not article_ids:
        return jsonify({"error": "article_ids is required"}), 400

    try:
        from app.services.sfdc_knowledge import fetch_article_body, articles_to_chunks

        articles = []
        for aid in article_ids:
            # Skip if already ingested
            if db.session.query(SFDCArticle).filter_by(article_id=aid).first():
                continue
            article = fetch_article_body(sf, aid)
            if article and (article.get("body") or article.get("summary")):
                articles.append(article)

        if not articles:
            return jsonify({"message": "No new articles to ingest", "ingested": 0}), 200

        chunk_list = articles_to_chunks(articles)
        ingested_count = 0

        for chunk_info in chunk_list:
            load_embeddings_to_pg([chunk_info["text"]], chunk_info["source"])

        # Record ingested articles
        for article in articles:
            record = SFDCArticle(
                article_id=article["id"],
                article_number=article.get("article_number", ""),
                title=article.get("title", ""),
            )
            db.session.add(record)
            ingested_count += 1

        db.session.commit()

        return jsonify({
            "message": f"Ingested {ingested_count} articles ({len(chunk_list)} chunks)",
            "ingested": ingested_count,
        }), 200

    except Exception as e:
        logger.error("[SFDC] Ingest error: %s", e)
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@rag_bp.route("/sfdc/reconnect", methods=["POST"])
def sfdc_reconnect():
    """Re-authenticate SFDC (useful when session ID expires)."""
    try:
        from app.services.sfdc_client import get_sfdc_client
        sf = get_sfdc_client()
        if sf and sf.test_connection():
            current_app.sfdc_client = sf
            return jsonify({"connected": True, "message": "Reconnected successfully"}), 200
        else:
            current_app.sfdc_client = None
            return jsonify({"connected": False, "message": "Failed to reconnect"}), 200
    except Exception as e:
        return jsonify({"connected": False, "message": str(e)}), 500


# ============================================================================
# JSON /ask endpoint — used by Slack bot and API clients
# ============================================================================

@rag_bp.route("/ask", methods=["POST"])
def ask():
    """
    JSON endpoint: {"question": "...", "source": "both|pdf|sfdc", "product_line": ""}
    Returns: {"answer": "...", "source": "..."}
    """
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    source_mode = data.get("source", "both")
    product_line = data.get("product_line", "")
    sfdc_available = getattr(current_app, "sfdc_client", None) is not None

    chunks = []
    sfdc_articles = []
    source_used = source_mode

    if source_mode in ("pdf", "both"):
        chunks.extend(retrieve_relevant_chunks_pg(question, top_k=Config.MAX_RESULTS))

    if source_mode in ("sfdc", "both") and sfdc_available:
        try:
            from app.services.sfdc_knowledge import fetch_articles_for_query, articles_to_chunks
            sf = current_app.sfdc_client
            sfdc_articles = fetch_articles_for_query(sf, question, limit=Config.SFDC_SEARCH_LIMIT, product_line=product_line)
            if sfdc_articles:
                chunks.extend([c["text"] for c in articles_to_chunks(sfdc_articles)])
        except Exception as e:
            logger.error("[SFDC] /ask search failed: %s", e)

    if source_mode == "sfdc":
        answer = _format_sfdc_articles(sfdc_articles, question)
    else:
        answer = llm_answer("".join(chunks), question)

    try:
        qa = QAHist(question=question, answer=answer, source=source_used)
        db.session.add(qa)
        db.session.commit()
    except Exception:
        db.session.rollback()

    return jsonify({"answer": answer, "source": source_used}), 200


# ============================================================================
# Helpers
# ============================================================================

def _format_sfdc_articles(articles, question):
    """
    Format SFDC articles as readable text without LLM.
    Called when source_mode == 'sfdc' to skip LLM inference.
    """
    if not articles:
        return "No SFDC Knowledge Articles found for your query."

    lines = [f"**SFDC Knowledge Articles for: {question}**\n"]
    for i, art in enumerate(articles, 1):
        title = art.get("title", "Untitled")
        num = art.get("article_number", "")
        summary = (art.get("summary") or "").strip()
        body = (art.get("body") or "").strip()

        lines.append(f"---\n### {i}. {title}")
        if num:
            lines.append(f"**Article #:** {num}")
        if summary:
            lines.append(f"\n**Summary:**\n{summary}")
        if body:
            # Truncate body to 1500 chars per article to keep response readable
            body_preview = body[:1500] + ("..." if len(body) > 1500 else "")
            lines.append(f"\n**Details:**\n{body_preview}")
        lines.append("")

    return "\n".join(lines)
