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
        if source_mode in ("sfdc", "both") and sfdc_available:
            try:
                from app.services.sfdc_knowledge import fetch_articles_for_query, articles_to_chunks
                sf = current_app.sfdc_client
                articles = fetch_articles_for_query(sf, question, limit=Config.SFDC_SEARCH_LIMIT)
                if articles:
                    sfdc_chunks = articles_to_chunks(articles)
                    chunks.extend([c["text"] for c in sfdc_chunks])
                    source_used = "sfdc" if source_mode == "sfdc" else "both"
            except Exception as e:
                logger.error("[SFDC] Live search failed: %s", e)

        if not source_used:
            source_used = source_mode if source_mode == "pdf" else "pdf"

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
        }), 200
    except Exception as e:
        return jsonify({"connected": False, "message": str(e)}), 200


@rag_bp.route("/sfdc/search", methods=["POST"])
def sfdc_search():
    """Search SFDC Knowledge Articles by keyword (returns article list, does NOT ingest)."""
    sf = getattr(current_app, "sfdc_client", None)
    if sf is None:
        return jsonify({"error": "SFDC not configured"}), 400

    data = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        from app.services.sfdc_knowledge import search_knowledge_articles, search_knowledge_articles_soql
        results = search_knowledge_articles(sf, query, limit=20)
        if not results:
            results = search_knowledge_articles_soql(sf, query, limit=20)

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