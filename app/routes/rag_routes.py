import os
from flask import Blueprint, request, render_template,current_app
from werkzeug.utils import secure_filename
from app.utils import (
    extract_text_from_pdfs_single, split_text,
    retrieve_relevant_chunks_pg, load_embeddings_to_pg,
    llm_answer, overlapping_chunks
)
from app.models import KBChunk, QAHist, db
from app.config import Config



rag_bp = Blueprint("rag", __name__)
PDF_FOLDER = Config.PDF_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'


@rag_bp.route("/", methods=["GET","POST"])
def index():
    answer = ""
    html_response = ""

    embedder = current_app.embedder

    llama = current_app.llama


    if request.method == "POST":
        question = request.form["question"]

        chunks = retrieve_relevant_chunks_pg(question, top_k=50)

        full_context = "".join(chunks)

        print(f"\n [CONTEXT USED]\n {full_context}")

        answer = llm_answer(full_context, question)

        qa = QAHist(question=question, answer=answer)
        db.session.add(qa)
        db.session.commit()


        print(f"The Answer to be rendered {answer}\n\n")
    return render_template("index.html", answer=answer)



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