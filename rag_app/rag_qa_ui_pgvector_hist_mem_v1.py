"""
Flask-based Retrieval-Augmented Generation (RAG) application that uses:

PostgreSQL with pgvector to store and search embeddings,

sentence-transformers to generate embeddings,

llama.cpp to generate answers using a local LLM,

and a basic HTML frontend for input/output and QA history.

"""
import argparse
from urllib.parse import quote_plus
from sqlalchemy import cast
from pgvector.sqlalchemy import Vector
from sqlalchemy.sql import text as sa_text
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import VARCHAR
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from werkzeug.utils import secure_filename
from llama_cpp import Llama
import PyPDF2
import re
import os
import numpy as np
from dotenv import load_dotenv


# Configuration
PDF_FOLDER = "/app/pdf_kb_files"
MODEL_PATH = "/app/models/llama-2-7b-chat.Q4_K_M.gguf"
EMBEDDING_MODEL = "/app/models/embedding/all-MiniLM-L6-v2"
PGVECTOR_DIM = 384
MAX_RESULTS = 3
ALLOWED_EXTENSIONS = {'pdf'}

# Read from .env File (Place at project root)
load_dotenv()

DB_USER = os.getenv("DB_USER")
#DB_PASS = os.getenv("DB_PASS")
DB_PASS = quote_plus(os.getenv("DB_PASS"))
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")



# Flask and DB setup
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class KBChunk(db.Model):
    __tablename__ = "kb_chunks"
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    embedding = db.Column(Vector(PGVECTOR_DIM), nullable=False)

class QAHist(db.Model):
    __tablename__ = "qa_history"
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)

with app.app_context():
    db.create_all()

# Utility functions
def extract_text_from_pdfs(pdf_folder):

    text_chunks = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        lines = text.split('\n')
                        paragraph = ""
                        for line in lines:
                            if line.strip() == "":
                                if paragraph:
                                    text_chunks.append(paragraph.strip())
                                    paragraph = ""
                            else:
                                paragraph += line + "\n"
                        if paragraph:
                            text_chunks.append(paragraph.strip())
    return [chunk for chunk in text_chunks if len(chunk) > 40]

def extract_text_from_pdfs_single(filepath):
    text_chunks = []
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                lines = text.split('\n')
                paragraph = ""
                for line in lines:
                    if line.strip() == "":
                        if paragraph:
                            text_chunks.append(paragraph.strip())
                            paragraph = ""
                    else:
                        paragraph += line + "\n"
                if paragraph:
                    text_chunks.append(paragraph.strip())
    return [chunk for chunk in text_chunks if len(chunk) > 40]

def split_text(text, chunk_size=500):
    sentences = text.split('. ')
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < chunk_size:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def load_embeddings_to_pg(chunks, embedder):
    print("Embedding and saving chunks to PostgreSQL")
    for chunk in chunks:
        vec = embedder.encode(chunk).tolist()
        db.session.add(KBChunk(text=chunk, embedding=vec))
    db.session.commit()




@lru_cache(maxsize=1000)
def cached_embedding(text):
    return embedder.encode(text).tolist()


def retrieve_relevant_chunks_pg(query, embedder, top_k=3):
    #query_vec = embedder.encode(query).tolist()
    query_vec = cached_embedding(query)


    sql = sa_text("""
        SELECT text
        FROM kb_chunks
        ORDER BY embedding <-> CAST(:query_vec AS vector)
        LIMIT :top_k
    """)

    results = db.session.execute(sql, {
        "query_vec": query_vec,
        "top_k": top_k
    }).fetchall()
    print("Retrieved Chunks results:\n", results)
    return [r[0] for r in results]


def generate_answer(context_chunks, question, llama):
    context = "\n".join(context_chunks)
    prompt = f"""
You are a senior technical assistant specializing in Kubernetes and HPE Ezmeral Container Platform.

You are answering a troubleshooting or operations question from a platform administrator. 
Your answers are based on internal HPE technical documentation from support PDFs.

Context:
{context}

Question:
{question}

Instructions:
- Use only verified information from the internal HPE PDF documents.
- Include terminal commands in code blocks (```)
- Highlight critical instructions with bold or bullet points.
- Do NOT hallucinate if the document does not cover the query — respond with "Not covered in documentation".
- Provide a clear, step-by-step guide tailored to the Ezmeral Runtime environment.
- If the query is about certificate renewal or expiration, include commands like `kubeadm certs renew all`, `openssl x509 -noout -enddate`, and how to update the ECP config.
- Always mention critical post-renewal actions like restarting static pods, replacing `.kube/config`, and checking ECP UI integration.
- If any special commands like `bdconfig --getk8sc` or Erlang console updates are needed, include them.
- Use proper CLI formatting and avoid generic advice.
"""

    output = llama(prompt=prompt, max_tokens=256, stop=["###"])
    #print(f"The prompt is {prompt}")
    #print(f"COntext is {context}")
    return output["choices"][0]["text"].strip()

# Load llama model and embedder once
try:
   print("[LOADING...] Loading embedder and llama.cpp model")
   embedder = SentenceTransformer(EMBEDDING_MODEL)
   print("[OK] Embedder loaded.")
except Exception as e:
   print(f"[ERROR] failed to load embedder: {e}")

try:
   llama = Llama(
      model_path=MODEL_PATH,
      n_ctx=2048,
      temperature=0.7,
      top_p=0.9,
      repeat_penalty=1.1,
      verbose=False,
      use_mlock=True,
      use_mmap=True,

    )
   print("[OK] LLaMA loaded.")
except Exception as e:
   print(f"[ERROR] loading llama model: {e}")
   raise


# Response formatting
def clean_response(text):
    # Remove markdown-style formatting
    text = re.sub(r'[*#_~]', '', text)
    return text.strip()

# Load docs into pgvector if db is empty
with app.app_context():
    if KBChunk.query.first() is None:
        print("Populating database from PDFs ")
        raw_docs = extract_text_from_pdfs(PDF_FOLDER)
        all_chunks = []
        for doc in raw_docs:
            chunks = split_text(doc)
            all_chunks.extend(chunks)
        load_embeddings_to_pg(all_chunks, embedder)

# Web endpoints
@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        question = request.form["question"]
        top_chunks = retrieve_relevant_chunks_pg(question, embedder, top_k=MAX_RESULTS)
        if not top_chunks:
            return "No relevant data found in the knowledge base."
        answer = generate_answer(top_chunks, question, llama)
        answer = clean_response(answer)
        qa = QAHist(question=question, answer=answer)
        db.session.add(qa)
        db.session.commit()

    return render_template("index.html", answer=answer)

@app.route("/history")
def history():
    all_qa = QAHist.query.order_by(QAHist.id.desc()).all()
    return render_template("history.html", history=all_qa)



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if 'pdfs' not in request.files:
            return "No file part",400
        files = request.files.getlist['pdfs']
        if not files or all(f.filename == '' for f in files):
            return "No files selected", 400
        uploaded_files = []
        all_chunks = []
        for file in files:
           if file and allowed_file(file.filename):
               filename = secure_filename(file.filename)
               filepath = os.path.join(PDF_FOLDER, filename)
               file.save(filepath)

            # Process and embed this new file
            filepath = os.path.join(PDF_FOLDER, filename)
            raw_docs = extract_text_from_pdfs_single(filepath)          
            
            for doc in raw_docs:
                chunks = split_text(doc)
                all_chunks.extend(chunks)
            uploaded_files.append(filename)
            
        if all_chunks:
            load_embeddings_to_pg(all_chunks, embedder)

            return "PDF uploaded and processed successfully.",200
        else:
            return "No valid content found in uploaded PDFs.", 400
    return render_template("upload.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', help='Host address to listen on')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)
