#Install Postgresql for pgvector connector to store embeddings
#pip install flask flask_sqlalchemy
#pip install psycopg2-binary sqlalchemy pgvector 
#pip install flask_sqlalchemy
#pip install python-dotenv
#Gunicorn with worker limits avoids Flask dev server overhead:
#gunicorn -w 2 -b 127.0.0.1:5000 app:app

from urllib.parse import quote_plus
from sqlalchemy import cast
from pgvector.sqlalchemy import Vector
from sqlalchemy.sql import text as sa_text
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import VARCHAR
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from llama_cpp import Llama


import re
import os
import numpy as np
from dotenv import load_dotenv


# Configuration

PDF_FOLDER = "\ere_kb_tool\pdf_kb_files"
MODEL_PATH = "\models\llama-2-7b-chat.Q4_K_M.gguf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PDF_FOLDER = "/pdf_kb_files"
MODEL_PATH = "/models/llama-2-7b-chat.Q4_K_M.gguf"
EMBEDDING_MODEL = "/models/embedding/all-MiniLM-L6-v2"
PDF_FOLDER = "/app/pdf_kb_files"
MODEL_PATH = "/app/models/llama-2-7b-chat.Q4_K_M.gguf"
#EMBEDDING_MODEL = "/app/models/embedding/all-MiniLM-L6-v2"
EMBEDDING_MODEL = "/app/models/embedding/all-MiniLM-L6-v2"
PGVECTOR_DIM = 384
MAX_RESULTS = 3

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
print(f"Connecting to DB at postgresql://{DB_USER}:<hidden>@{DB_HOST}:{DB_PORT}/{DB_NAME}")
db = SQLAlchemy(app)
print(f"Connecting to DB at postgresql://{DB_USER}:<hidden>@{DB_HOST}:{DB_PORT}/{DB_NAME}")



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
    import PyPDF2
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
### Context:
{context}

### Question:
    {question}

### Answer:"""

    output = llama(prompt=prompt, max_tokens=512, stop=["###"])
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
      n_ctx=4096,
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

if __name__ == "__main__":
    app.run(debug=False)
