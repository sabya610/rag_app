from fastapi import FastAPI, HTTPException, BackgroundTasks
from pathlib import Path

from rag_meeting_app.llm.model_loader import load_models
from rag_meeting_app.llm.llm_client import LLMClient
from rag_meeting_app.llm.llm_service import LLMService
from rag_meeting_app.embeddings.vector_store import VectorStore

from rag_meeting_app.ingestion.batch_loader import ingest_folder
from rag_meeting_app.ingestion.indexer import index_chunks
from rag_meeting_app.app.query import query_issues

app = FastAPI()

embedder = None
llm_service = None
vector_store = None


@app.on_event("startup")
def startup_event():
    global embedder, llm_service, vector_store

    print("Loading embedding model...")
    embedder, llama = load_models()

    print("Loading LLM model...")
    llm_service = LLMService(LLMClient(llama))

    vector_store = VectorStore()


@app.post("/ingest")
def ingest(background_tasks: BackgroundTasks):

    transcripts_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "transcripts"
    )

    background_tasks.add_task(run_ingestion, transcripts_path)

    return {"status": "Chunk ingestion started"}


def run_ingestion(path):
    global embedder, vector_store

    try:
        print("Background ingestion running...")

        chunks = ingest_folder(str(path))
        print("Total chunks extracted:", len(chunks))

        index_chunks(chunks, embedder, vector_store)

        print("Background ingestion finished.")

    except Exception as e:
        print("INGESTION FAILED:", str(e))


@app.get("/ask")
def ask(q: str):
    results = query_issues(q, embedder, vector_store)

    if not results:
        return {"answer": "No indexed context available. Run /ingest first."}

    context = "\n\n".join([r["text"] for r in results])
    context = context[:3000]

    prompt = f"""
You are a meeting assistant.

Context from meeting transcripts:
{context}

Question: {q}

Answer clearly with:
- Summary
- Action items
- Owners (if known)
"""

    answer = llm_service.llm.generate(prompt)

    return {"answer": answer, "sources": results}


@app.get("/status")
def status():
    return {
        "indexed": len(vector_store.store) if vector_store else 0
    }
