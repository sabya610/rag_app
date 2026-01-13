from fastapi import FastAPI, HTTPException
from pathlib import Path

from rag_meeting_app.llm.model_loader import load_models
from rag_meeting_app.llm.llm_client import LLMClient
from rag_meeting_app.llm.llm_service import LLMService
from rag_meeting_app.embeddings.vector_store import VectorStore
from rag_meeting_app.ingestion.batch_loader import ingest_folder
from rag_meeting_app.ingestion.indexer import index_issues
from rag_meeting_app.app.query import query_issues

app = FastAPI()

# ---- Global state (intentional) ----
embedder = None
llm_service = None
vector_store = None


@app.on_event("startup")
def startup_event():
    global embedder, llm_service, vector_store

    embedder, llama = load_models()
    llm_service = LLMService(LLMClient(llama))
    vector_store = VectorStore()


@app.post("/ingest")
def ingest():
    if llm_service is None:
        raise HTTPException(status_code=500, detail="LLM not initialized")

    transcripts_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw_transcripts"
    )

    issues = ingest_folder(str(transcripts_path), llm_service)
    index_issues(issues, embedder, vector_store)

    return {"indexed_issues": len(issues)}


@app.get("/query")
def query(q: str):
    if not vector_store or not vector_store.store:
        raise HTTPException(status_code=400, detail="No issues indexed yet")

    results = query_issues(q, embedder, vector_store)
    return {"results": results}
