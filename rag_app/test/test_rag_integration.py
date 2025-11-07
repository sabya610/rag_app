# tests/test_rag_integration.py
import pytest
from unittest.mock import patch, MagicMock
from app.rag_pipeline import RAGPipeline  # hypothetical wrapper class
from app.utils import extract_text_from_pdfs_single

SAMPLE_PDF_TEXT = """
Issue: Kubernetes certificate expired
Cause: Certificate was not renewed automatically
Resolution:
Step 1: Check certificate expiration
#kubectl get csr
Step 2: Renew certificate
#sudo kubeadm cert renew all
Note: Restart control plane pods
"""

@pytest.fixture
def sample_pdf(tmp_path):
    pdf_path = tmp_path / "k8s_cert.pdf"
    pdf_path.write_text("fake-pdf-content")
    return pdf_path

@patch("app.utils.extract_text_pdfminer")
@patch("sentence_transformers.SentenceTransformer.encode")
@patch("app.llm_client.query_model")  # this could be your llama.cpp wrapper
def test_rag_pipeline_end_to_end(mock_llm, mock_encode, mock_pdfminer, sample_pdf):
    # ---------------- Mock Setup ----------------
    mock_pdfminer.return_value = SAMPLE_PDF_TEXT
    mock_encode.side_effect = lambda texts: [[float(len(t))] for t in texts]  # deterministic embedding
    mock_llm.return_value = "You can renew Kubernetes certificates using `kubeadm cert renew all`"

    # ---------------- Pipeline Execution ----------------
    rag = RAGPipeline()  # your main orchestrator class
    rag.ingest_pdf(str(sample_pdf))
    response = rag.query("How to renew Kubernetes certificate?")

    # ---------------- Assertions ----------------
    assert "renew" in response.lower()
    assert "kubeadm" in response.lower()
    assert mock_encode.called, "Embeddings should be generated for chunks"
    assert mock_llm.called, "LLM should be invoked with context + query"
    assert len(rag.vector_store) > 0, "Embeddings should be stored in vector store"

    print("RAG Pipeline Test Passed ")
