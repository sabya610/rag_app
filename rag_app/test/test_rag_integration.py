# tests/test_rag_integration.py
import pytest
from unittest.mock import patch, MagicMock
from app.rag_pipeline import RAGPipeline  #  wrapper class
from app.utils import extract_text_from_pdfs_single
#pytest.set_trace()


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

@patch("app.rag_pipeline.extract_text_from_pdfs_single")
@patch("app.rag_pipeline.SentenceTransformer")
@patch("app.rag_pipeline.query_model")  #  llama.cpp wrapper
def test_rag_pipeline_end_to_end(mock_llm, mock_st, mock_pdfminer, sample_pdf):
    # ---------------- Mock Setup ----------------
    #Mock PDF Extraction
    mock_pdfminer.return_value = SAMPLE_PDF_TEXT
    
    #Mock SentenceTransformer instance
    mock_embedder_instance=MagicMock()
    mock_embedder_instance.encode.side_effect = lambda texts: [[float(len(t))] for t in texts] # embedding
    mock_st.return_value = mock_embedder_instance
    
    
    #Mock LLM response
    mock_llm.return_value = "You can renew Kubernetes certificates using `kubeadm cert renew all`"

    # ---------------- Pipeline Execution ----------------
    rag = RAGPipeline()  # orchestrator class
    rag.ingest_pdf(str(sample_pdf))
    #pytest.set_trace() 
    response = rag.query("How to renew Kubernetes certificate?")
    #pytest.set_trace()
    # ---------------- Assertions ----------------
    assert "renew" in response.lower()
    assert "kubeadm" in response.lower()
    assert mock_st.called, "Embeddings should be generated for chunks"
    assert mock_llm.called, "LLM should be invoked with context + query"
    assert len(rag.vector_store) > 0, "Embeddings should be stored in vector store"

    print("RAG Pipeline Test Passed ")
