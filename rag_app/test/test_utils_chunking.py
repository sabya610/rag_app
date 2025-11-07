import pytest
from unittest.mock import patch
from app.utils import extract_text_from_pdfs_single

SAMPLE_TEXT="""
Issue: Kubernetes certificate expired
Cause: Certificate was not renewed automatically
Resolution: 
Step 1: Check certificate expiration
#kubectl get csr
Step 2: Renew certificate
#sudo kubeadm cert renew all
Note: Restart control plane pods
"""

@patch("app.utils.extract_text_pdfminer")
def test_extract_text_from_pdfs_single(mock_extract_text_pdfminer,tmp_path):
    mock_extract_text_pdfminer.return_value = SAMPLE_TEXT
    dummy_pdf=tmp_path/"sample.pdf"
    dummy_pdf.write_text("fake-pdf-content")
    
    chunks = extract_text_from_pdfs_single(str(dummy_pdf))
    
    #-----------Assertions-----------------
    # Ensure the function returns multiple chunks
    assert isinstance(chunks,list)
    assert len(chunks) > 0
    
    #Check if major section are preserved
    section_found = any("#Resolution" in c or "Step 1" in c for c in chunks)
    assert section_found , "Section Header should be preserved in chunks"
    
    # Verify command lines inside fenced bash block
    cli_blocks=[c for c in chunks if "```bash" in c]
    assert cli_blocks, "At least one chunk should contain CLI commands"
    
    # Ensure note lines are properly formatted
    note_lines = [c for c in chunks if "**Note:**" in c or "> **Note:" in c]
    assert note_lines, "Note lines should be formatted with Markdown"
    
    #No Duplicates entry
    unique_chunks = set(chunks)
    assert len(unique_chunks) == len(chunks)
    assert all(len(c.strip()) > 5 for c in chunks)
    assert all(20 < len(c) < 2000 for c in chunks)
    
    
    
    