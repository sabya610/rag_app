from app.utils import split_text
import pytest
from unittest.mock import patch
from app.utils import extract_text_from_pdfs,clean_and_merge_lines


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
def test_split_text(mock_extract_text_pdfminer,tmp_path):
    mock_extract_text_pdfminer.return_value = SAMPLE_TEXT
    dummy_pdf=tmp_path/"sample.pdf"
    dummy_pdf.write_text("fake-pdf-content")
    
    #text = SAMPLE_TEXT
    full_text = extract_text_from_pdfs(str(tmp_path))
    
    #Test to check ```bash` block and Note Markdown
    lines = SAMPLE_TEXT.splitlines()
    merged = clean_and_merge_lines(lines)
    text = "\n".join(merged)
    chunks = split_text(text,chunk_size=100)
    pytest.set_trace()
    assert all(len(c) <= 100 for c in chunks)
    
    cli_block = [c for c in chunks if "```bash" in c]
    assert cli_block, "At least one chunk should contain cli commands"
    
    # Ensure note lines are properly formatted
    note_lines = [c for c in chunks if "**Note:**" in c or "> **Note:" in c]
    assert note_lines, "Note lines should be formatted with Markdown"