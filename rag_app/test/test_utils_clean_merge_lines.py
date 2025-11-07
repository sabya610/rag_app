from app.utils import clean_and_merge_lines

def test_clean_and_merge_lines():
        lines = [
        "Issue: Something failed",
        "kubectl get pods",
        "Resolution:",
        "Step 1: Restart pod",
        "sudo systemctl restart kubelet",
    ]
        
        chunks = clean_and_merge_lines(lines)
        assert any("#Resolution" in c for c in chunks)
        assert any("```bash" in c for c in chunks)