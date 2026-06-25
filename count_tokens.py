#!/usr/bin/env python3
"""Count tokens for the last query's prompt using the loaded LLaMA tokenizer."""
import sys, os
sys.stderr = open(os.devnull, "w")

from llama_cpp import Llama

llm = Llama(
    model_path="/app/rag_app/models/llm/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,
    verbose=False,
    use_mlock=True,
    use_mmap=True,
)

# Read last context from the log (or use a sample prompt)
# We'll tokenize just the prompt template to show structure overhead
sample_prompt = """You are a Kubernetes platform assistant for HPE Ezmeral.

CRITICAL RULES:
1. If the context below does NOT contain information directly relevant to the question, you MUST respond ONLY with:
   "I don't have enough information in the knowledge base to answer this question. Please refine your search or check the HPE Ezmeral documentation at https://support.hpe.com."
2. Do NOT make up, guess, or fabricate any steps, commands, or instructions.
3. Use only CLI commands and instructions found **exactly** in the context below.
4. Do not invent commands that are not in the context.
5. Ensure numbered steps are consistent and not appended multiple times.
6. Make sure each bash block starts and ends properly.
7. Ensure the output does not have duplicated instructions.
8. Provide a clear, **numbered** list of steps in order.
9. Each step must be on a new line and contain any commands inside fenced ```bash code blocks.

### Context:
{CONTEXT_PLACEHOLDER}

### Question:
{QUESTION_PLACEHOLDER}

---
If the context does not contain relevant information to answer the question, say so. Otherwise provide:
Numbered Steps:
"""

# Count template tokens (without context)
template_only = sample_prompt.replace("{CONTEXT_PLACEHOLDER}", "").replace("{QUESTION_PLACEHOLDER}", "")
template_tokens = llm.tokenize(template_only.encode("utf-8"))

print(f"=== TOKEN USAGE ANALYSIS ===")
print(f"Prompt template (without context): {len(template_tokens)} tokens")
print(f"Model context window (n_ctx):      4096 tokens")
print(f"Max completion tokens:             2048 tokens")
print(f"Available for context + question:  {4096 - len(template_tokens) - 2048} tokens")
print()

# Now try to read the actual log to find last context
import subprocess
result = subprocess.run(["grep", "-c", ".", "/proc/1/fd/1"], capture_output=True, text=True)

# Alternative: read from a known context size
# Estimate from the log output the user shared
sample_context = open("/tmp/last_context.txt", "r").read() if os.path.exists("/tmp/last_context.txt") else None

if sample_context:
    ctx_tokens = llm.tokenize(sample_context.encode("utf-8"))
    print(f"Last context tokens:               {len(ctx_tokens)} tokens")
    total = len(template_tokens) + len(ctx_tokens)
    print(f"Total prompt tokens:               {total} tokens")
    print(f"Remaining for completion:          {4096 - total} tokens")
    if total > 2048:
        print(f"WARNING: Prompt uses {total} tokens, only {4096 - total} left for completion!")
else:
    print("No cached context found. Showing estimates by context size:")
    for chars in [2000, 4000, 6000, 8000, 10000]:
        est_tokens = int(chars / 3.5)  # rough estimate for mixed text
        total = len(template_tokens) + est_tokens
        remaining = 4096 - total
        status = "OK" if remaining > 500 else "TIGHT" if remaining > 0 else "OVERFLOW"
        print(f"  ~{chars} chars context -> ~{est_tokens} prompt tokens, {remaining} left for completion [{status}]")
