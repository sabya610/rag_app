import tiktoken
import sys

enc = tiktoken.get_encoding("cl100k_base")

context = sys.stdin.read().strip()

prompt = f"""You are a Kubernetes platform assistant for HPE Ezmeral.

CRITICAL RULES:
1. If the context below does NOT contain information directly relevant to the question, you MUST respond ONLY with:
   I dont have enough information in the knowledge base to answer this question.
2. Do NOT make up, guess, or fabricate any steps, commands, or instructions.
3. Use only CLI commands and instructions found exactly in the context below.
4. Do not invent commands that are not in the context.
5. Ensure numbered steps are consistent and not appended multiple times.
6. Make sure each bash block starts and ends properly.
7. Ensure the output does not have duplicated instructions.
8. Provide a clear, numbered list of steps in order.
9. Each step must be on a new line and contain any commands inside fenced bash code blocks.

### Context:
{context}

### Question:
what are the steps for manual restart ERE?

---
If the context does not contain relevant information to answer the question, say so. Otherwise provide:
Numbered Steps:
"""

tokens = enc.encode(prompt)
template_no_ctx = enc.encode(prompt.replace(context, ""))

print("=== TOKEN USAGE ANALYSIS (tiktoken cl100k_base estimate) ===")
print(f"Prompt template (without context): {len(template_no_ctx)} tokens")
print(f"Context tokens:                    {len(tokens) - len(template_no_ctx)} tokens")
print(f"Total prompt tokens:               {len(tokens)} tokens")
print(f"Model context window (n_ctx):      4096 tokens")
print(f"Max completion tokens:             2048 tokens")
print(f"Tokens remaining for completion:   {4096 - len(tokens)} tokens")
print()
if len(tokens) > 4096:
    print(f"*** WARNING: Prompt EXCEEDS context window! Context will be truncated. ***")
elif len(tokens) > 2048:
    print(f"*** WARNING: Only {4096 - len(tokens)} tokens left for completion (max_tokens=2048 will be capped). ***")
else:
    print(f"OK: {4096 - len(tokens)} tokens available for completion.")
print(f"Context character count:           {len(context)} chars")
