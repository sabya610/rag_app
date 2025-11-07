def query_model(prompt: str) -> str:
    """
    Stub for LLM call — in production, this could call llama.cpp or OpenAI API.
    During tests, this method is patched (mocked).
    """
    # Real call example:
    # return llama_cpp_infer(prompt)
    return f"Mocked LLM response for: {prompt[:50]}"