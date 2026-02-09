class LLMClient:
    def __init__(self, llama):
        self.llama = llama

    def generate(self, prompt: str) -> str:
        response = self.llama(
            prompt,
            max_tokens=512,
            stop=["</s>"]
        )
        return response["choices"][0]["text"].strip()
