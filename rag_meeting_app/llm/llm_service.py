import json
import re
from pathlib import Path


class LLMService:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.prompt_path = Path(__file__).parent / "prompts" / "issues_extraction.md"

    def _extract_json(self, text: str) -> dict | None:
        """Try to extract a JSON object from within the LLM response text."""
        # Find the first { ... } block in the response
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def extract_meeting_issue(self, chunk: str) -> dict | None:
        prompt_template = self.prompt_path.read_text(encoding="utf-8")

        raw_prompt = prompt_template.replace("{{chunk}}", chunk)
        # Wrap in Llama-2 chat format
        prompt = f"[INST] {raw_prompt} [/INST]"
        print("Prompt length chars:", len(prompt))
        response = self.llm.generate(prompt).strip()

        # Fuzzy NO_SIGNAL detection
        if "NO_SIGNAL" in response:
            return None

        # Try to parse clean JSON first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON embedded in conversational text
        extracted = self._extract_json(response)
        if extracted:
            return extracted

        # If we still can't parse, skip this chunk
        return None
