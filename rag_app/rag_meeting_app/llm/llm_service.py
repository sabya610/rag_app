class LLMService:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.prompt_path = Path(__file__).parent / "prompts" / "issues_extraction.md"

    def extract_meeting_issue(self, chunk: str) -> dict | None:
        prompt_template = self.prompt_path.read_text(encoding="utf-8")

        prompt = prompt_template.replace("{{chunk}}", chunk)

        response = self.llm.generate(prompt).strip()

        if response == "NO_SIGNAL":
            return None

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "discussion_type": "unknown",
                "problem_statement": response,
                "blockers_or_risks": "",
                "decisions_made": "",
                "action_items": [],
                "urgency_level": "medium",
            }
