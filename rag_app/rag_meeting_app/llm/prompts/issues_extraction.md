You are analyzing a meeting transcript excerpt.

Your task:
1. Decide whether the text contains a meaningful discussion.
2. If it does, extract structured information in JSON using this schema:

{
  "discussion_type": "",
  "problem_statement": "",
  "blockers_or_risks": "",
  "decisions_made": "",
  "action_items": [],
  "urgency_level": "low | medium | high"
}

Guidelines:
- discussion_type should be a short category like:
  coordination_and_planning, program_explanation, access_issue,
  decision_making, technical_blocker, follow_up, escalation
- action_items must be short, clear bullet-style strings
- urgency is:
  - high → deadlines, escalation, blockers
  - medium → planning, clarifications
  - low → informational only

If the text contains only greetings, confirmations, acknowledgements,
or filler (e.g. "ok", "yeah", "thanks"), respond with exactly:

NO_SIGNAL

Text:
{{chunk}}