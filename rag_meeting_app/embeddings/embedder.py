def build_embedding_text(issue: dict) -> str:
    actions = issue.get("action_items") or []
    if isinstance(actions, str):
        actions = [actions]
    return f"""
    Problem: {issue['problem_statement']}
    Blockers: {issue['blockers_or_risks']}
    Decisions: {issue['decisions_made']}
    Actions: {', '.join(actions)}
    Urgency: {issue['urgency_level']}
    """.strip()
