def build_embedding_text(issue: dict) -> str:
    return f"""
    Problem: {issue['problem_statement']}
    Blockers: {issue['blockers_or_risks']}
    Decisions: {issue['decisions_made']}
    Actions: {', '.join(issue['action_items'])}
    Urgency: {issue['urgency_level']}
    """.strip()
