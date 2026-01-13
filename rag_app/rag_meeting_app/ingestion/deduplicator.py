import hashlib


def issue_hash(issue: dict) -> str:
    raw = (
        issue.get("problem_statement", "") +
        issue.get("decisions_made", "")
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()