"""
Flag posting, filtering, and the tech debt ledger.

Responsibilities:
- format_digest()            — plain-text summary for no-flag and fallback cases.
- filter_dismissed_flags()   — drops flags matching repo-level dismissal patterns.
- post_flags_to_pr()         — severity-aware posting (inline for high, batched for medium).
- post_old_ledger_reminders()— resurfaces unresolved flags when the same files change again.
- generate_tech_debt_markdown() / update_tech_debt_file() — TECH_DEBT.md generation.
"""

import supabase_store
from rubber_duck.github_client import (
    commit_file_to_repo,
    parse_diff_added_lines,
    post_inline_review_comment,
    post_pr_comment,
    get_pr_head_sha,
)


# ---------- Digest formatting ----------

def format_digest(flags: list, future_hire_score: dict | None = None) -> str:
    score_line = ""
    if future_hire_score and future_hire_score.get("score") is not None:
        score_line = f"\n📋 **Future-hire readiness: {future_hire_score['score']}/10** — {future_hire_score.get('reason', '')}\n"

    if not flags:
        return f"**🦆 Rubber Duck** — No flags this time. Looks good.{score_line}"

    high = [f for f in flags if f.get("severity") == "high"]
    medium = [f for f in flags if f.get("severity") == "medium"]

    lines = [f"**🦆 Rubber Duck — {len(flags)} flag(s) ({len(high)} high, {len(medium)} medium)**", ""]
    icon = {"high": "🔴", "medium": "🟡"}
    for f in flags:
        lines.append(f"{icon.get(f.get('severity'), '⚪')} **{f.get('title')}** — {f.get('detail')} (`{f.get('file', '')}`)")

    if score_line:
        lines.append(score_line)

    lines.append("")
    lines.append("_Reply directly under any flag's comment to dismiss it, or comment `@RubberDuck <question>` anywhere on this PR._")
    return "\n".join(lines)


# ---------- Dismiss filtering ----------

def filter_dismissed_flags(repo_full_name: str, flags: list) -> list:
    """Drops any flag whose title closely matches something the founder already dismissed for this repo."""
    if not supabase_store.is_configured():
        return flags
    dismissed = supabase_store.get_dismissed_patterns(repo_full_name)
    if not dismissed:
        return flags

    kept = []
    for f in flags:
        title_lower = f.get("title", "").lower()
        if any(pattern in title_lower or title_lower in pattern for pattern in dismissed):
            continue
        kept.append(f)
    return kept


# ---------- Severity-aware flag posting ----------

def post_flags_to_pr(repo_full_name: str, pr_number: int, diff: str, flags: list):
    """
    Severity-aware posting to cut noise:
    - HIGH severity flags each get their OWN inline comment (anchored to a diff line when
      possible) — these are important enough to be impossible to miss, and individual
      comments support reply-threading for precise dismiss/discuss.
    - MEDIUM/lower severity flags get batched into ONE combined comment instead of N separate
      ones — still useful, but doesn't spam the PR with a comment per minor nit.
    Each flag (whether posted individually or as part of a batch) is still saved to Supabase
    individually, so the ledger and dismiss-tracking work the same either way.
    """
    if not flags:
        post_pr_comment(repo_full_name, pr_number, format_digest([]))
        return

    added_lines_by_file = parse_diff_added_lines(diff)
    commit_sha = get_pr_head_sha(repo_full_name, pr_number)
    icon = {"high": "🔴", "medium": "🟡"}

    high_flags = [f for f in flags if f.get("severity") == "high"]
    other_flags = [f for f in flags if f.get("severity") != "high"]

    # --- HIGH severity: individual inline comments ---
    for f in high_flags:
        file = f.get("file", "")
        title = f.get("title", "Flag")
        detail = f.get("detail", "")
        severity = f.get("severity", "high")
        body = (
            f"{icon.get(severity, '⚪')} **🦆 Rubber Duck — {title}**\n\n{detail}\n\n"
            f"_Reply here to discuss or dismiss this flag._"
        )

        candidate_lines = added_lines_by_file.get(file, set())
        line = f.get("line")
        anchored_line = line if line in candidate_lines else (max(candidate_lines) if candidate_lines else None)

        comment = None
        comment_type = "issue"
        if anchored_line:
            comment = post_inline_review_comment(repo_full_name, pr_number, commit_sha, file, anchored_line, body)
            comment_type = "review"
        if not comment:
            comment = post_pr_comment(repo_full_name, pr_number, body)
            comment_type = "issue"

        if supabase_store.is_configured() and comment:
            supabase_store.save_flag(
                repo=repo_full_name, pr_number=pr_number, github_comment_id=comment.get("id"),
                comment_type=comment_type, file=file, line=anchored_line,
                severity=severity, title=title, detail=detail,
            )

    # --- MEDIUM/other severity: one batched comment ---
    if other_flags:
        lines = [f"**🦆 Rubber Duck — {len(other_flags)} additional flag(s) (lower severity, batched)**", ""]
        for f in other_flags:
            sev = f.get("severity", "medium")
            lines.append(f"{icon.get(sev, '⚪')} **{f.get('title')}** — {f.get('detail')} (`{f.get('file', '')}`)")
        lines.append("")
        lines.append("_Reply `@RubberDuck <which flag> not relevant` to dismiss one of these — batched flags don't support direct reply-threading._")
        batch_body = "\n".join(lines)

        batch_comment = post_pr_comment(repo_full_name, pr_number, batch_body)

        if supabase_store.is_configured() and batch_comment:
            for f in other_flags:
                # Batched flags share the same github_comment_id (the batch comment) —
                # dismiss-by-reply-threading won't resolve to one specific flag here,
                # but @RubberDuck mentions can still reference them by title.
                supabase_store.save_flag(
                    repo=repo_full_name, pr_number=pr_number, github_comment_id=batch_comment.get("id"),
                    comment_type="issue_batched", file=f.get("file", ""), line=None,
                    severity=f.get("severity", "medium"), title=f.get("title", "Flag"), detail=f.get("detail", ""),
                )


# ---------- Tech debt ledger ----------

def post_old_ledger_reminders(repo_full_name: str, pr_number: int, changed_files: list):
    """
    Tech debt ledger / post-merge drift: if any of the files in THIS PR have old
    unresolved flags from PREVIOUS PRs, resurface them as a reminder comment.
    This is the differentiator vs. stateless review bots that only look at one PR at a time.
    """
    if not supabase_store.is_configured():
        return
    old_flags = supabase_store.get_old_open_flags_for_files(repo_full_name, changed_files, exclude_pr_number=pr_number)
    if not old_flags:
        return

    lines = ["**🦆 Rubber Duck — ↩️ Resurfaced from earlier PRs**", ""]
    for f in old_flags[:5]:
        lines.append(f"- **{f.get('title')}** in `{f.get('file')}` (flagged in PR #{f.get('pr_number')}, still unresolved)")
    lines.append("")
    lines.append("_These were flagged before and the related code changed again. Worth a look, or reply `not relevant` to dismiss._")

    post_pr_comment(repo_full_name, pr_number, "\n".join(lines))


def generate_tech_debt_markdown(open_flags: list) -> str:
    """Builds the full TECH_DEBT.md content from current open flags, grouped by severity."""
    if not open_flags:
        return (
            "# Tech Debt Ledger\n\n"
            "_Auto-generated by Rubber Duck. No open flags right now — nice._\n"
        )

    high = [f for f in open_flags if f.get("severity") == "high"]
    medium = [f for f in open_flags if f.get("severity") != "high"]

    lines = [
        "# Tech Debt Ledger",
        "",
        "_Auto-generated by Rubber Duck after each PR review. Reflects currently unresolved flags._",
        "",
        f"**{len(open_flags)} open flag(s)** — {len(high)} high, {len(medium)} medium/other",
        "",
    ]

    if high:
        lines.append("## 🔴 High severity")
        lines.append("")
        for f in high:
            lines.append(f"- **{f.get('title')}** — `{f.get('file')}` (from PR #{f.get('pr_number')})")
            lines.append(f"  {f.get('detail')}")
        lines.append("")

    if medium:
        lines.append("## 🟡 Medium / other")
        lines.append("")
        for f in medium:
            lines.append(f"- **{f.get('title')}** — `{f.get('file')}` (from PR #{f.get('pr_number')})")
            lines.append(f"  {f.get('detail')}")
        lines.append("")

    return "\n".join(lines)


def update_tech_debt_file(repo_full_name: str, ref: str):
    """Regenerates and commits TECH_DEBT.md directly to the repo's default branch."""
    if not supabase_store.is_configured():
        return
    open_flags = supabase_store.get_all_open_flags_for_repo(repo_full_name)
    content = generate_tech_debt_markdown(open_flags)
    commit_file_to_repo(
        repo_full_name, "TECH_DEBT.md", content, ref,
        commit_message="🦆 Rubber Duck: update tech debt ledger",
    )
