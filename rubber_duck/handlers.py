"""
Webhook event handlers — one function per GitHub event type.
Each handler is a pure function that accepts the raw webhook payload dict
and returns a status dict. All side-effects (GitHub API calls, Supabase writes)
are delegated to the appropriate module.
"""

import supabase_store
from rubber_duck.context import gather_repo_context
from rubber_duck.flags import (
    filter_dismissed_flags,
    post_flags_to_pr,
    post_old_ledger_reminders,
    update_tech_debt_file,
)
from rubber_duck.github_client import (
    get_changed_files,
    get_default_branch,
    get_pr_diff,
    post_pr_comment,
    reply_to_review_comment,
)
from rubber_duck.llm import (
    answer_mention_with_llm,
    filter_low_confidence_flags,
    reply_is_dismissal,
    review_with_llm,
)
from rubber_duck.stack_detection import (
    all_detected_tags,
    build_stack_label,
    detect_stacks_per_package,
    is_monorepo,
)


def handle_pull_request_event(payload: dict) -> dict:
    action = payload.get("action")

    repo_full_name = payload["repository"]["full_name"]
    pr_number = payload["pull_request"]["number"]

    if action == "closed":
        # If merged, assume flags were addressed via review; mark resolved so the
        # ledger doesn't resurface stale items forever. (Doesn't verify the fix —
        # just stops nagging once the PR is gone.)
        if payload["pull_request"].get("merged") and supabase_store.is_configured():
            supabase_store.mark_flags_resolved_for_pr(repo_full_name, pr_number)
        return {"status": "ok", "action": "closed_handled"}

    if action not in ("opened", "synchronize", "reopened"):
        return {"status": "ignored", "reason": f"action {action} not handled"}

    diff = get_pr_diff(repo_full_name, pr_number)
    changed_files = get_changed_files(repo_full_name, pr_number)
    context, all_paths = gather_repo_context(repo_full_name, changed_files)

    ref = get_default_branch(repo_full_name)
    groups = detect_stacks_per_package(repo_full_name, changed_files, all_paths, ref)
    detected_stacks = all_detected_tags(groups)
    stack_label = build_stack_label(groups)

    review = review_with_llm(diff, context, detected_stacks, stack_label)
    flags = review.get("flags", [])

    flags_before_confidence = len(flags)
    flags = filter_low_confidence_flags(flags)
    flags = filter_dismissed_flags(repo_full_name, flags)

    # Tech debt ledger: resurface old unresolved flags on files touched by this PR.
    post_old_ledger_reminders(repo_full_name, pr_number, changed_files)

    post_flags_to_pr(repo_full_name, pr_number, diff, flags)

    future_hire_score = review.get("future_hire_score")
    if future_hire_score:
        post_pr_comment(
            repo_full_name, pr_number,
            f"📋 **Future-hire readiness: {future_hire_score.get('score', '?')}/10** — {future_hire_score.get('reason', '')}",
        )

    # Keep TECH_DEBT.md in sync with the latest ledger state, visible directly in the repo.
    update_tech_debt_file(repo_full_name, ref)

    return {
        "status": "ok",
        "flags_found": len(flags),
        "flags_dropped_low_confidence": flags_before_confidence - len(flags),
        "detected_stacks": sorted(detected_stacks),
        "is_monorepo": is_monorepo(all_paths),
        "groups": {k: {"stacks": sorted(v["stacks"]), "files": v["changed_files"]} for k, v in groups.items()},
    }


def handle_review_comment_reply(payload: dict) -> dict:
    """
    Fires on `pull_request_review_comment` (action=created). If the new comment is a
    reply to one of OUR inline flag comments, we know exactly which flag it's about
    via in_reply_to_id — no guessing needed.
    """
    comment = payload.get("comment", {})
    in_reply_to_id = comment.get("in_reply_to_id")
    if not in_reply_to_id:
        return {"status": "ignored", "reason": "not a reply"}

    if not supabase_store.is_configured():
        return {"status": "ignored", "reason": "supabase not configured"}

    matching_flags = supabase_store.get_flags_by_comment_id(in_reply_to_id)
    if not matching_flags:
        return {"status": "ignored", "reason": "reply not tied to a known flag"}

    repo_full_name = payload["repository"]["full_name"]
    pr_number = payload["pull_request"]["number"]
    reply_text = comment.get("body", "")

    if len(matching_flags) > 1:
        # This is a reply on a BATCHED comment (multiple flags share this comment_id) —
        # we can't know which specific flag they mean without asking, so answer
        # conversationally using all of them as context instead of guessing one to dismiss.
        answer = answer_mention_with_llm(reply_text, matching_flags)
        reply_to_review_comment(repo_full_name, pr_number, in_reply_to_id, f"🦆 {answer}")
        return {"status": "ok", "action": "answered_batched", "flag_count": len(matching_flags)}

    flag = matching_flags[0]

    if reply_is_dismissal(reply_text):
        supabase_store.mark_flag_status(flag["id"], "dismissed")
        supabase_store.add_dismissed_pattern(repo_full_name, flag["title"])
        reply_to_review_comment(
            repo_full_name, pr_number, in_reply_to_id,
            "🦆 Got it — dismissed. I'll stop flagging this pattern in this repo.",
        )
        return {"status": "ok", "action": "dismissed", "flag_id": flag["id"]}

    # Not a clear dismissal — treat it as a question/discussion and answer conversationally.
    answer = answer_mention_with_llm(reply_text, [flag])
    reply_to_review_comment(repo_full_name, pr_number, in_reply_to_id, f"🦆 {answer}")
    return {"status": "ok", "action": "answered", "flag_id": flag["id"]}


def handle_mention_comment(payload: dict) -> dict:
    """
    Fires on `issue_comment` (action=created) on a PR. If the comment mentions
    @RubberDuck (case-insensitive), answer conversationally using that PR's open flags.
    """
    comment_body = payload.get("comment", {}).get("body", "")
    if "@rubberduck" not in comment_body.lower():
        return {"status": "ignored", "reason": "no mention found"}

    # issue_comment fires for both issues and PRs — only handle PRs (they have a pull_request key).
    if "pull_request" not in payload.get("issue", {}):
        return {"status": "ignored", "reason": "comment is on an issue, not a PR"}

    repo_full_name = payload["repository"]["full_name"]
    pr_number = payload["issue"]["number"]

    open_flags = supabase_store.get_open_flags_for_pr(repo_full_name, pr_number) if supabase_store.is_configured() else []
    question = comment_body.lower().replace("@rubberduck", "").strip() or comment_body

    answer = answer_mention_with_llm(question, open_flags)
    post_pr_comment(repo_full_name, pr_number, f"🦆 {answer}")

    return {"status": "ok", "action": "mention_answered"}
