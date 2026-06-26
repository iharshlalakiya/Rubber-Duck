"""
LLM interaction layer — all calls to the Hugging Face Inference API live here.

Three responsibilities:
1. review_with_llm()          — the main PR review call with the structured JSON prompt.
2. llm_classify_dismissal()   — intent classification for ambiguous dismiss replies.
3. answer_mention_with_llm()  — conversational answers when @RubberDuck is mentioned.
"""

import json
import requests

from rubber_duck.config import HF_API_TOKEN, HF_API_URL, HF_MODEL
from rubber_duck.stack_detection import build_checklist

# ---------- Review prompt ----------

CHECKLIST_PROMPT_TEMPLATE = """You are reviewing a pull request for a small startup's codebase. You are NOT a generic linter — you only flag things a linter would miss.

This may be a monorepo with multiple sub-projects. Detected stack per changed area:
{detected_stack_label}

Review the DIFF below using this checklist:
{checklist}

RULES:
- Only flag things you are genuinely confident about. Skip anything borderline or stylistic.
- Severity: "high" (security/correctness risk) or "medium" (clarity/maintainability risk).
- confidence: a number 0.0-1.0 for how sure you are this is a real, valid issue (not a guess).
- Max 5 flags. If there's nothing worth flagging, return an empty list — do not invent filler comments.
- Respond ONLY with valid JSON, no markdown fences, no preamble. Format:

{{
  "flags": [
    {{
      "severity": "high",
      "title": "short title",
      "detail": "1-2 sentence explanation, reference the specific file/line if possible",
      "file": "filename",
      "confidence": 0.85
    }}
  ],
  "future_hire_score": {{
    "score": 7,
    "reason": "1 sentence on why this diff would or wouldn't confuse a new hire in 6 months"
  }}
}}

REPO CONTEXT (existing related files, for cross-file comparison):
{context}

PR DIFF:
{diff}
"""

# Minimum confidence (0.0-1.0) a flag needs to actually get posted — cuts noise from guesses.
MIN_CONFIDENCE_THRESHOLD = 0.6


def review_with_llm(diff: str, context: str, detected_stacks: set, stack_label: str) -> dict:
    checklist = build_checklist(detected_stacks)

    prompt = CHECKLIST_PROMPT_TEMPLATE.format(
        detected_stack_label=stack_label,
        checklist=checklist,
        context=context or "(no related context files found)",
        diff=diff[:20000],
    )

    resp = requests.post(
        HF_API_URL,
        headers={
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": HF_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0.2,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    text = data["choices"][0]["message"]["content"]
    text = text.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"flags": [], "_raw_error": text[:500]}


def filter_low_confidence_flags(flags: list, threshold: float = MIN_CONFIDENCE_THRESHOLD) -> list:
    """
    Drops flags below the confidence threshold. Defaults missing confidence to 1.0
    (assume confident) so this stays backward-compatible if a model ever omits the field.
    """
    return [f for f in flags if f.get("confidence", 1.0) >= threshold]


# ---------- Dismiss intent classification ----------

DISMISS_KEYWORDS = ("not relevant", "ignore", "dismiss", "wontfix", "won't fix", "false positive", "n/a")


def reply_is_dismissal(reply_text: str) -> bool:
    """
    Fast-path: obvious keyword matches skip the LLM call entirely (cheaper, instant).
    Falls back to an LLM intent check for everything else, since real replies are
    often phrased creatively ("nah skip that", "this is fine as-is", "not a real issue").
    """
    lowered = reply_text.lower()
    if any(k in lowered for k in DISMISS_KEYWORDS):
        return True
    return llm_classify_dismissal(reply_text)


def llm_classify_dismissal(reply_text: str) -> bool:
    prompt = f"""A developer replied to a code review flag with this message:
"{reply_text}"

Is this message DISMISSING the flag (saying it's not relevant, intentional, a false positive,
or otherwise telling the reviewer to drop it) — as opposed to asking a question, agreeing to fix it,
or saying something unrelated?

Respond with exactly one word: YES or NO."""

    try:
        resp = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"},
            json={
                "model": HF_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 5,
                "temperature": 0.0,
            },
            timeout=20,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        print(f"[llm_classify_dismissal] failed, defaulting to NO: {e}")
        return False


# ---------- Conversational @mention answering ----------

def answer_mention_with_llm(question: str, open_flags: list) -> str:
    flags_context = "\n".join(
        f"- [{f.get('severity')}] {f.get('title')} ({f.get('file')}): {f.get('detail')}"
        for f in open_flags
    ) or "(no open flags currently on this PR)"

    prompt = f"""You are Rubber Duck, a code review bot. A developer asked you a question in a PR comment.
Answer concisely and conversationally (2-4 sentences max), like a helpful teammate, not a formal report.

Open flags currently on this PR:
{flags_context}

Developer's question:
{question}

Respond with plain text only, no JSON, no markdown headers."""

    resp = requests.post(
        HF_API_URL,
        headers={"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"},
        json={
            "model": HF_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.4,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
