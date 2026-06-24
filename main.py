"""
Rubber Duck — MVP
Stack focus: FastAPI + Supabase repos being reviewed.
This service itself is FastAPI (the bot), but it reviews ANY repo you point it at.

Flow:
1. GitHub sends a webhook when a PR is opened/updated.
2. We fetch the diff + relevant repo context.
3. We send it to Claude with a stack-specific checklist prompt.
4. We parse Claude's structured response and post ONE digest comment back to the PR.
"""

import os
import hmac
import hashlib
import json
import requests
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # personal access token or GitHub App token

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# Pick any instruction-tuned model available via HF Inference Providers.
# Good options for code review:
#   "Qwen/Qwen2.5-Coder-32B-Instruct"  (strong at code, recommended)
#   "meta-llama/Llama-3.1-8B-Instruct" (faster, weaker reasoning)
#   "mistralai/Mistral-7B-Instruct-v0.3"
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"


# ---------- Webhook signature verification ----------
def verify_signature(payload_body: bytes, signature_header: str) -> bool:
    if not GITHUB_WEBHOOK_SECRET or not signature_header:
        return False
    hash_object = hmac.new(
        GITHUB_WEBHOOK_SECRET.encode("utf-8"), msg=payload_body, digestmod=hashlib.sha256
    )
    expected_signature = "sha256=" + hash_object.hexdigest()
    return hmac.compare_digest(expected_signature, signature_header)


# ---------- GitHub API helpers ----------
def gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def get_pr_diff(repo_full_name: str, pr_number: int) -> str:
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"
    headers = gh_headers()
    headers["Accept"] = "application/vnd.github.v3.diff"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.text


def get_changed_files(repo_full_name: str, pr_number: int) -> list:
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files"
    resp = requests.get(url, headers=gh_headers())
    resp.raise_for_status()
    return [f["filename"] for f in resp.json()]


def fetch_file_content(repo_full_name: str, path: str, ref: str = "main") -> str:
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
    resp = requests.get(url, headers=gh_headers(), params={"ref": ref})
    if resp.status_code != 200:
        return ""
    import base64
    content = resp.json().get("content", "")
    try:
        return base64.b64decode(content).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def post_pr_comment(repo_full_name: str, pr_number: int, body: str):
    url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"
    resp = requests.post(url, headers=gh_headers(), json={"body": body})
    resp.raise_for_status()
    return resp.json()


# ---------- Context gathering: FULL REPO (Tier 1) ----------

# File extensions we actually want to read as code/text context.
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".sql", ".json", ".yaml", ".yml",
    ".md", ".env.example", ".toml", ".cfg", ".ini", ".html",
}
# Paths to skip entirely — noise, generated, or irrelevant to review.
SKIP_PATH_PARTS = {
    "node_modules", ".git", "dist", "build", "venv", ".venv", "__pycache__",
    "migrations", "lock", ".lock", "coverage", ".next", "vendor",
}

MAX_TOTAL_CONTEXT_CHARS = 60000  # keep prompt size sane; ~15-20k tokens
MAX_FILE_CHARS = 4000           # cap any single file so one huge file can't eat the budget


def get_default_branch(repo_full_name: str) -> str:
    url = f"https://api.github.com/repos/{repo_full_name}"
    resp = requests.get(url, headers=gh_headers())
    resp.raise_for_status()
    return resp.json().get("default_branch", "main")


def list_all_repo_files(repo_full_name: str, ref: str) -> list:
    """Uses the Git Trees API (recursive) to list every file path in one call."""
    url = f"https://api.github.com/repos/{repo_full_name}/git/trees/{ref}"
    resp = requests.get(url, headers=gh_headers(), params={"recursive": "1"})
    resp.raise_for_status()
    tree = resp.json().get("tree", [])

    paths = []
    for item in tree:
        if item.get("type") != "blob":
            continue
        path = item["path"]
        if any(skip in path for skip in SKIP_PATH_PARTS):
            continue
        if not any(path.endswith(ext) for ext in CODE_EXTENSIONS):
            continue
        paths.append(path)
    return paths


def gather_repo_context(repo_full_name: str, changed_files: list):
    """
    Tier 1 context: pull the FULL repo (every relevant code/text file),
    not just files matching keywords. This lets the model catch real
    cross-file duplication and inconsistency, not just guess from filenames.

    Files already in the diff are skipped here (the diff itself covers them) —
    we only need the *surrounding* codebase for comparison.

    Returns (context_text, all_paths) — all_paths is reused for stack detection.
    """
    ref = get_default_branch(repo_full_name)
    all_paths = list_all_repo_files(repo_full_name, ref)

    context_chunks = []
    total_chars = 0

    for path in all_paths:
        if path in changed_files:
            continue  # already covered by the diff itself

        content = fetch_file_content(repo_full_name, path, ref)
        if not content:
            continue

        snippet = content[:MAX_FILE_CHARS]
        chunk = f"\n--- {path} ---\n{snippet}"

        if total_chars + len(chunk) > MAX_TOTAL_CONTEXT_CHARS:
            break  # budget hit, stop adding more files

        context_chunks.append(chunk)
        total_chars += len(chunk)

    return "\n".join(context_chunks), all_paths


# ---------- Stack detection (monorepo-aware, per-package) ----------

# Manifest files that mark the "root" of a package/app and declare its dependencies.
MANIFEST_FILENAMES = {
    "package.json": "node",
    "requirements.txt": "python",
    "pyproject.toml": "python",
    "go.mod": "go",
    "Gemfile": "ruby",
    "composer.json": "php",
}

# Dependency name -> stack tag. Checked against parsed manifest dependency lists,
# NOT raw text grep — far fewer false positives than scanning whole-file text.
DEPENDENCY_SIGNALS = {
    "fastapi": "fastapi",
    "django": "django",
    "flask": "flask",
    "express": "node-express",
    "next": "nextjs",
    "@supabase/supabase-js": "supabase",
    "supabase": "supabase",
    "firebase": "firebase",
    "firebase-admin": "firebase",
}

MONOREPO_MARKERS = {"turbo.json", "nx.json", "pnpm-workspace.yaml", "lerna.json"}


def is_monorepo(all_paths: list) -> bool:
    filenames = {p.split("/")[-1] for p in all_paths}
    if filenames & MONOREPO_MARKERS:
        return True
    # crude fallback: more than one package.json/requirements.txt outside root = monorepo-ish
    manifest_dirs = {
        "/".join(p.split("/")[:-1]) for p in all_paths if p.split("/")[-1] in MANIFEST_FILENAMES
    }
    return len(manifest_dirs) > 1


def find_nearest_manifest(file_path: str, all_paths_set: set) -> str | None:
    """
    Walks up the directory tree from file_path looking for the nearest manifest file.
    Returns the manifest's path, or None if none found (e.g. file is at repo root
    with no manifest, or repo has no manifests at all).
    """
    parts = file_path.split("/")[:-1]  # drop the filename itself
    while True:
        prefix = "/".join(parts)
        for manifest_name in MANIFEST_FILENAMES:
            candidate = f"{prefix}/{manifest_name}" if prefix else manifest_name
            if candidate in all_paths_set:
                return candidate
        if not parts:
            break
        parts = parts[:-1]
    return None


def parse_manifest_dependencies(repo_full_name: str, manifest_path: str, ref: str) -> set:
    """
    Parses actual dependency fields instead of grepping raw text.
    Returns a set of detected stack tags for this one manifest.
    """
    content = fetch_file_content(repo_full_name, manifest_path, ref)
    if not content:
        return set()

    tags = set()
    filename = manifest_path.split("/")[-1]

    if filename == "package.json":
        try:
            data = json.loads(content)
            deps = {}
            deps.update(data.get("dependencies", {}))
            deps.update(data.get("devDependencies", {}))
            for dep_name in deps:
                for signal, tag in DEPENDENCY_SIGNALS.items():
                    if signal in dep_name.lower():
                        tags.add(tag)
        except json.JSONDecodeError:
            pass

    elif filename in ("requirements.txt", "pyproject.toml"):
        lower = content.lower()
        for signal, tag in DEPENDENCY_SIGNALS.items():
            # word-boundary-ish check: signal followed by version pin char, newline, or end
            if signal in lower:
                tags.add(tag)

    return tags


def detect_stacks_per_package(repo_full_name: str, changed_files: list, all_paths: list, ref: str) -> dict:
    """
    Core fix for monorepos: instead of one global stack label for the whole repo,
    detect the stack PER CHANGED FILE by walking up to its nearest manifest and
    parsing that manifest's actual dependencies.

    Returns: { manifest_path_or_"unscoped": {changed_files: [...], stacks: {...}} }
    """
    all_paths_set = set(all_paths)
    manifest_cache: dict = {}  # manifest_path -> set of tags, to avoid re-fetching/parsing
    groups: dict = {}

    for file_path in changed_files:
        manifest_path = find_nearest_manifest(file_path, all_paths_set)
        group_key = manifest_path or "unscoped"

        if group_key not in groups:
            groups[group_key] = {"changed_files": [], "stacks": set()}
        groups[group_key]["changed_files"].append(file_path)

        if manifest_path:
            if manifest_path not in manifest_cache:
                manifest_cache[manifest_path] = parse_manifest_dependencies(repo_full_name, manifest_path, ref)
            groups[group_key]["stacks"] |= manifest_cache[manifest_path]

    return groups


STACK_CHECKLISTS = {
    "fastapi": """
FASTAPI-SPECIFIC:
- New public endpoint with no rate limiting, when sibling endpoints have one
- Raw SQL built from user input (injection risk)
- New endpoint missing Pydantic input validation when others have it
""",
    "django": """
DJANGO-SPECIFIC:
- View missing `@login_required` / permission check where similar views have one
- Raw SQL or `.extra()` usage with unsanitized user input
- `DEBUG = True` or secrets present in settings.py
""",
    "flask": """
FLASK-SPECIFIC:
- New route with no auth decorator when sibling routes have one
- User input passed directly into a database query without sanitization
- Secrets hardcoded instead of read from environment/config
""",
    "node-express": """
NODE/EXPRESS-SPECIFIC:
- New route with no auth middleware when sibling routes have one
- User input passed directly into a database query without sanitization
- Secrets/API keys hardcoded instead of read from environment variables
""",
    "nextjs": """
NEXT.JS-SPECIFIC:
- API route (app/api or pages/api) with no auth check when siblings have one
- Server-only secrets (e.g. service role keys) referenced in a client component
- Missing input validation on a new API route handler
""",
    "supabase": """
SUPABASE-SPECIFIC:
- New table created without RLS (Row Level Security) policy enabled
- Service role key used outside trusted server context (e.g. found in frontend/client code)
- RLS policy that is effectively unrestricted (e.g. `USING (true)`)
- Auth-protected route that doesn't verify the requesting user owns the row being accessed
""",
    "firebase": """
FIREBASE-SPECIFIC:
- Firestore security rules missing or overly permissive (e.g. `allow read, write: if true`)
- Firebase config/API keys committed with elevated (admin SDK) credentials in client code
- Cloud Function with no auth check on a callable/HTTP trigger
""",
}

GENERAL_CHECKLIST = """
CROSS-FILE / GENERAL (always check, regardless of stack):
- Logic that duplicates a function already in the provided repo context
- Magic numbers or unclear naming with no explanatory comment ("future hire confusion")
- New environment variable introduced but not documented
- Obvious security smells: hardcoded secrets/credentials, missing auth on what looks like a protected route, unsanitized user input reaching a query
"""


def build_stack_label(groups: dict) -> str:
    """
    Produces a human-readable breakdown like:
      apps/api/requirements.txt -> fastapi, supabase (files: apps/api/main.py, apps/api/db.py)
      apps/web/package.json -> nextjs (files: apps/web/pages/index.tsx)
    instead of one flat label for the whole repo — this is the actual monorepo fix.
    """
    if not groups:
        return "general / unrecognized"

    lines = []
    for manifest_path, info in groups.items():
        stacks = ", ".join(sorted(info["stacks"])) if info["stacks"] else "unrecognized"
        files_preview = ", ".join(info["changed_files"][:5])
        lines.append(f"- [{manifest_path}] stacks: {stacks} (files: {files_preview})")
    return "\n".join(lines)


def all_detected_tags(groups: dict) -> set:
    tags = set()
    for info in groups.values():
        tags |= info["stacks"]
    return tags


def build_checklist(detected_stacks: set) -> str:
    sections = [GENERAL_CHECKLIST]
    for tag in detected_stacks:
        if tag in STACK_CHECKLISTS:
            sections.append(STACK_CHECKLISTS[tag])
    return "\n".join(sections)


# ---------- LLM review prompt ----------
CHECKLIST_PROMPT_TEMPLATE = """You are reviewing a pull request for a small startup's codebase. You are NOT a generic linter — you only flag things a linter would miss.

This may be a monorepo with multiple sub-projects. Detected stack per changed area:
{detected_stack_label}

Review the DIFF below using this checklist:
{checklist}

RULES:
- Only flag things you are genuinely confident about. Skip anything borderline or stylistic.
- Severity: "high" (security/correctness risk) or "medium" (clarity/maintainability risk).
- Max 5 flags. If there's nothing worth flagging, return an empty list — do not invent filler comments.
- Respond ONLY with valid JSON, no markdown fences, no preamble. Format:

{{
  "flags": [
    {{
      "severity": "high",
      "title": "short title",
      "detail": "1-2 sentence explanation, reference the specific file/line if possible",
      "file": "filename"
    }}
  ]
}}

REPO CONTEXT (existing related files, for cross-file comparison):
{context}

PR DIFF:
{diff}
"""


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


# ---------- Digest formatting ----------
def format_digest(flags: list) -> str:
    if not flags:
        return "**🦆 Rubber Duck** — No flags this time. Looks good."

    high = [f for f in flags if f.get("severity") == "high"]
    medium = [f for f in flags if f.get("severity") == "medium"]

    lines = [f"**🦆 Rubber Duck — {len(flags)} flag(s) ({len(high)} high, {len(medium)} medium)**", ""]
    icon = {"high": "🔴", "medium": "🟡"}
    for f in flags:
        lines.append(f"{icon.get(f.get('severity'), '⚪')} **{f.get('title')}** — {f.get('detail')} (`{f.get('file', '')}`)")

    lines.append("")
    lines.append("_Reply `not relevant` under any flag to stop seeing that pattern._")
    return "\n".join(lines)


# ---------- Webhook endpoint ----------
@app.post("/webhook/github")
async def github_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")

    if GITHUB_WEBHOOK_SECRET and not verify_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    if not body:
        return {"status": "ignored", "reason": "empty body received"}

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return {"status": "ignored", "reason": "could not parse JSON body"}
    event = request.headers.get("X-GitHub-Event")

    if event != "pull_request":
        return {"status": "ignored", "reason": f"event {event} not handled"}

    action = payload.get("action")
    if action not in ("opened", "synchronize", "reopened"):
        return {"status": "ignored", "reason": f"action {action} not handled"}

    repo_full_name = payload["repository"]["full_name"]
    pr_number = payload["pull_request"]["number"]

    diff = get_pr_diff(repo_full_name, pr_number)
    changed_files = get_changed_files(repo_full_name, pr_number)
    context, all_paths = gather_repo_context(repo_full_name, changed_files)

    ref = get_default_branch(repo_full_name)
    groups = detect_stacks_per_package(repo_full_name, changed_files, all_paths, ref)
    detected_stacks = all_detected_tags(groups)
    stack_label = build_stack_label(groups)

    review = review_with_llm(diff, context, detected_stacks, stack_label)
    digest = format_digest(review.get("flags", []))

    post_pr_comment(repo_full_name, pr_number, digest)

    return {
        "status": "ok",
        "flags_found": len(review.get("flags", [])),
        "detected_stacks": sorted(detected_stacks),
        "is_monorepo": is_monorepo(all_paths),
        "groups": {k: {"stacks": sorted(v["stacks"]), "files": v["changed_files"]} for k, v in groups.items()},
    }


@app.get("/health")
def health():
    return {"status": "ok"}
