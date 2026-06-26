"""
GitHub API helpers — all direct HTTP calls to api.github.com live here.
Includes: diff fetching, file fetching (with cache), posting comments,
inline review comments, replies, and committing files back to a repo.
"""

import base64
import hmac
import hashlib
import requests

from rubber_duck.config import GITHUB_TOKEN, GITHUB_WEBHOOK_SECRET


# ---------- Webhook signature verification ----------

def verify_signature(payload_body: bytes, signature_header: str) -> bool:
    if not GITHUB_WEBHOOK_SECRET or not signature_header:
        return False
    hash_object = hmac.new(
        GITHUB_WEBHOOK_SECRET.encode("utf-8"), msg=payload_body, digestmod=hashlib.sha256
    )
    expected_signature = "sha256=" + hash_object.hexdigest()
    return hmac.compare_digest(expected_signature, signature_header)


# ---------- Auth headers ----------

def gh_headers() -> dict:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


# ---------- PR data ----------

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


def get_pr_head_sha(repo_full_name: str, pr_number: int) -> str:
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"
    resp = requests.get(url, headers=gh_headers())
    resp.raise_for_status()
    return resp.json()["head"]["sha"]


def get_default_branch(repo_full_name: str) -> str:
    url = f"https://api.github.com/repos/{repo_full_name}"
    resp = requests.get(url, headers=gh_headers())
    resp.raise_for_status()
    return resp.json().get("default_branch", "main")


# ---------- File fetching (with in-process cache) ----------

def fetch_file_content(repo_full_name: str, path: str, ref: str = "main", _cache: dict = {}) -> str:
    """
    Cache keyed by (repo, path, ref) — within a single webhook handling, the same file
    is sometimes requested more than once (e.g. context gathering + manifest parsing).
    Note: this dict persists across requests too (cheap win, low memory cost for small repos);
    fine for an MVP, but swap for a proper TTL cache if running long-lived against big repos.
    """
    cache_key = (repo_full_name, path, ref)
    if cache_key in _cache:
        return _cache[cache_key]

    url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
    resp = requests.get(url, headers=gh_headers(), params={"ref": ref})
    if resp.status_code != 200:
        _cache[cache_key] = ""
        return ""

    content = resp.json().get("content", "")
    try:
        decoded = base64.b64decode(content).decode("utf-8", errors="ignore")
    except Exception:
        decoded = ""
    _cache[cache_key] = decoded
    return decoded


def list_all_repo_files(repo_full_name: str, ref: str, skip_path_parts: set, code_extensions: set) -> list:
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
        if any(skip in path for skip in skip_path_parts):
            continue
        if not any(path.endswith(ext) for ext in code_extensions):
            continue
        paths.append(path)
    return paths


# ---------- Posting comments ----------

def post_pr_comment(repo_full_name: str, pr_number: int, body: str) -> dict:
    url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"
    resp = requests.post(url, headers=gh_headers(), json={"body": body})
    resp.raise_for_status()
    return resp.json()


def post_inline_review_comment(
    repo_full_name: str, pr_number: int, commit_sha: str,
    file: str, line: int, body: str
) -> dict | None:
    """Posts a comment anchored to a specific line in the diff (supports reply-threading)."""
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/comments"
    payload = {"body": body, "commit_id": commit_sha, "path": file, "line": line, "side": "RIGHT"}
    resp = requests.post(url, headers=gh_headers(), json=payload)
    if resp.status_code not in (200, 201):
        print(f"[github] inline comment failed for {file}:{line} -> {resp.status_code} {resp.text}")
        return None
    return resp.json()


def reply_to_review_comment(
    repo_full_name: str, pr_number: int, comment_id, body: str
) -> dict | None:
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/comments/{comment_id}/replies"
    resp = requests.post(url, headers=gh_headers(), json={"body": body})
    if resp.status_code not in (200, 201):
        print(f"[github] reply failed -> {resp.status_code} {resp.text}")
        return None
    return resp.json()


# ---------- Committing files back to repo ----------

def get_file_sha_if_exists(repo_full_name: str, path: str, ref: str) -> str | None:
    """Returns the blob SHA of a file if it exists (required by GitHub's update API), else None."""
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
    resp = requests.get(url, headers=gh_headers(), params={"ref": ref})
    if resp.status_code == 200:
        return resp.json().get("sha")
    return None


def commit_file_to_repo(
    repo_full_name: str, path: str, content: str, ref: str, commit_message: str
) -> dict | None:
    """Creates or updates a file directly on the given branch via GitHub's Contents API."""
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
    existing_sha = get_file_sha_if_exists(repo_full_name, path, ref)

    payload = {
        "message": commit_message,
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
        "branch": ref,
    }
    if existing_sha:
        payload["sha"] = existing_sha

    resp = requests.put(url, headers=gh_headers(), json=payload)
    if resp.status_code not in (200, 201):
        print(f"[github] commit_file_to_repo failed for {path} -> {resp.status_code} {resp.text}")
        return None
    return resp.json()


# ---------- Diff parsing ----------

def parse_diff_added_lines(diff_text: str) -> dict:
    """
    Parses a unified diff into { filename: set(new-file line numbers that were added/changed) }.
    GitHub's inline review comment API requires the line to actually be part of the diff,
    so we only anchor a flag to a line if it shows up here — otherwise we fall back
    to an unanchored issue comment.
    """
    files = {}
    current_file = None
    new_line_num = None

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ b/"):
            current_file = raw_line[6:]
            files.setdefault(current_file, set())
        elif raw_line.startswith("@@"):
            # Example: @@ -10,6 +12,8 @@
            try:
                plus_part = raw_line.split("+")[1].split(" ")[0]
                new_line_num = int(plus_part.split(",")[0])
            except (IndexError, ValueError):
                new_line_num = None
        elif current_file and new_line_num is not None:
            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                files[current_file].add(new_line_num)
                new_line_num += 1
            elif raw_line.startswith("-") and not raw_line.startswith("---"):
                pass  # removed line, doesn't consume a new-file line number
            else:
                new_line_num += 1

    return files
