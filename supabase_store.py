"""
Minimal Supabase persistence layer using PostgREST directly via `requests`.
No supabase-py dependency needed — keeps this consistent with the rest of the bot.

Requires in .env:
  SUPABASE_URL=https://xxxx.supabase.co
  SUPABASE_SERVICE_ROLE_KEY=eyJ...   (service role key, NOT the anon key — needed for server-side writes)
"""

import os
import requests

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


def _headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _table_url(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"


def insert_row(table: str, row: dict) -> dict | None:
    resp = requests.post(_table_url(table), headers=_headers(), json=row, timeout=15)
    if resp.status_code not in (200, 201):
        print(f"[supabase] insert into {table} failed: {resp.status_code} {resp.text}")
        return None
    data = resp.json()
    return data[0] if isinstance(data, list) and data else None


def select_rows(table: str, filters: dict, limit: int = 100) -> list:
    """
    filters: dict of column -> value, applied as equality filters (PostgREST eq.).
    """
    params = {f"{col}": f"eq.{val}" for col, val in filters.items()}
    params["limit"] = str(limit)
    resp = requests.get(_table_url(table), headers=_headers(), params=params, timeout=15)
    if resp.status_code != 200:
        print(f"[supabase] select from {table} failed: {resp.status_code} {resp.text}")
        return []
    return resp.json()


def update_row(table: str, filters: dict, updates: dict) -> bool:
    params = {f"{col}": f"eq.{val}" for col, val in filters.items()}
    resp = requests.patch(_table_url(table), headers=_headers(), params=params, json=updates, timeout=15)
    if resp.status_code not in (200, 204):
        print(f"[supabase] update {table} failed: {resp.status_code} {resp.text}")
        return False
    return True


# ---------- Domain-specific helpers ----------

def save_flag(repo: str, pr_number: int, github_comment_id, comment_type: str,
              file: str, line, severity: str, title: str, detail: str) -> dict | None:
    return insert_row("flags", {
        "repo": repo,
        "pr_number": pr_number,
        "github_comment_id": github_comment_id,
        "comment_type": comment_type,
        "file": file,
        "line": line,
        "severity": severity,
        "title": title,
        "detail": detail,
        "status": "open",
    })


def get_open_flags_for_pr(repo: str, pr_number: int) -> list:
    rows = select_rows("flags", {"repo": repo, "pr_number": pr_number, "status": "open"})
    return rows


def get_flag_by_comment_id(github_comment_id) -> dict | None:
    rows = select_rows("flags", {"github_comment_id": github_comment_id}, limit=1)
    return rows[0] if rows else None


def mark_flag_status(flag_id: str, status: str) -> bool:
    return update_row("flags", {"id": flag_id}, {"status": status})


def add_dismissed_pattern(repo: str, title_pattern: str):
    insert_row("dismissed_patterns", {"repo": repo, "title_pattern": title_pattern})


def get_dismissed_patterns(repo: str) -> list:
    rows = select_rows("dismissed_patterns", {"repo": repo}, limit=500)
    return [r["title_pattern"].lower() for r in rows]


def is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)
