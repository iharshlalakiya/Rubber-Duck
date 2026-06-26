"""
Full-repo context gathering (Tier 1).
Pulls every relevant code/text file from the target repo so the LLM can catch
real cross-file duplication and inconsistency.
"""

from rubber_duck.github_client import fetch_file_content, get_default_branch, list_all_repo_files

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
MAX_FILES_TO_SCAN = 150         # hard cap on GitHub API calls per review — protects against huge repos


def gather_repo_context(repo_full_name: str, changed_files: list) -> tuple[str, list]:
    """
    Tier 1 context: pull the FULL repo (every relevant code/text file),
    not just files matching keywords. This lets the model catch real
    cross-file duplication and inconsistency, not just guess from filenames.

    Files already in the diff are skipped here (the diff itself covers them) —
    we only need the *surrounding* codebase for comparison.

    Returns (context_text, all_paths) — all_paths is reused for stack detection.
    """
    ref = get_default_branch(repo_full_name)
    all_paths = list_all_repo_files(repo_full_name, ref, SKIP_PATH_PARTS, CODE_EXTENSIONS)

    if len(all_paths) > MAX_FILES_TO_SCAN:
        print(
            f"[gather_repo_context] {repo_full_name} has {len(all_paths)} candidate files, "
            f"capping at {MAX_FILES_TO_SCAN} to limit API calls/cost. "
            f"Consider Tier 2 (embeddings/RAG) for repos this size."
        )
        all_paths = all_paths[:MAX_FILES_TO_SCAN]

    context_chunks = []
    total_chars = 0
    files_fetched = 0

    for path in all_paths:
        if path in changed_files:
            continue  # already covered by the diff itself

        content = fetch_file_content(repo_full_name, path, ref)
        files_fetched += 1
        if not content:
            continue

        snippet = content[:MAX_FILE_CHARS]
        chunk = f"\n--- {path} ---\n{snippet}"

        if total_chars + len(chunk) > MAX_TOTAL_CONTEXT_CHARS:
            break  # budget hit, stop adding more files

        context_chunks.append(chunk)
        total_chars += len(chunk)

    print(f"[gather_repo_context] {repo_full_name}: fetched {files_fetched} files, {total_chars} context chars")
    return "\n".join(context_chunks), all_paths
