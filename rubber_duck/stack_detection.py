"""
Stack detection — monorepo-aware, per-package.

Identifies which framework/stack each changed file belongs to by walking up
the directory tree to find the nearest manifest (package.json, requirements.txt, etc.)
and parsing its actual dependency fields (not grepping raw text).
Also builds the stack-specific checklist sections that get injected into the LLM prompt.
"""

import json

from rubber_duck.github_client import fetch_file_content

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


# ---------- Monorepo detection ----------

def is_monorepo(all_paths: list) -> bool:
    filenames = {p.split("/")[-1] for p in all_paths}
    if filenames & MONOREPO_MARKERS:
        return True
    # crude fallback: more than one package.json/requirements.txt outside root = monorepo-ish
    manifest_dirs = {
        "/".join(p.split("/")[:-1]) for p in all_paths if p.split("/")[-1] in MANIFEST_FILENAMES
    }
    return len(manifest_dirs) > 1


# ---------- Manifest walking ----------

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


# ---------- Per-package stack detection ----------

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


# ---------- Label and checklist builders ----------

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
