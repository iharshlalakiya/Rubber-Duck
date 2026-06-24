# Rubber Duck Code Review Bot (MVP)

A GitHub webhook bot that reviews PRs for FastAPI + Supabase repos:
security blind spots, cross-file duplication, and "future hire confusion" —
posted as ONE digest comment per PR, not inline spam.

## 1. Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in .env with your real keys (see below)
```

### Get a GitHub token
- GitHub → Settings → Developer settings → Personal access tokens → Fine-grained token
- Give it `pull_requests: read & write` and `contents: read` on the repo you're testing against.

### Get an Anthropic API key
- console.anthropic.com → API Keys

### Pick a webhook secret
- Any random string, e.g. `openssl rand -hex 20`. Put the same value in `.env` and in GitHub's webhook config (step 3).

## 2. Run it locally

```bash
uvicorn main:app --reload --port 8000
```

To let GitHub reach your local server, tunnel it (e.g. with ngrok):

```bash
ngrok http 8000
```

Copy the `https://xxxx.ngrok.io` URL it gives you.

## 3. Point a test repo's webhook at it

On the OTHER repo you want to test against (your real FastAPI+Supabase project):
- Repo → Settings → Webhooks → Add webhook
- Payload URL: `https://xxxx.ngrok.io/webhook/github`
- Content type: `application/json`
- Secret: same string as `GITHUB_WEBHOOK_SECRET` in your `.env`
- Events: select "Pull requests" only

## 4. Test it

Open a PR on that test repo (or push a new commit to an existing PR).
Within a few seconds you should see a single comment posted on the PR from your bot account,
listing flags like missing RLS policies, missing rate limiting, duplicated logic, etc.

## 5. What it actually checks right now (v1 checklist)

- Supabase: missing RLS on new tables, exposed service role key, `USING (true)` policies, missing ownership checks
- FastAPI: missing rate limiting on new endpoints, raw SQL injection risk, missing input validation
- General: duplicate logic vs. existing files, unexplained magic numbers, undocumented env vars

## 6. Known MVP limitations (next steps, not yet built)

- No persistent "tech debt ledger" yet — every PR review is stateless (doesn't remember past flags). Would need a small DB (Postgres/SQLite) keyed by file+flag.
- No "mark as not relevant" learning loop yet — bot doesn't currently parse replies to comments.
- Context-gathering is a simple keyword heuristic (`auth`, `models`, `db`, etc.) — works okay for small repos, gets noisy/incomplete on large ones. A proper version would use embeddings/RAG over the repo.
- No GitHub App install flow — this MVP uses a personal access token + manual webhook setup. Fine for testing on your own repos, not for distributing to other users yet.
