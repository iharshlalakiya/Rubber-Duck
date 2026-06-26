# Rubber Duck (MVP)

A GitHub webhook bot that reviews PRs: security blind spots, cross-file duplication,
"future hire confusion," and resurfaces old unresolved issues when related code
changes again. Each flag posts as its own inline comment, supports reply-based
dismissal/discussion, and `@RubberDuck` mentions for ad-hoc questions.

## 1. Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in .env with your real keys (see below)
```

### Get a GitHub token
- GitHub → Settings → Developer settings → Personal access tokens → Fine-grained token
- Give it `pull_requests: read & write`, `contents: read` on the repo you're testing against.

### Get a Hugging Face token
- huggingface.co → Settings → Access Tokens → create a "Read" token

### Pick a webhook secret
- Any random string, e.g. `openssl rand -hex 20`. Put the same value in `.env` and in GitHub's webhook config (step 3).

## 2. Run it locally

```bash
uvicorn main:app --reload --port 8000
```

To let GitHub reach your local server, tunnel it:

```bash
ngrok http 8000
```

Copy the `https://xxxx.ngrok-free.dev` URL it gives you.

## 3. Point a test repo's webhook at it

On the repo you want to test against:
- Repo → Settings → Webhooks → Add webhook
- Payload URL: `https://xxxx.ngrok-free.dev/webhook/github`
- Content type: `application/json`
- Secret: same string as `GITHUB_WEBHOOK_SECRET` in your `.env`

## 4. Set up Supabase (required for reply-threading, dismiss memory, and the ledger)

1. Create a free project at supabase.com
2. Go to SQL Editor → paste the contents of `supabase_schema.sql` → Run (choose "Run and enable RLS" if prompted — your bot uses the service role key, which bypasses RLS automatically)
3. Go to Project Settings → API → copy the **Project URL** and **service_role key** (NOT the anon key)
4. Put both in your `.env`:
   ```dotenv
   SUPABASE_URL=https://xxxx.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=eyJ...
   ```

If these aren't set, the bot still runs — it just skips reply-memory, dismiss tracking, and the tech debt ledger.

## 5. Add these webhook events on GitHub

Repo → Settings → Webhooks → your webhook → "Let me select individual events" → check:
- **Pull requests** (triggers the review, and marks flags resolved on merge)
- **Pull request review comments** (catches replies under a specific flag)
- **Issue comments** (catches `@RubberDuck` mentions anywhere on the PR)

## 6. Test it

Open a PR (or push a commit to an existing PR). Within a few seconds you should see
individual flag comments appear, each anchored to a specific line where possible.

## 7. What it actually checks (v1 checklist)

- **Supabase:** missing RLS on new tables, exposed service role key, `USING (true)` policies, missing ownership checks
- **FastAPI:** missing rate limiting on new endpoints, raw SQL injection risk, missing input validation
- **General:** duplicate logic vs. existing files, unexplained magic numbers, undocumented env vars
- Each flag includes a **confidence score** — flags below 0.6 are dropped automatically before posting (tune via `MIN_CONFIDENCE_THRESHOLD` in `main.py`)
- Each PR also gets a **Future-hire readiness score (X/10)** with a one-line reason

## 8. How interaction works

- **Reply directly under a flag's inline comment** → dismiss detection uses a fast keyword check first, falling back to an LLM intent check for less obvious phrasing (e.g. "nah skip that"). Dismissals are remembered per-repo and suppress similar future flags.
- **Comment `@RubberDuck <question>` anywhere on the PR** → answered using context from that PR's currently open flags.

## 9. Tech debt ledger (post-merge drift)

When a new PR touches a file that had an **old unresolved flag** from a previous PR, the bot
posts a "↩️ Resurfaced from earlier PRs" reminder listing those old flags. When a PR merges,
its still-open flags are marked `resolved` so they stop being tracked as outstanding (note:
this doesn't verify the fix actually happened — it just stops nagging once the PR is gone).

## 10. Cost/rate-limit awareness

- File fetches are cached within a run to avoid redundant GitHub API calls.
- Repo scanning is capped at `MAX_FILES_TO_SCAN` (default 150) — larger repos log a warning
  and only scan the first N files. That's the signal you'd want to move to embeddings-based
  retrieval (Tier 2) instead of a full repo dump.

## 11. Known MVP limitations

- Dismissed-pattern matching is still substring-based on flag titles, not deep semantic matching.
- No GitHub App install flow yet — uses a personal access token + manual webhook setup. Worth converting once you want other founders to install this on their own repos (see chat notes on JWT/installation-token auth).
- See `TESTING_GUIDE.md` for how to manually validate edge cases and model quality — these aren't things code alone can "finish."