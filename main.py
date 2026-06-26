"""
Rubber Duck — MVP
Entry point: creates the FastAPI app and registers the two HTTP endpoints.
All business logic lives in the rubber_duck/ package.
"""

import json

from fastapi import FastAPI, Request, HTTPException

from rubber_duck.github_client import verify_signature
from rubber_duck.config import GITHUB_WEBHOOK_SECRET
from rubber_duck.handlers import (
    handle_pull_request_event,
    handle_review_comment_reply,
    handle_mention_comment,
)

app = FastAPI()


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

    if event == "pull_request":
        return handle_pull_request_event(payload)
    elif event == "pull_request_review_comment" and payload.get("action") == "created":
        return handle_review_comment_reply(payload)
    elif event == "issue_comment" and payload.get("action") == "created":
        return handle_mention_comment(payload)
    else:
        return {"status": "ignored", "reason": f"event {event} not handled"}


@app.get("/health")
def health():
    return {"status": "ok"}