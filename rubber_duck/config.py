"""
Centralised configuration — loaded once at import time.
All other modules import from here instead of calling os.getenv() directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------- GitHub ----------
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # personal access token or GitHub App token

# ---------- Hugging Face LLM ----------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# Pick any instruction-tuned model available via HF Inference Providers.
# Good options for code review:
#   "Qwen/Qwen2.5-Coder-32B-Instruct"  (strong at code, recommended)
#   "meta-llama/Llama-3.1-8B-Instruct" (faster, weaker reasoning)
#   "mistralai/Mistral-7B-Instruct-v0.3"
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
