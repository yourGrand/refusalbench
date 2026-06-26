#!/usr/bin/env python3
"""Smoke-test Bedrock model access via LiteLLM."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import litellm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "refusalbench" / "naturalquestions"))

from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME

os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_REGION_NAME"] = AWS_REGION_NAME

DEFAULT_MODELS = [
    "bedrock/us.anthropic.claude-sonnet-4-6",
]


async def try_model(model_id: str) -> None:
    try:
        response = await litellm.acompletion(
            model=model_id,
            messages=[{"role": "user", "content": "Reply with exactly the word: ok"}],
            max_tokens=16,
        )
        text = (response.choices[0].message.content or "").strip()
        print(f"PASS  {model_id}")
        print(f"      response: {text!r}")
    except Exception as exc:
        print(f"FAIL  {model_id}")
        print(f"      error: {exc}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Bedrock models.")
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()
    print(f"Region: {AWS_REGION_NAME}\n")
    for model_id in args.models:
        await try_model(model_id)
        print("---")


if __name__ == "__main__":
    asyncio.run(main())
