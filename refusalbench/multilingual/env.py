"""load repo .env and resolve bedrock judge settings without hardcoding secrets."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

JUDGE_ENV_KEYS: dict[str, str] = {
    "sonnet4_5": "REFUSALBENCH_JUDGE_SONNET4_5",
    "mistral-large-3": "REFUSALBENCH_JUDGE_MISTRAL_LARGE_3",
    "nemotron-3-super": "REFUSALBENCH_JUDGE_NEMOTRON_3_SUPER",
}


def load_project_env(
    repo_root: Path | None = None,
    env_file: Path | str | None = None,
) -> Path | None:
    """
    load variables from a .env file at the repository root.

    existing shell environment wins over .env (override=False).

    Parameters
    ----------
    repo_root
        repository root directory.
    env_file
        explicit env file path, or none to use repo_root/.env.

    Returns
    -------
    pathlib.Path or None
        path that was loaded, or none if the file is missing.
    """
    root = repo_root or REPO_ROOT
    if env_file is not None:
        path = Path(env_file)
        if not path.is_absolute():
            path = root / path
    else:
        path = root / ".env"
    if not path.is_file():
        return None

    load_dotenv(path, override=False)

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region and not os.environ.get("AWS_REGION_NAME"):
        os.environ["AWS_REGION_NAME"] = region

    normalize_bedrock_bearer_env()

    return path


def normalize_bedrock_bearer_env() -> None:
    """
    align bearer token env names for bedrock api key auth.

    litellm and boto3 expect AWS_BEARER_TOKEN_BEDROCK. if you set
    AWS_BEARER_TOKEN in .env, it is copied to that name when missing.
    """
    bedrock = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()
    generic = os.environ.get("AWS_BEARER_TOKEN", "").strip()
    if generic and not bedrock:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = generic
    elif bedrock and not generic:
        os.environ["AWS_BEARER_TOKEN"] = bedrock


def get_bedrock_bearer_token() -> str | None:
    """
    return the bedrock api key used as a bearer token, if configured.

    Returns
    -------
    str or None
        token value when set in the environment.
    """
    raw = (
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or os.environ.get("AWS_BEARER_TOKEN")
        or ""
    ).strip()
    return raw or None


def has_bedrock_auth_configured() -> bool:
    """
    check whether any supported aws auth path appears configured.

    Returns
    -------
    bool
        true when bearer token, access keys, or profile is set.
    """
    if get_bedrock_bearer_token():
        return True
    if os.environ.get("AWS_PROFILE"):
        return True
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return True
    return False


def get_aws_region(default: str = "us-east-1") -> str:
    """
    resolve aws region for bedrock calls.

    Parameters
    ----------
    default
        fallback when no region env var is set.

    Returns
    -------
    str
        region name.
    """
    return (
        os.environ.get("AWS_REGION_NAME")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or default
    )


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.lower()
    return (
        not value.strip()
        or "<paste" in lowered
        or "your_" in lowered
        or value.strip() == "..."
    )


def get_judge_models() -> dict[str, str]:
    """
    read litellm bedrock model ids for each judge from the environment.

    Returns
    -------
    dict[str, str]
        judge short name to litellm model id.

    Raises
    ------
    SystemExit
        when a required judge variable is missing or still a placeholder.
    """
    judges: dict[str, str] = {}
    missing: list[str] = []
    placeholder: list[str] = []

    for name, env_key in JUDGE_ENV_KEYS.items():
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            missing.append(env_key)
            continue
        if _looks_like_placeholder(raw):
            placeholder.append(env_key)
            continue
        judges[name] = raw

    if missing or placeholder:
        lines = [
            "bedrock judge model ids must be set in .env (see .env.example).",
        ]
        if missing:
            lines.append("missing: " + ", ".join(missing))
        if placeholder:
            lines.append("replace placeholders for: " + ", ".join(placeholder))
        raise SystemExit("\n".join(lines))

    return judges


def ensure_aws_credentials_hint() -> None:
    """
    log a short hint when no obvious aws auth env vars are present.

    does not raise, because boto3 may still resolve credentials from imds or sso cache.
    """
    if has_bedrock_auth_configured():
        return
    logger.warning(
        "no bedrock auth in environment after loading .env "
        "(set AWS_BEARER_TOKEN or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY or AWS_PROFILE). "
        "bedrock may still use ~/.aws/credentials or an instance role"
    )
