from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Student TODO: load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass

    state_dir = root / "state"
    profiles_dir = state_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)

    provider_settings = {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL"),
        },
        "custom": {
            "api_key": os.getenv("CUSTOM_API_KEY"),
            "base_url": os.getenv("CUSTOM_BASE_URL"),
        },
        "gemini": {
            "api_key": os.getenv("GEMINI_API_KEY"),
            "base_url": os.getenv("GEMINI_BASE_URL"),
        },
        "anthropic": {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "base_url": os.getenv("ANTHROPIC_BASE_URL"),
        },
        "ollama": {
            "api_key": None,
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        },
        "openrouter": {
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        },
    }

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "320")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "6")),
        model=ProviderConfig(
            provider=provider,
            model_name=model_name,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            api_key=provider_settings.get(provider, {}).get("api_key"),
            base_url=provider_settings.get(provider, {}).get("base_url"),
        ),
        judge_model=ProviderConfig(
            provider=judge_provider,
            model_name=judge_model_name,
            temperature=float(os.getenv("JUDGE_TEMPERATURE", "0.0")),
            api_key=provider_settings.get(judge_provider, {}).get("api_key"),
            base_url=provider_settings.get(judge_provider, {}).get("base_url"),
        ),
    )
