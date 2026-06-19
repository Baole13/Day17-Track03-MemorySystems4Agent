from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Student TODO: map aliases like `anthorpic` -> `anthropic`."""
    normalized = value.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "openai": "openai",
        "custom": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "googlegenai": "gemini",
        "anthropic": "anthropic",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "ollama": "ollama",
        "openrouter": "openrouter",
        "router": "openrouter",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported provider: {value}")
    return aliases[normalized]


def build_chat_model(config: ProviderConfig):
    """Student TODO: instantiate the real chat model for the selected provider.

    Pseudocode:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        chat_cls = getattr(import_module("langchain_openai"), "ChatOpenAI")
        return chat_cls(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "custom":
        chat_cls = getattr(import_module("langchain_openai"), "ChatOpenAI")
        return chat_cls(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    if provider == "gemini":
        chat_cls = getattr(import_module("langchain_google_genai"), "ChatGoogleGenerativeAI")
        return chat_cls(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        chat_cls = getattr(import_module("langchain_anthropic"), "ChatAnthropic")
        return chat_cls(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "ollama":
        chat_cls = getattr(import_module("langchain_ollama"), "ChatOllama")
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return chat_cls(**kwargs)

    if provider == "openrouter":
        chat_cls = getattr(import_module("langchain_openrouter"), "ChatOpenRouter")
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return chat_cls(**kwargs)

    raise ValueError(f"Unsupported provider: {config.provider}")
