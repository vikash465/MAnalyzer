import logging

from app.services.llm.base import BaseLLM
from app.services.llm.claude import ClaudeLLM
from app.services.llm.openai import OpenAILLM

logger = logging.getLogger(__name__)

_PROVIDERS: dict[str, type[BaseLLM]] = {
    "claude": ClaudeLLM,
    "openai": OpenAILLM,
}


def get_llm(provider: str | None = None, **kwargs) -> BaseLLM:
    """Factory that returns the requested LLM provider instance.

    Falls back to the default provider from settings when *provider* is None.
    """
    if provider is None:
        from app.config import get_settings
        provider = get_settings().default_llm_provider

    cls = _PROVIDERS.get(provider)
    if cls is None:
        available = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown LLM provider '{provider}'. Available: {available}")

    return cls(**kwargs)


def get_llm_from_request(request, provider: str | None = None) -> BaseLLM:
    """Build an LLM instance, preferring the user's encrypted key from headers."""
    encrypted_token = request.headers.get("X-LLM-Key", "")
    if encrypted_token:
        try:
            from app.services.crypto import decrypt_value
            decrypted = decrypt_value(encrypted_token)
            token_provider, _, api_key = decrypted.partition(":")
            if api_key:
                use_provider = provider or token_provider
                return get_llm(provider=use_provider, api_key=api_key)
        except (ValueError, Exception):
            logger.warning("Invalid X-LLM-Key header, falling back to server key")

    return get_llm(provider=provider)


def register_provider(name: str, cls: type[BaseLLM]) -> None:
    """Register a custom LLM provider at runtime."""
    _PROVIDERS[name] = cls
