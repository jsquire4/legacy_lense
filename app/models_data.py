"""LLM model pricing reference data with provider routing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    input_cost_per_1m: float
    output_cost_per_1m: float


# Pricing per 1M tokens (input / output) as of 2026-03
MODELS: dict[str, ModelInfo] = {
    "gpt-3.5-turbo": ModelInfo("openai", 0.50, 1.50),
    "gpt-4o-mini": ModelInfo("openai", 0.15, 0.60),
    "gpt-4o": ModelInfo("openai", 2.50, 10.00),
    "gpt-4.1-nano": ModelInfo("openai", 0.10, 0.40),
    "gpt-4.1-mini": ModelInfo("openai", 0.40, 1.60),
    "gpt-4.1": ModelInfo("openai", 2.00, 8.00),
    "gpt-5-nano": ModelInfo("openai", 0.05, 0.40),
    "gpt-5-mini": ModelInfo("openai", 0.25, 2.00),
    "gpt-5.2": ModelInfo("openai", 1.75, 14.00),
    "gemini-2.0-flash": ModelInfo("gemini", 0.10, 0.40),
    "gemini-2.5-flash": ModelInfo("gemini", 0.30, 2.50),
    "gemini-2.5-pro": ModelInfo("gemini", 1.25, 10.00),
}


def get_provider(model: str) -> str:
    """Return the provider for a model name ('openai' or 'gemini').

    Raises ValueError for unknown models.
    """
    info = MODELS.get(model)
    if info:
        return info.provider
    raise ValueError(f"Unknown model: '{model}'")


def is_reasoning_model(model: str) -> bool:
    """Return True if the model is a reasoning model (GPT-5 series).

    Reasoning models reject temperature/top_p and need reasoning_effort instead.
    Non-OpenAI and unknown models are never treated as reasoning models.
    """
    try:
        return get_provider(model) == "openai" and model.startswith("gpt-5")
    except ValueError:
        return False


def uses_legacy_max_tokens(model: str) -> bool:
    """Return True if the model uses max_tokens instead of max_completion_tokens.

    Older models (gpt-3.5-turbo) don't support max_completion_tokens.
    Non-OpenAI and unknown models don't use this parameter.
    """
    try:
        return get_provider(model) == "openai" and model.startswith("gpt-3.5")
    except ValueError:
        return False
