"""OpenAI model pricing reference data."""

# Pricing per 1M tokens (input / output) as of 2026-03
MODELS: dict[str, dict] = {
    "gpt-3.5-turbo": {"input_cost_per_1m": 0.50, "output_cost_per_1m": 1.50},
    "gpt-4o-mini": {"input_cost_per_1m": 0.15, "output_cost_per_1m": 0.60},
    "gpt-4o": {"input_cost_per_1m": 2.50, "output_cost_per_1m": 10.00},
    "gpt-4.1-nano": {"input_cost_per_1m": 0.10, "output_cost_per_1m": 0.40},
    "gpt-4.1-mini": {"input_cost_per_1m": 0.40, "output_cost_per_1m": 1.60},
    "gpt-4.1": {"input_cost_per_1m": 2.00, "output_cost_per_1m": 8.00},
    "gpt-5-nano": {"input_cost_per_1m": 0.05, "output_cost_per_1m": 0.40},
    "gpt-5-mini": {"input_cost_per_1m": 0.25, "output_cost_per_1m": 2.00},
    "gpt-5.2": {"input_cost_per_1m": 1.75, "output_cost_per_1m": 14.00},
}

MODEL_NAMES: list[str] = list(MODELS.keys())

def is_reasoning_model(model: str) -> bool:
    """Return True if the model is a reasoning model (GPT-5 series).

    Reasoning models reject temperature/top_p and need reasoning_effort instead.
    """
    return model.startswith("gpt-5")


def uses_legacy_max_tokens(model: str) -> bool:
    """Return True if the model uses max_tokens instead of max_completion_tokens.

    Older models (gpt-3.5-turbo) don't support max_completion_tokens.
    """
    return model.startswith("gpt-3.5")
