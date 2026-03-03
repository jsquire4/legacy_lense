"""OpenAI model pricing reference data."""

# Pricing per 1M tokens (input / output) as of 2025-05
MODELS: dict[str, dict] = {
    "gpt-3.5-turbo": {"input_cost_per_1m": 0.50, "output_cost_per_1m": 1.50},
    "gpt-4o-mini": {"input_cost_per_1m": 0.15, "output_cost_per_1m": 0.60},
    "gpt-4o": {"input_cost_per_1m": 2.50, "output_cost_per_1m": 10.00},
    "gpt-4.1-nano": {"input_cost_per_1m": 0.10, "output_cost_per_1m": 0.40},
    "gpt-4.1-mini": {"input_cost_per_1m": 0.40, "output_cost_per_1m": 1.60},
    "gpt-4.1": {"input_cost_per_1m": 2.00, "output_cost_per_1m": 8.00},
    "gpt-5": {"input_cost_per_1m": 1.25, "output_cost_per_1m": 10.00},
    "gpt-5.1": {"input_cost_per_1m": 1.25, "output_cost_per_1m": 10.00},
    "gpt-5.2": {"input_cost_per_1m": 1.75, "output_cost_per_1m": 14.00},
}

MODEL_NAMES: list[str] = list(MODELS.keys())
