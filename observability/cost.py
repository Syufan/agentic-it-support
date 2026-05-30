"""Cost estimation for LLM token usage.

Pure function with a scalar signature so it stays dependency-free: callers pass
the per-1k rates (from Settings) rather than this module reaching into config.
"""


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cost_per_1k: float,
    completion_cost_per_1k: float,
) -> float:
    return round(
        prompt_tokens / 1000 * prompt_cost_per_1k
        + completion_tokens / 1000 * completion_cost_per_1k,
        6,
    )
