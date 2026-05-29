import os

from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini-2024-07-18")

# Budget parameters (from state diagram)
MAIN_TOOL_BUDGET = 5
RETRY_TOOL_BUDGET = 3
EXCEPTION_TOOL_BUDGET = 2
MAX_RESOLUTION_ATTEMPTS = 2

# Confidence thresholds
CONFIDENCE_HIGH = 0.8
CONFIDENCE_LOW = 0.5

# Cost estimation (USD per 1K tokens; defaults track gpt-4o-mini list pricing)
LLM_PROMPT_COST_PER_1K = float(os.getenv("LLM_PROMPT_COST_PER_1K", "0.00015"))
LLM_COMPLETION_COST_PER_1K = float(os.getenv("LLM_COMPLETION_COST_PER_1K", "0.0006"))


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        prompt_tokens / 1000 * LLM_PROMPT_COST_PER_1K
        + completion_tokens / 1000 * LLM_COMPLETION_COST_PER_1K,
        6,
    )
