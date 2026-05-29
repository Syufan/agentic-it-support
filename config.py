import os

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL_ID = "claude-sonnet-4-6"

# Budget parameters (from state diagram)
MAIN_TOOL_BUDGET = 5
RETRY_TOOL_BUDGET = 3
EXCEPTION_TOOL_BUDGET = 2
MAX_RESOLUTION_ATTEMPTS = 2

# Confidence thresholds
CONFIDENCE_HIGH = 0.8
CONFIDENCE_LOW = 0.5
