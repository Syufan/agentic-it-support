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
