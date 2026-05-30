"""State-machine / domain-rule constants.

These are the spec (from the state diagram), not deployment config: they do not
belong in Settings/.env and are not threaded through pure functions — runtime
modules import them directly.
"""

# Tool-call budgets per investigation mode
MAIN_TOOL_BUDGET = 5
RETRY_TOOL_BUDGET = 3
EXCEPTION_TOOL_BUDGET = 2

MAX_RESOLUTION_ATTEMPTS = 2

# Confidence thresholds
CONFIDENCE_HIGH = 0.8
CONFIDENCE_LOW = 0.5
