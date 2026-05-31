"""State-machine / domain-rule constants.

These are the spec (from the state diagram), not deployment config: they do not
belong in Settings/.env and are not threaded through pure functions — runtime
modules import them directly.
"""

MAX_RESOLUTION_ATTEMPTS = 2

# Confidence thresholds
CONFIDENCE_HIGH = 0.8
CONFIDENCE_LOW = 0.5
