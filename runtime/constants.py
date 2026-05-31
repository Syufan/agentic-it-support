"""State-machine / domain-rule constants.

These are the spec (from the state diagram), not deployment config: they do not
belong in Settings/.env and are not threaded through pure functions — runtime
modules import them directly.
"""

MAX_RESOLUTION_ATTEMPTS = 2

# Confidence thresholds
CONFIDENCE_HIGH = 0.8
# Minimum evidence-based confidence before the agent may propose a fix — roughly one
# distinct successful tool source (see runtime/confidence.py). The RESOLVE *action*
# drives the RESOLVING phase; this is the gate (in diagnosis_policy) that authorizes it.
CONFIDENCE_RESOLVE_MIN = 0.35
