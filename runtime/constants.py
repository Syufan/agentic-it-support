"""State-machine / domain-rule constants.

These are the spec (from the state diagram), not deployment config: they do not
belong in Settings/.env and are not threaded through pure functions — runtime
modules import them directly.
"""

MAX_RESOLUTION_ATTEMPTS = 2

# Confidence thresholds.
# The confident "likely fix" wording (vs. a hedged "safe first step") shows at or above
# this. It equals the evidence ceiling in runtime/confidence.py — i.e. reached with two
# distinct successful tool sources.
CONFIDENCE_HIGH = 0.7
# Minimum evidence-based confidence before the agent may propose a fix — roughly one
# distinct successful tool source (see runtime/confidence.py). The RESOLVE *action*
# drives the RESOLVING phase; this is the gate (in diagnosis_policy) that authorizes it.
CONFIDENCE_RESOLVE_MIN = 0.35
