from observability.cost import estimate_cost_usd

# Default gpt-4o-mini rates, kept here so the migration is value-preserving.
_PROMPT_RATE = 0.00015
_COMPLETION_RATE = 0.0006


def test_estimate_combines_prompt_and_completion():
    assert estimate_cost_usd(1000, 1000, _PROMPT_RATE, _COMPLETION_RATE) == round(
        _PROMPT_RATE + _COMPLETION_RATE, 6
    )


def test_estimate_zero_tokens_is_zero():
    assert estimate_cost_usd(0, 0, _PROMPT_RATE, _COMPLETION_RATE) == 0.0


def test_estimate_scales_with_tokens():
    assert estimate_cost_usd(2000, 0, _PROMPT_RATE, _COMPLETION_RATE) == round(
        2000 / 1000 * _PROMPT_RATE, 6
    )
