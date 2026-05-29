import pytest
from state.budget import exhausted, limit, remaining
from state.case_state import BudgetMode


@pytest.mark.parametrize("mode,expected", [
    (BudgetMode.MAIN, 5),
    (BudgetMode.RETRY, 3),
    (BudgetMode.EXCEPTION, 2),
])
def test_limit(mode, expected):
    assert limit(mode) == expected


@pytest.mark.parametrize("mode,used,expected", [
    (BudgetMode.MAIN, 0, 5),
    (BudgetMode.MAIN, 3, 2),
    (BudgetMode.MAIN, 5, 0),
    (BudgetMode.RETRY, 3, 0),
    (BudgetMode.EXCEPTION, 1, 1),
])
def test_remaining(mode, used, expected):
    assert remaining(mode, used) == expected


def test_remaining_never_negative():
    assert remaining(BudgetMode.MAIN, 99) == 0


@pytest.mark.parametrize("mode,used,expected", [
    (BudgetMode.MAIN, 4, False),
    (BudgetMode.MAIN, 5, True),
    (BudgetMode.MAIN, 6, True),
    (BudgetMode.RETRY, 2, False),
    (BudgetMode.RETRY, 3, True),
])
def test_exhausted(mode, used, expected):
    assert exhausted(mode, used) == expected
