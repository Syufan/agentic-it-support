"""Tool-budget computation: how much of a budget mode is left / spent.

Governance logic, so it lives in runtime/ and may read config. The state layer
stays config-free: it only defines the BudgetMode enum.
"""

from config import EXCEPTION_TOOL_BUDGET, MAIN_TOOL_BUDGET, RETRY_TOOL_BUDGET
from state.case_state import BudgetMode

_LIMITS: dict[BudgetMode, int] = {
    BudgetMode.MAIN: MAIN_TOOL_BUDGET,
    BudgetMode.RETRY: RETRY_TOOL_BUDGET,
    BudgetMode.EXCEPTION: EXCEPTION_TOOL_BUDGET,
}


def remaining(mode: BudgetMode, used: int) -> int:
    return max(0, _LIMITS[mode] - used)


def exhausted(mode: BudgetMode, used: int) -> bool:
    return used >= _LIMITS[mode]


def limit(mode: BudgetMode) -> int:
    return _LIMITS[mode]
