from state.case_state import CaseState


class SessionStore:
    def __init__(self) -> None:
        self._cases: dict[str, CaseState] = {}

    def create(self) -> CaseState:
        case = CaseState()
        self._cases[case.case_id] = case
        return case

    def get(self, case_id: str) -> CaseState | None:
        return self._cases.get(case_id)

    def delete(self, case_id: str) -> None:
        self._cases.pop(case_id, None)

    def active_count(self) -> int:
        return len(self._cases)


store = SessionStore()
