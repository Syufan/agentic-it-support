from state.case_state import CaseState


class LLMInput:
    pass


def build_messages(case: CaseState) -> LLMInput:
    raise NotImplementedError
