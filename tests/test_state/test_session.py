import pytest
from state.case_state import Phase
from state.session import SessionStore


@pytest.fixture
def store():
    return SessionStore()


def test_create_returns_case(store):
    case = store.create()
    assert case is not None
    assert case.case_id is not None


def test_create_sets_intake_phase(store):
    case = store.create()
    assert case.phase == Phase.INTAKE


def test_get_returns_same_case(store):
    case = store.create()
    fetched = store.get(case.case_id)
    assert fetched is case


def test_get_unknown_id_returns_none(store):
    assert store.get("nonexistent") is None


def test_delete_removes_case(store):
    case = store.create()
    store.delete(case.case_id)
    assert store.get(case.case_id) is None


def test_delete_nonexistent_does_not_raise(store):
    store.delete("nonexistent")


def test_active_count(store):
    assert store.active_count() == 0
    store.create()
    store.create()
    assert store.active_count() == 2


def test_each_created_case_is_independent(store):
    case_a = store.create()
    case_b = store.create()
    case_a.facts["key"] = "value"
    assert case_b.facts == {}
