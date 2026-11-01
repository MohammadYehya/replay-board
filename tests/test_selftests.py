"""Generated-lecture self-tests (SPEC 5.4). Slow: run with  pytest -m selftest"""
import pytest

from tests import selftest_board

pytestmark = pytest.mark.selftest


@pytest.mark.parametrize("case", list(selftest_board.CASES))
def test_board_selftest(case):
    sc = selftest_board.run_case(case)
    failed = [f"{name}: {detail}" for name, ok, detail in sc.checks if not ok]
    assert not failed, "; ".join(failed)
