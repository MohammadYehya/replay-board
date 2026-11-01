"""Generated-lecture self-tests (SPEC 5.4). Slow: run with  pytest -m selftest"""
import pytest

from tests import selftest_board

pytestmark = pytest.mark.selftest


@pytest.mark.parametrize("case", list(selftest_board.CASES))
def test_board_selftest(case):
    sc = selftest_board.run_case(case)
    failed = [f"{name}: {detail}" for name, ok, detail in sc.checks if not ok]
    assert not failed, "; ".join(failed)


@pytest.mark.parametrize("case", ["base", "dark", "fade", "pip"])
def test_slides_selftest(case):
    from tests import selftest_slides

    sc = selftest_slides.run_case(case)
    failed = [f"{name}: {detail}" for name, ok, detail in sc.checks if not ok]
    assert not failed, "; ".join(failed)
