"""Tests that expose bugs in buggy_code — used as agent benchmark fixture."""

import pytest

from . import buggy_code


def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        buggy_code.divide(1, 0)


def test_parse_age_non_numeric():
    with pytest.raises(ValueError):
        buggy_code.parse_age("old")


def test_greet_none():
    result = buggy_code.greet(None)
    assert isinstance(result, str)
