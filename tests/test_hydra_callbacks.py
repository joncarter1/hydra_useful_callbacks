#!/usr/bin/env python

"""Tests for example package."""

import os

import pytest


@pytest.fixture(scope='function')  # https://docs.pytest.org/en/6.2.x/fixture.html#fixture-scopes
def my_fixture():
    """Example fixture."""
    # Fixtures can be used to configure the test environment.
    os.environ['MY_VAL'] = '1'
    # Fixtures can provide data to a test.
    yield '2'
    # Tear-down can be performed after a yield to restore state.
    del os.environ['MY_VAL']


def test_env_var_is_set(my_fixture):
    assert os.environ['MY_VAL'] == '1'
    assert my_fixture == '2'
