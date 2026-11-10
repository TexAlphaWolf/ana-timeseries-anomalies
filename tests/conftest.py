from __future__ import annotations

import os

import pytest

from tsanom import gen

POINTS = int(os.environ.get("TSANOM_TEST_POINTS", "20000"))
SEED = int(os.environ.get("TSANOM_TEST_SEED", "1"))


@pytest.fixture(scope="session")
def series() -> gen.Series:
    return gen.generate(POINTS, SEED)
