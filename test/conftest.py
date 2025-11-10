import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def enable_test_config():
    os.environ["USE_TEST_CONFIG"] = "1"
