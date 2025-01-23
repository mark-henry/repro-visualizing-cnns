import pytest
from types import SimpleNamespace

@pytest.fixture
def small_config():
    """Config with small number of channels for testing"""
    return SimpleNamespace(
        conv1_channels=2,
        conv2_channels=2,
        kernel_size=3,
        pool_size=2,
        fc_units=10
    )

@pytest.fixture
def normal_config():
    """Config with normal number of channels for testing"""
    return SimpleNamespace(
        conv1_channels=32,
        conv2_channels=64,
        kernel_size=3,
        pool_size=2,
        fc_units=10
    ) 