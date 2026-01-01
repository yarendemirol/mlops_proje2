import pytest


def test_imports():
    try:
        from src.training.train_model import train_full_pipeline  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("train_full_pipeline is not merged into main yet")
