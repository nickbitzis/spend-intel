import pytest

from spendintel.pipeline import Config, run


@pytest.fixture(scope="session")
def results():
    """One small end to end run shared by the integration tests (about 5 seconds)."""
    return run(Config(lines=6000, parents=140, seed=11, labeled_share=0.15))
