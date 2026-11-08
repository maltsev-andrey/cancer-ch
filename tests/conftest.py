"""Shared fixtures. The pipeline modules compute their data paths at import time from
config/sources.yaml; tests point CANCER_CH_DATA at a temporary directory so nothing in a
real data tree is read or written."""
import importlib.util
import os
import sys
from pathlib import Path

import numpyro
numpyro.set_host_device_count(4)        # before jax initialises: parallel chains in the slow test

import pytest

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("CANCER_CH_DATA", str(Path(os.environ.get("TMPDIR", "/tmp")) / "cancer-ch-test-data"))


def load_module(name, rel):
    """Import pipeline/<file>.py as a module even though the file names start with digits."""
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def s04():
    return load_module("s04", "pipeline/s04_harmonise.py")


@pytest.fixture(scope="session")
def s06():
    return load_module("s06", "pipeline/s06_apc.py")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: MCMC tests, minutes on CPU; run with -m slow")
