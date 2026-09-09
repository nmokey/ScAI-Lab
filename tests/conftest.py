"""
Shared test configuration.

Import shims
------------
Neither `scripts/` nor `vlm/` is an installable package:

  * `scripts/*.py` are standalone scripts with no `__init__.py`.
  * `vlm/` uses bare intra-package imports (`from utils.misc_utils import ...`)
    that only resolve when CWD is `vlm/`; the entry points in `vlm/run/` fix this
    with a `sys.path.insert` at the top of the file.

Rather than restructure the research code (which would churn every entry point
and every documented command line), we replicate that path setup here and expose
`load_script_module()` for importing a script by path.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
VLM_DIR = REPO_ROOT / "vlm"

# `vlm/` on sys.path mirrors what vlm/run/*.py do, so `from data.eval import ...`
# and `from utils.misc_utils import ...` resolve the same way in tests as in prod.
for _p in (str(REPO_ROOT), str(VLM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Headless plotting: several scripts import pyplot at module scope.
os.environ.setdefault("MPLBACKEND", "Agg")


def load_script_module(name):
    """
    Import a module from scripts/ by filename stem, e.g. load_script_module("train_longitudinal").

    Uses importlib rather than a plain import because scripts/ has no __init__.py
    and the filenames are not importable as a package.
    """
    import importlib.util

    path = SCRIPTS_DIR / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"No such script: {path}")

    cached = sys.modules.get(f"_scai_script_{name}")
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(f"_scai_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Don't leave a half-initialised module cached: a missing dependency would
        # otherwise surface as a confusing AttributeError in every later test.
        sys.modules.pop(spec.name, None)
        raise
    return module


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def scripts():
    """Lazy accessor for scripts/ modules: scripts("train_longitudinal")."""
    return load_script_module


@pytest.fixture(scope="session")
def data_root():
    """
    Real dataset root, or skip. Set SCAI_DATA_ROOT to the processed NIfTI dir
    (the `output_dir` from config.yaml) to enable `realdata` tests.
    """
    root = os.environ.get("SCAI_DATA_ROOT")
    if not root or not Path(root).exists():
        pytest.skip("SCAI_DATA_ROOT not set or missing — real-data test skipped")
    return Path(root)


def pytest_collection_modifyitems(config, items):
    """Skip `realdata` tests unless SCAI_DATA_ROOT is present."""
    if os.environ.get("SCAI_DATA_ROOT"):
        return
    skip = pytest.mark.skip(reason="SCAI_DATA_ROOT not set")
    for item in items:
        if "realdata" in item.keywords:
            item.add_marker(skip)
