"""Root conftest: make ``src/`` importable without installing the package.

This lets ``python -m pytest`` work directly from a checkout on any pytest
version (the ``pythonpath`` ini option only exists from pytest 7). Installing
the package with ``pip install -e .`` also works and makes this a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
