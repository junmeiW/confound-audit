"""Compatibility shim.

Editable installs (``pip install -e .``) with older setuptools require a
``setup.py``; all real metadata lives in ``pyproject.toml``. Newer setuptools
and all PEP 517 frontends use ``pyproject.toml`` alone.
"""

from setuptools import setup

setup()
