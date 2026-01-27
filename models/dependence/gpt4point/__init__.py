"""Anchor module that exposes `lavis` as a top-level package when used inside PointQA."""

import importlib
import os
import sys


def _ensure_lavis_importable():
    pkg_root = os.path.abspath(os.path.dirname(__file__))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    if "lavis" not in sys.modules:
        try:
            importlib.import_module("lavis")
        except ModuleNotFoundError:
            pass


_ensure_lavis_importable()