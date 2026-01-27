"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys
import importlib

# Make this embedded copy act as top-level `lavis`
_pkg_root = os.path.abspath(os.path.dirname(__file__))
# let importlib treat this as a package rooted here
__path__ = [_pkg_root]
if _pkg_root not in sys.path:
	sys.path.insert(0, _pkg_root)

# Alias this module to top-level name so `import lavis.*` works
sys.modules.setdefault("lavis", sys.modules[__name__])

# Ensure subpackage modules (e.g., lavis.common.*) can resolve during early imports
# by preloading the namespace package entry before other submodules are imported.
importlib.import_module("lavis.common")

from omegaconf import OmegaConf
from lavis.common.registry import registry  # type: ignore


root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

# Avoid double-registration when the module is imported via both
# `models.dependence.gpt4point.lavis` and aliased `lavis` in the same process.
def _register_path_once(name: str, path: str) -> None:
	if name not in registry.mapping["paths"]:
		registry.register_path(name, path)
	else:
		# if the existing path differs, keep the original to stay deterministic
		# (mirrors typical idempotent init behavior)
		pass

_register_path_once("library_root", root_dir)
repo_root = os.path.join(root_dir, "..")
_register_path_once("repo_root", repo_root)
cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
_register_path_once("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])
