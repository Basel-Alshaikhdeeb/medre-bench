"""Dataset registry auto-discovery."""

import importlib
import pathlib
import pkgutil

_pkg_dir = pathlib.Path(__file__).parent
for _info in pkgutil.iter_modules([str(_pkg_dir)]):
    importlib.import_module(f".{_info.name}", __package__)
