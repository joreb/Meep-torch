# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Dump real pymeep's public API surface (classes, __init__ params, methods)
to JSON for the parity checker.

Run: ~/miniforge3/envs/meep-oracle/bin/python scripts/dump_meep_api.py scripts/meep_api.json
(writes to the file argument — meep prints a runtime banner to stdout at exit)
"""
import inspect
import json
import sys

import meep as mp


def params_of(fn):
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return None
    return [p.name for p in sig.parameters.values() if p.name not in ("self",)]


out = {"classes": {}, "functions": {}}
for name in sorted(dir(mp)):
    if name.startswith("_"):
        continue
    obj = getattr(mp, name)
    if inspect.isclass(obj):
        methods = sorted(
            m for m, v in vars(obj).items()
            if not m.startswith("_") and callable(v))
        out["classes"][name] = {
            "init_params": params_of(obj.__init__),
            "methods": methods,
            "bases": [b.__name__ for b in obj.__bases__ if b.__name__ != "object"],
            "module": getattr(obj, "__module__", ""),
        }
    elif inspect.isfunction(obj):
        out["functions"][name] = {"params": params_of(obj)}

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=1)
