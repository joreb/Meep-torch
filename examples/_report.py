# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""One JSON per run, so a port and its oracle twin can be compared as data.

WHY THIS EXISTS. An example already ends with a line carrying a number
(``example_template.py`` rule 7) and already writes a CSV beside every figure
(rule 4). Neither is comparable across engines without a human in the loop: the
closing line is prose, and two CSVs of different lengths need a reader to say
which column means what. This writes the same numbers a third time, as named
scalars, which is the form a diff can consume.

THE FOLDER IS PER EXPERIMENT, NOT PER SCRIPT. One experiment has up to three
scripts, and they are three different claims:

    chirped_pulse.py        the meep-torch native API        -> chirped_pulse.json
    chirped_pulse_mp.py     upstream's own script, run on    -> chirped_pulse_mp.json
                            the meep_torch.mp shim
    chirped_pulse_meep.py   the same script on real pymeep   -> chirped_pulse_meep.json

All three land in ``output/chirped_pulse/``. That is the whole trick: the
pairing a comparison needs falls out of the layout instead of out of a
name-mapping table that can drift, and it is the ``examples/10_periodic_mpb``
layout, where each script already owns a subdirectory of ``output/``.

The ``_mp`` against ``_meep`` comparison is the load-bearing one, because those
two are the SAME SCRIPT and differ only in which engine the import resolves to.
A difference there is the engine's. The native ``.py`` against ``_meep`` is a
weaker claim -- it can also differ because the port was written differently --
so it is compared too, and reported separately rather than averaged in.

NO TORCH IMPORT AT MODULE LEVEL, EVER. The twin runs under the meep-oracle
conda env, which has pymeep and numpy and no torch. A ``import torch`` here
would make every twin die on import, and the failure would look like a meep
problem. Tensors are handled by duck-typing ``.detach`` instead.

NO TIMESTAMP. A run stamp makes every JSON differ from the last one and turns
a real regression into one line among many in the diff. What identifies a run
is the engine and the device, both of which are recorded.
"""
import json
import pathlib

import numpy as np


def _plain(v):
    """A JSON-able version of ``v``, preserving what a comparison needs.

    Complex becomes ``{"re": ..., "im": ...}`` rather than ``abs(v)``: a phase
    error and an amplitude error are different bugs, and collapsing to the
    modulus hides the first one completely. That is not hypothetical here --
    the source-injection sign bug was a pure phase flip with an unchanged
    magnitude.
    """
    if hasattr(v, "detach"):                     # torch tensor, without importing torch
        v = v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        if np.iscomplexobj(v):
            return {"re": v.real.tolist(), "im": v.imag.tolist()}
        return v.tolist()
    if isinstance(v, (np.generic,)):
        v = v.item()
    if isinstance(v, complex):
        return {"re": v.real, "im": v.imag}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def _split(stem):
    """``"foo_mp"`` -> ``("foo", "mp")``. The suffix is the CODE PATH, not the
    engine: ``mp`` and ``native`` are both meep-torch, reached two ways."""
    for suffix, kind in (("_meep", "meep"), ("_mp", "mp")):
        if stem.endswith(suffix):
            return stem[:-len(suffix)], kind
    return stem, "native"


def record(script, **values):
    """Write ``output/<experiment>/<stem>.json`` and return its path.

    ``script`` is the caller's ``__file__``. Every keyword becomes a named
    entry; pass the same names from the port and from the twin, because a name
    present on one side only is reported as a gap rather than silently skipped.
    """
    path = pathlib.Path(script).resolve()
    stem = path.stem
    experiment, kind = _split(stem)
    out = path.parent / "output" / experiment
    out.mkdir(parents=True, exist_ok=True)
    payload = {"engine": "meep" if kind == "meep" else "meep-torch",
               "path": kind, "script": stem}
    payload.update({k: _plain(v) for k, v in values.items()})
    dest = out / f"{stem}.json"
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return dest


if __name__ == "__main__":
    # The smallest self-check that fails if the layout rule breaks: a port and
    # a twin must land in ONE folder under two names.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        kw = dict(peak=1.5, curve=np.array([1.0, 2.0]), amp=complex(0.0, -1.0))
        a = record(pathlib.Path(d) / "demo.py", **kw)
        m = record(pathlib.Path(d) / "demo_mp.py", **kw)
        b = record(pathlib.Path(d) / "demo_meep.py", **kw)
        assert a.parent == b.parent == m.parent, (a, m, b)
        assert a.parent.name == "demo", a.parent
        assert {a.name, m.name, b.name} == {"demo.json", "demo_mp.json",
                                            "demo_meep.json"}
        assert json.loads(m.read_text())["path"] == "mp"
        assert json.loads(m.read_text())["engine"] == "meep-torch"
        got = json.loads(a.read_text())
        assert got["engine"] == "meep-torch" and got["curve"] == [1.0, 2.0]
        assert got["amp"] == {"re": 0.0, "im": -1.0}, got["amp"]
        assert json.loads(b.read_text())["engine"] == "meep"
        print("_report self-check ok:", a.parent)
