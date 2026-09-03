# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Meep API parity checker: compares meep_torch.mp against the real pymeep
surface dumped by scripts/dump_meep_api.py (scripts/meep_api.json).

For every meep class: present? same bases? which __init__ params accepted?
which methods present? Prints a compliance table and a summary score.

Run: uv run python scripts/check_meep_parity.py [--full]
  --full  list every missing method/param (default: counts only)
"""
import inspect
import json
import pathlib
import sys

import meep_torch.mp as tmp

HERE = pathlib.Path(__file__).parent
API = json.load(open(HERE / "meep_api.json"))
FULL = "--full" in sys.argv

# meep exposes swig internals + MPB solver etc.; grade the python-layer
# classes users script with (module filter keeps swig wrappers out)
PY_MODULES = ("meep.geom", "meep.source", "meep.simulation", "meep.solver",
              "meep.visualization", "meep.verbosity_mgr", "meep")


def params_of(obj):
    try:
        return [p.name for p in inspect.signature(obj.__init__).parameters.values()
                if p.name not in ("self", "args", "kwargs")]
    except (ValueError, TypeError):
        return []


# Out of scope by decision (2026-09-02, also master spec): meep-torch moves
# away from MPI — chunked domain decomposition, comms, and load balancing are
# replaced by whole-domain GPU tensors. Their classes are not parity targets.
MPI_EXCLUDED = {
    # python layer
    "BinaryPartition", "FragmentStatsVector",
    # C++ bindings: chunking/partitioning/comms
    "binary_partition", "split_plane", "fragment_stats",
    "structure_chunk", "fields_chunk", "dft_chunk", "comms_key",
    "comms_manager", "comms_operation", "comms_sequence", "chunkloop_field_components",
}
# SWIG container artifacts (exist only as binding plumbing, not API)
SWIG_ARTIFACTS = {
    "SwigPyIterator", "DoubleVector", "IntVector", "ComplexVector",
    "Size_t_Vector", "VolumeVector", "GridVolumeVector", "DftDataVector",
}

# Module FUNCTIONS excluded from parity, same MPI decision as MPI_EXCLUDED:
# chunk-tree partitioning / tiling / raw-buffer comms have no single-process
# meaning (the single-process parallel API itself -- am_master, sum_to_all,
# divide_parallel_processes, ... -- IS implemented in mp/functions.py).
MPI_EXCLUDED_FUNCS = {
    "calculate_tree", "split_by_binarytree", "split_into_tiles",
    "check_tiles",                  # binary-tree chunk decomposition
    "compute_fragment_stats",       # chunk cost model for load balancing
    "bw_or_to_all",                 # raw C buffer reduction (out-pointer)
}
# SWIG plumbing functions: operate on raw C++ pointers (structure*, fields*,
# dft_chunk*, material_data*, out-parameter buffers) that never surface in
# the python scripting layer; pymeep itself wraps every one of these behind
# Simulation/class methods which meep-torch implements instead.
SWIG_ARTIFACT_FUNCS = {
    # structure/geometry setup plumbing (Simulation.__init__ does this here)
    "create_structure", "init_libctl", "setup", "set_materials_from_geometry",
    "make_geom_epsilon", "gv2box", "make_volume_list",
    "create_absorber_list", "destroy_absorber_list", "add_absorbing_layer",
    # material_data* / material-grid C accessors (adjoint tier wraps these)
    "material_free", "make_file_material", "make_user_material",
    "make_material_grid", "is_material_grid", "epsilon_file_material",
    "epsilon_material_grid", "material_grid_grad", "material_grid_val",
    "material_grids_addgradient", "matgrid_grad", "matgrid_val",
    "update_weights",
    # raw dft_chunk/fields accessors (use Simulation.get_dft_array etc.)
    "get_dft_fields_array", "get_dft_flux_array", "get_dft_force_array",
    "get_dft_near2far_array", "load_dft_hdf5", "save_dft_hdf5",
    "get_eigenmode_coefficients_and_kpoints", "eigenmode_amplitude",
    # C internals with out-parameter buffers / C callback factories
    "do_harminv",                   # py_do_harminv is the API; implemented
    "green2d", "green3d",           # near2far Green's-function internals
    "make_casimir_gfunc", "make_casimir_gfunc_kz",
    "matrix_invert", "get_epsilon_grid", "derived_component_func",
}


def tier_of(name, spec):
    # lowercase names in module "meep" are SWIG bindings of the C++ core
    # (fields, structure_chunk, dft_chunk, ivec, ...); CamelCase in the
    # meep.* python modules are the user scripting layer
    if spec["module"] == "meep" and name[0].islower():
        return "cpp-binding"
    return "python-layer"


rows, n_ok, n_partial, n_missing = [], 0, 0, 0
tier_counts = {"python-layer": [0, 0, 0], "cpp-binding": [0, 0, 0]}  # ok/part/miss
n_excluded = 0
for name, spec in sorted(API["classes"].items()):
    if spec["module"] not in PY_MODULES:
        continue
    if name in MPI_EXCLUDED or name in SWIG_ARTIFACTS:
        n_excluded += 1
        continue
    tier = tier_of(name, spec)
    have = getattr(tmp, name, None)
    if have is None or not inspect.isclass(have):
        rows.append((name, "MISSING", tier, ""))
        n_missing += 1
        tier_counts[tier][2] += 1
        continue
    # filter *args/**kwargs on the spec side too (params_of already drops them
    # on the have side; a `**kwargs` signature can otherwise never grade OK)
    want_p = set(spec["init_params"] or []) - {"args", "kwargs"}
    have_p = set(params_of(have))
    missing_p = sorted(want_p - have_p)
    want_m = set(spec["methods"])
    have_m = {m for m in dir(have) if not m.startswith("_")}
    missing_m = sorted(want_m - have_m)
    want_b = set(spec["bases"])
    have_b = {b.__name__ for b in have.__bases__}
    base_ok = want_b <= have_b or not want_b
    if not missing_p and not missing_m and base_ok:
        rows.append((name, "OK", tier, ""))
        n_ok += 1
        tier_counts[tier][0] += 1
    else:
        n_partial += 1
        tier_counts[tier][1] += 1
        pd = f"-{len(missing_p)} params" + (f" {missing_p}" if FULL else "")
        md = f"-{len(missing_m)} methods" + (f" {missing_m}" if FULL else "")
        bd = "" if base_ok else f"bases {sorted(want_b)} != {sorted(have_b)}"
        rows.append((name, "PARTIAL", tier,
                     " ".join(x for x in (pd if missing_p else "",
                                          md if missing_m else "", bd) if x)))

w = max(len(r[0]) for r in rows) + 1
for tier in ("python-layer", "cpp-binding"):
    print(f"\n== {tier} ==")
    for name, st, t, detail in rows:
        if t == tier:
            print(f"{name:<{w}} {st:<8} {detail}".rstrip())

total = n_ok + n_partial + n_missing
print(f"\nALL graded classes: {total}  OK: {n_ok}  PARTIAL: {n_partial}  "
      f"MISSING: {n_missing}  (excluded {n_excluded}: MPI/chunking + SWIG artifacts)")
for tier, (ok, part, miss) in tier_counts.items():
    t = ok + part + miss
    if t:
        print(f"  {tier:13s}: {t:3d} classes — OK {ok}, PARTIAL {part}, MISSING {miss} "
              f"(structural parity {100 * ok / t:.0f}%, presence {100 * (ok + part) / t:.0f}%)")

fn_excluded = [n for n in API["functions"]
               if n in MPI_EXCLUDED_FUNCS or n in SWIG_ARTIFACT_FUNCS]
fn_missing = [n for n in API["functions"]
              if n not in fn_excluded and not hasattr(tmp, n)]
print(f"module functions: {len(API['functions'])}  missing: {len(fn_missing)}"
      f"  (excluded {len(fn_excluded)}: MPI/chunking + SWIG plumbing)")
if FULL:
    print("missing functions:", sorted(fn_missing))
sys.exit(0 if n_missing == 0 else 1)
