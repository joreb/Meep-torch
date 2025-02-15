# MPB track — band structures via Bloch + harminv (spec)

Date: 2026-08-09
Workstream: MPB track wave 1-2 (the ~15 MPB-blocked stubs; master plan's
separate Q1 track picked up here). No plane-wave-expansion solver: bands
come from the FDTD route meep itself uses in holey-wvg-bands — Bloch-
periodic cell at fixed k, broadband pulse, harminv the ringdown ->
omega_n(k). The mpb oracle (meep.mpb in the conda env) gates the truly
periodic cases.

## Physics / machinery (pinned)

- **Wave 1 — waveguide bands** (holey-wvg-bands): unit cell periodic
  along the waveguide axis + PML transverse. That is the EXISTING TE
  Bloch stepper (step_bloch_te: periodic y + x-CPML) with meep's (x, y)
  mapped to our (y-periodic, x-transverse); k_x^meep = our k_y. Hz-pol
  (meep) = our TE. Bands: per k in the sweep, pulse -> harminv Hz ->
  modes with Q above a threshold are the bands.
- **Wave 2 — fully periodic 2D lattices**: new `step_bloch2_tm` — TM
  envelope two-quadrature stepper periodic in BOTH axes (x-roll like the
  y-roll, no PML anywhere), Bloch couplings i k_x and i k_y. Rectangular
  lattices only (oblique/tri lattice basis deferred). Bands via a
  low-symmetry seed pulse + harminv Ez at a probe.
- Bands are lossless -> harminv undamped handling (W3 Q=inf) applies;
  select modes by amp with Q > 50.

## In scope

1. `examples/holey-wvg-bands.py` port + meep-oracle JSON gate.
2. `core/stepper_bloch2.py` (doubly periodic TM) + square-lattice rod
   bands vs the meep.mpb oracle.
3. `examples/mpb_bragg_sine.py` port: 1D sinusoidal-eps Bragg bands
   (doubly-periodic stepper with ny minimal, k_x sweep) vs mpb oracle.

## Out of scope

Oblique (tri/honeycomb) lattice basis vectors, 3D slab bands
(mpb_hole_slab), mode-profile output (mpb_data_analysis), strip
dispersion (mpb_strip needs the effective-index route instead) — stubs
re-scoped accordingly.

## Gates

- (a) holey-wvg-bands: first two guided bands at 5 k-points within 1%
  of the meep-oracle harminv values.
- (b) square lattice (a = 1, r = 0.2, eps = 8.9 rods, TM): first two
  bands at Gamma-X k-points within 2% of meep.mpb.
- (c) bragg_sine (1D, eps = 6 + 5 sin): first gap edges at k = pi/a
  within 2% of meep.mpb.

## Tests / acceptance

`tests/examples/test_mpb_track.py` vs `holey_bands.json`,
`sq_lattice_mpb.json`, `bragg_sine_mpb.json`. Oracles via
`mamba run -n meep-oracle`. Suite addition <10 min. MIT, <=500
lines/file, TDD.
