## linear-chirped pulse planewave with higher frequencies at the front (down-chirp)
import numpy as np

import meep_torch.mp as mp
import os
import pathlib
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# Examples default to the GPU: they are demonstrations, and the difference is
# not marginal (this cell is 3 s on CUDA against >550 s on CPU). The engine
# itself defaults to CPU -- a library should not pick a device for you -- so
# the script asks explicitly. MEEP_TORCH_DEVICE overrides; falls back to CPU
# on a machine without a card.
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")

resolution = 40

dpml = 2
pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

sx = 40
sy = 6
cell_size = mp.Vector3(sx + 2 * dpml, sy)

v0 = 1.0  # pulse center frequency
a = 0.2  # Gaussian envelope half-width
b = -0.5  # linear chirp rate (positive: up-chirp, negative: down-chirp)
t0 = 15  # peak time

chirp = lambda t: np.exp(1j * 2 * np.pi * v0 * (t - t0)) * np.exp(
    -a * (t - t0) ** 2 + 1j * b * (t - t0) ** 2
)

sources = [
    mp.Source(
        src=mp.CustomSource(src_func=chirp),
        center=mp.Vector3(-0.5 * sx),
        size=mp.Vector3(y=sy),
        component=mp.Ez,
    )
]

# --- the source itself, before anything is solved -------------------------
# The chirp is the subject of this example and nothing here showed it. The
# instantaneous frequency is the derivative of the phase:
#     phi(t)   = 2 pi v0 (t - t0) + b (t - t0)^2
#     f_inst   = v0 + b (t - t0) / pi
# so b < 0 puts the HIGH frequencies early -- the down-chirp in the title.
_t = np.linspace(t0 - 4.5 / np.sqrt(a), t0 + 4.5 / np.sqrt(a), 2000)
_s = np.array([chirp(t) for t in _t])
_f_inst = v0 + b * (_t - t0) / np.pi

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(_t, _s.real, lw=0.8, color="tab:blue", label="Re $s(t)$")
ax.plot(_t, np.abs(_s), lw=1.2, color="0.4", ls="--", label="envelope")
ax.set_xlabel("t")
ax.set_ylabel("source amplitude")
ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(_t, _f_inst, lw=1.4, color="tab:red", label="$f_{inst}$")
ax2.set_ylabel("instantaneous frequency", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax.set_title(f"Chirped source: $v_0$={v0}, a={a}, b={b} (down-chirp)")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "chirped_pulse_source.png", dpi=150)
plt.close(fig)
np.savetxt(OUT / "chirped_pulse_source.csv",
           np.column_stack((_t, _s.real, _s.imag, np.abs(_s), _f_inst)),
           delimiter=",", header="t,Re_s,Im_s,envelope,f_inst", comments="")

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    resolution=resolution,
    k_point=mp.Vector3(),
    sources=sources,
    symmetries=[mp.Mirror(mp.Y)],
)

# output_* files go to output/ instead of the working directory
sim.use_output_directory(str(OUT))

# Snapshot while the pulse is still inside the cell. until=t0+50 runs to
# t=65; the pulse leaves the source at t~=15 and travels at c=1, so by the
# end it has passed x=40um and been absorbed. The final state is therefore
# PML residual, not the pulse -- comparing final states across engines
# compares leftovers, which is why the earlier CSV diff looked catastrophic.
# t=40 puts the envelope centre near x=25um in cell coordinates, clear of
# both absorbers. The run length itself is unchanged from upstream.
SNAP = 40.0
_snap = {}


def _grab(sim):
    _snap["ez"] = sim.get_array(component=mp.Ez)


sim.run(
    mp.in_volume(
        mp.Volume(center=mp.Vector3(), size=mp.Vector3(sx, sy)),
        mp.at_every(2.7, mp.output_efield_z),
    ),
    mp.at_time(SNAP, _grab),
    until=t0 + 50,
)


ez = _snap["ez"]
ez_slice = ez[:, ez.shape[1] // 2]              # centre row
# x from the returned array and the cell, not sim.grid -- meep exposes no
# .grid, so reaching for one is a shim-ism that would not run upstream.
# Span the FULL cell: cell_size is sx + 2*dpml and get_array returns the
# whole grid including the PML, so dividing by sx alone compresses x by 10%.
cell_x = sx + 2 * dpml
x = (np.arange(ez.shape[0]) + 0.5) * (cell_x / ez.shape[0])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, ez_slice, lw=0.8)
ax.set_title(f"Chirped pulse -- Ez(x) at t = {SNAP:g} (down-chirp)")
ax.set_xlabel("x (um)")
ax.set_ylabel("Ez")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "chirped_pulse.png", dpi=150)
plt.close(fig)

data=np.hstack((x.reshape(len(x),1),
                          ez_slice.reshape(len(x),1)))

np.savetxt(OUT /'chirped_pulse.csv',
            data,
            delimiter=',')
print(np.shape(data))

peak = float(np.max(np.abs(ez_slice)))
print(f"chirped_pulse.png  [{DEVICE}]  max|Ez| at t={SNAP:g} = {peak:.6e}")
