# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Process-variation study (FDFD Y splitter)
#
# Process-variation re-optimization case study on the FDFD Y splitter
# (c_y_splitter_fdfd.py design, DEFAULT mmi1x2_with_sbend @674 nm).
#
# States:
# ```
#   1. baseline  : saved optimized rho evaluated at nominal LPCVD SiN index n1
#   2. degraded  : SAME frozen rho evaluated with core index n2 = n1 - 0.04
#                  (PECVD-like deficit); also n1 + 0.04 as cheap cross-check
#   3. recovered : Adam re-optimization at n2 seeded from the baseline rho
# ```
#
# All numbers are actual FDFD solves (pardiso backend, same normalization
# scheme as c_y_splitter_fdfd.py: mode amplitude ratio vs a straight-guide
# reference solve rebuilt per index so the calibration tracks the material).
#
# Run: `uv run python examples_2/process_variation/process_variation_study.py`

# %%
import math
import pathlib
import time

import gdsfactory as gf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

EX2 = pathlib.Path(__file__).parent.parent
OUT = pathlib.Path(__file__).parent
from meep_torch.plugins.gdsfactory import component_to_eps

import functools
from meep_torch.adjoint import DesignRegion, MaterialGrid
from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.fdfd import make_mode_source, mode_amplitude
from meep_torch.fdfd import solve_fdfd_tm as _solve
from meep_torch.modes.fd1d import solve_1d
from meep_torch.ports.port import Port

solve_fdfd_tm = functools.partial(_solve, backend="pardiso")
torch.set_num_threads(8)

# %% [markdown]
# ## Wavelength, indices, and index deltas
# Nominal LPCVD Si3N4 at 674 nm plus the index perturbations under study:
# -0.04 (PECVD-like), +0.04 (cross-check), and -0.10 (severe).

# %%
WL = 0.674
FCEN = 1 / WL
N_SIN1 = Si3N4.n(FCEN).real          # nominal LPCVD stoichiometric Si3N4
N_CLAD = SiO2.n(FCEN).real
RES = 30
dx = 1.0 / RES
NPML = RES
DELTAS = {"n1": 0.0, "n2": -0.04, "n1+0.04": +0.04, "n3": -0.10}

# %% [markdown]
# ## Device: default MMI 1x2 with S-bends
# The same gdsfactory component the baseline design was optimized on, with
# 4 um straights appended at each port.

# %%
gf.gpdk.PDK.activate()
mmi = gf.components.mmi1x2_with_sbend()
xs = gf.get_cross_section("strip")
comp = gf.Component("y_splitter_default_sbend")
ref = comp.add_ref(mmi)
for pname in ("o1", "o2", "o3"):
    st = comp.add_ref(gf.components.straight(length=4.0, cross_section=xs))
    st.connect("o1", ref.ports[pname])
    comp.add_port(pname, port=st.ports["o2"])
comp.flatten()


# %% [markdown]
# ## Per-index evaluation context
# Each Setup rebuilds the slab effective index, the rasterized eps, the ports,
# the design region, AND the straight-guide normalization reference at its own
# core index -- so all transmissions are like-for-like mode-power ratios.

# %%
def neff_of(n_sin):
    """150 nm SiN slab lateral effective index (same recipe as c_y_splitter)."""
    eps_line = np.full(400, N_CLAD ** 2)
    eps_line[200 - 7: 200 + 8] = n_sin ** 2
    return float(solve_1d(eps_line, 0.01, FCEN, "te", k=1).neff[0])


class Setup:
    """Full evaluation context at one core index."""

    def __init__(self, n_sin):
        self.n_sin = n_sin
        self.neff = neff_of(n_sin)
        grid_c, eps_c, ports = component_to_eps(comp, res=RES, pad=3.0,
                                                n_core=self.neff, n_clad=N_CLAD)
        self.nx, self.ny = grid_c.nx, grid_c.ny
        self.eps_base = eps_c.to(torch.float64)
        o1x, o1y, _ = ports["o1"]
        o2x, o2y, _ = ports["o2"]
        o3x, o3y, _ = ports["o3"]
        ix0 = o1x + int(4.0 * RES)
        ix1 = o2x - int(4.0 * RES)
        iy0 = o3y - int(1.2 * RES)
        iy1 = o2y + int(1.2 * RES)
        NRX = int((ix1 - ix0) * dx / 0.08)
        NRY = int((iy1 - iy0) * dx / 0.08)
        mg = MaterialGrid((NRX, NRY), eps_min=N_CLAD ** 2, eps_max=self.neff ** 2)
        self.region = DesignRegion(mg, ix0=ix0, ix1=ix1, iy0=iy0, iy1=iy1)
        SPAN = int(1.2 * RES)
        self.p_in = Port(axis="x", index=max(o1x + int(2.0 * RES), NPML + 4),
                         lo=o1y - SPAN, hi=o1y + SPAN, direction=+1)
        self.p_up = Port(axis="x", index=min(o2x - int(2.0 * RES), self.nx - NPML - 4),
                         lo=o2y - SPAN, hi=o2y + SPAN, direction=-1)
        self.p_dn = Port(axis="x", index=min(o3x - int(2.0 * RES), self.nx - NPML - 4),
                         lo=o3y - SPAN, hi=o3y + SPAN, direction=-1)
        self.jz_in = make_mode_source(self.eps_base, self.p_in, FCEN, dx)
        # straight-guide reference at THIS index -- calibration tracks material
        ref_eps = self.eps_base[self.p_in.index, :].unsqueeze(0) \
            .expand(self.nx, self.ny).clone()
        p_ref = Port(axis="x", index=self.p_up.index,
                     lo=o1y - SPAN, hi=o1y + SPAN, direction=-1)
        Ez_ref = solve_fdfd_tm(ref_eps, dx, FCEN, self.jz_in, NPML)
        self.a_ref = mode_amplitude(Ez_ref, ref_eps, p_ref, FCEN, dx).detach()

    def TuTd(self, eps):
        Ez = solve_fdfd_tm(eps, dx, FCEN, self.jz_in, NPML)
        a_up = mode_amplitude(Ez, eps, self.p_up, FCEN, dx)
        a_dn = mode_amplitude(Ez, eps, self.p_dn, FCEN, dx)
        ru, rd = a_up / self.a_ref, a_dn / self.a_ref
        return (ru.conj() * ru).real, (rd.conj() * rd).real, Ez

    def eval_rho(self, rho, beta=64.0):
        with torch.no_grad():
            r = tanh_projection(conic_filter(rho, 0.12, dx), beta)
            eps = self.region.paint(self.eps_base, r)
            Tu, Td, Ez = self.TuTd(eps)
        return float(Tu), float(Td), eps, Ez

    def objective(self, rho, beta):
        r = tanh_projection(conic_filter(rho, 0.12, dx), beta)
        eps = self.region.paint(self.eps_base, r)
        Tu, Td, _ = self.TuTd(eps)
        return 2 * torch.sqrt(Tu * Td + 1e-12)


def report(tag, n_sin, neff, Tu, Td):
    tot = Tu + Td
    il = -10 * math.log10(max(tot, 1e-12))
    imb = abs(Tu - Td) / max(tot, 1e-12)
    print(f"STATE {tag}: n_SiN={n_sin:.4f} neff={neff:.4f} "
          f"Tu={Tu*100:.2f}% Td={Td*100:.2f}% total={tot*100:.2f}% "
          f"IL={il:.2f}dB imbalance={imb*100:.2f}%", flush=True)
    return dict(tag=tag, n=n_sin, neff=neff, Tu=Tu, Td=Td, tot=tot, il=il, imb=imb)


# %% [markdown]
# ## States 1-2: baseline and frozen designs at shifted indices
# Load the baseline rho, build a Setup per index, then evaluate the FROZEN
# design at n1, n2, n1+0.04, and n3 -- the degradation the fab shift causes.

# %%
rho0 = torch.from_numpy(np.load(EX2 / "y_splitter" / "output" / "c_y_splitter_fdfd_rho.npy")).to(torch.float64)
results = {}

t0 = time.time()
setups = {}
for tag, dn in DELTAS.items():
    setups[tag] = Setup(N_SIN1 + dn)
print(f"setups built in {time.time()-t0:.0f}s "
      f"(grid {setups['n1'].nx}x{setups['n1'].ny})", flush=True)

# 1. baseline at n1 + 2. frozen design at shifted indices
Tu, Td, _, _ = setups["n1"].eval_rho(rho0)
results["baseline@n1"] = report("baseline@n1", N_SIN1, setups["n1"].neff, Tu, Td)
Tu, Td, eps_deg, Ez_deg = setups["n2"].eval_rho(rho0)
results["frozen@n2"] = report("frozen@n2", N_SIN1 - 0.04, setups["n2"].neff, Tu, Td)
Tu, Td, _, _ = setups["n1+0.04"].eval_rho(rho0)
results["frozen@n1+0.04"] = report("frozen@n1+0.04", N_SIN1 + 0.04,
                                   setups["n1+0.04"].neff, Tu, Td)

Tu, Td, eps_deg3, Ez_deg3 = setups["n3"].eval_rho(rho0)
results["frozen@n3"] = report("frozen@n3", N_SIN1 - 0.10, setups["n3"].neff, Tu, Td)

# %% [markdown]
# ## State 3: re-optimization at the shifted indices
# Adam re-optimization at n2 and n3, seeded from the baseline rho, with
# early-stop on plateau; the re-optimized design is also cross-checked back
# at n1.

# %%
# 3. re-optimize at shifted index, seeded from baseline rho (design already
# near-binary: short beta=32 loosening then beta=64)
N_IT = 64  # cap; early-stop on plateau (<0.1% gain over last 8 its)


def reopt(s, tag):
    rho = rho0.clone().requires_grad_(True)
    adam = torch.optim.Adam([rho], lr=0.03)
    post64 = []
    best = (-1.0, None, -1)
    t_opt = time.time()
    for it in range(N_IT):
        BETA = 32 if it < 8 else 64
        adam.zero_grad()
        J = s.objective(rho.clamp(0.0, 1.0), BETA)
        (-J).backward()
        adam.step()
        with torch.no_grad():
            Jp = float(s.objective(rho.clamp(0.0, 1.0), 64.0))
        post64.append(Jp)
        if Jp > best[0]:
            best = (Jp, rho.detach().clamp(0, 1).clone(), it)
        print(f"reopt[{tag}] it={it} beta={BETA}: J={float(J.detach())*100:.1f}% "
              f"post64={Jp*100:.1f}%", flush=True)
        if it >= 16 and max(post64[-8:]) - max(post64[:-8]) < 1e-3:
            print(f"reopt[{tag}] plateau at it={it}", flush=True)
            break
    return best[1], post64, best[2], time.time() - t_opt


s2, s3 = setups["n2"], setups["n3"]
rho_re, post64_2, ibest2, wall2 = reopt(s2, "n2")
Tu, Td, eps_re, Ez_re = s2.eval_rho(rho_re)
results["reopt@n2"] = report("reopt@n2", N_SIN1 - 0.04, s2.neff, Tu, Td)
# sanity: re-optimized design back at n1
Tu, Td, _, _ = setups["n1"].eval_rho(rho_re)
results["reopt@n1"] = report("reopt-design@n1(xcheck)", N_SIN1,
                             setups["n1"].neff, Tu, Td)
np.save(OUT / "c_y_splitter_fdfd_rho_reopt_n2.npy", rho_re.cpu().numpy())
print(f"re-opt@n2 wall-clock {wall2/60:.1f} min, {len(post64_2)} iterations, "
      f"best it{ibest2}", flush=True)

rho_re3, post64_3, ibest3, wall3 = reopt(s3, "n3")
Tu, Td, eps_re3, Ez_re3 = s3.eval_rho(rho_re3)
results["reopt@n3"] = report("reopt@n3", N_SIN1 - 0.10, s3.neff, Tu, Td)
np.save(OUT / "c_y_splitter_fdfd_rho_reopt_n3.npy", rho_re3.cpu().numpy())
print(f"re-opt@n3 wall-clock {wall3/60:.1f} min, {len(post64_3)} iterations, "
      f"best it{ibest3}", flush=True)

# %% [markdown]
# ## Figure
# Bar chart of total transmission per state, the re-optimization curves, and
# |Ez| for the frozen vs re-optimized design at the severe -0.10 shift.

# %%
# figure
nx, ny = s2.nx, s2.ny
fig, axs = plt.subplots(2, 2, figsize=(13, 7))
ax1, ax2, ax3, ax4 = axs.flat
keys = ["baseline@n1", "frozen@n2", "reopt@n2", "frozen@n3", "reopt@n3"]
labels = [f"baseline\nΔn=0",
          "frozen\nΔn=-0.04", "re-opt\nΔn=-0.04",
          "frozen\nΔn=-0.10", "re-opt\nΔn=-0.10"]
vals = [results[k]["tot"] * 100 for k in keys]
bars = ax1.bar(labels, vals, color=["#2a7", "#c66", "#27c", "#c33", "#27c"])
for b, k in zip(bars, keys):
    ax1.text(b.get_x() + b.get_width() / 2, 50,
             f"{results[k]['tot']*100:.1f}%\nIL {results[k]['il']:.2f} dB",
             ha="center", fontsize=8)
ax1.set_ylabel("total transmission Tu+Td [%]")
ax1.set_ylim(0, 105)
ax1.set_title(f"SiN core index variation @{WL*1e3:.0f} nm")
ax2.plot(np.array(post64_2) * 100, "bo-", label="re-opt @Δn=-0.04")
ax2.plot(np.array(post64_3) * 100, "ms-", label="re-opt @Δn=-0.10")
ax2.axhline(results["baseline@n1"]["tot"] * 100, color="#2a7", ls="--",
            lw=1, label="baseline@n1 total")
ax2.axhline(results["frozen@n3"]["tot"] * 100, color="#c33", ls="--",
            lw=1, label="frozen@Δn=-0.10 total")
ax2.set_xlabel("re-opt iteration"); ax2.set_ylabel("2·sqrt(Tu·Td) [%]")
ax2.set_title("re-optimization seeded from baseline rho")
ax2.legend(fontsize=7)
ax3.imshow(np.abs(Ez_deg3.cpu().numpy()).T, origin="lower", cmap="inferno",
           extent=[0, nx * dx, 0, ny * dx], aspect="auto")
ax3.set_title(f"|Ez| frozen design @Δn=-0.10 ({results['frozen@n3']['tot']*100:.1f}%)")
ax4.imshow(np.abs(Ez_re3.cpu().numpy()).T, origin="lower", cmap="inferno",
           extent=[0, nx * dx, 0, ny * dx], aspect="auto")
ax4.set_title(f"|Ez| re-optimized @Δn=-0.10 ({results['reopt@n3']['tot']*100:.1f}%)")
for ax in (ax3, ax4):
    ax.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "process_variation_study.png", dpi=500)
plt.close(fig)

# %% [markdown]
# ## Markdown report
# A results table plus the run metadata, written next to the figure.

# %%
# markdown table
rows = ["| state | n_SiN | n_eff | Tu [%] | Td [%] | total [%] | IL [dB] | imbalance [%] |",
        "|---|---|---|---|---|---|---|---|"]
for k in ["baseline@n1", "frozen@n2", "reopt@n2", "frozen@n3", "reopt@n3",
          "frozen@n1+0.04", "reopt@n1"]:
    r = results[k]
    rows.append(f"| {r['tag']} | {r['n']:.4f} | {r['neff']:.4f} | "
                f"{r['Tu']*100:.2f} | {r['Td']*100:.2f} | {r['tot']*100:.2f} | "
                f"{r['il']:.2f} | {r['imb']*100:.2f} |")
md = (
    "# Process-variation re-optimization study\n\n"
    "Device: inverse-designed MMI 1x2 splitter (DEFAULT gdsfactory "
    "mmi1x2_with_sbend seed), SiN 150 nm / SiO2, 674 nm, 2D lateral "
    "effective index, exact-adjoint FDFD (pardiso). Baseline design: "
    "`examples_2/y_splitter/output/c_y_splitter_fdfd_rho.npy` "
    "(24-iteration FDFD optimization).\n\n"
    "Perturbation: core (film) index n_SiN shifted by Δn = -0.04 "
    "(PECVD-like deficit vs stoichiometric LPCVD) and Δn = -0.10; the slab "
    "effective index and the straight-guide normalization reference are "
    "rebuilt at each index, so all transmissions are like-for-like "
    "mode-power ratios.\n\n"
    + "\n".join(rows) + "\n\n"
    f"- Re-optimization: Adam (lr=0.03, beta 32→64), seeded from baseline "
    f"rho, early-stop on plateau; {len(post64_2)} iterations / best "
    f"it{ibest2} / {wall2/60:.1f} min (Δn=-0.04), {len(post64_3)} iterations "
    f"/ best it{ibest3} / {wall3/60:.1f} min (Δn=-0.10).\n"
    f"- Re-optimized designs saved: "
    f"`c_y_splitter_fdfd_rho_reopt_n2.npy` (Δn=-0.04), "
    f"`c_y_splitter_fdfd_rho_reopt_n3.npy` (Δn=-0.10).\n"
    "- All values from actual FDFD solves; nothing extrapolated.\n"
)
(OUT / "process_variation_study.md").write_text(md)
print("process_variation_study.png / .md written", flush=True)
