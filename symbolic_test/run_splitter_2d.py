# SPDX-License-Identifier: MIT
# Run the 2D Y-splitter scenario three ways and compare S-parameters:
#   1. meep_torch reference (step_tm)
#   2. pylops drop-in integrator (curl as a sparse LinearOperator)
#   3. devito drop-in integrator (curl as a symbolic derivative), if installed
# Same eps, CPML coeffs, ModeSourceTM, FluxMonitorTM, decompose for all three;
# only the Yee curl kernel differs. Output bar plot to symbolic_test/output/.
#
# Run:  uv run python -m symbolic_test.run_splitter_2d
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import splitter_setup as SS


def main() -> None:
    out = pathlib.Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)

    print("[1] meep_torch reference (step_tm) ...")
    ref = SS.meep_reference()[1]   # (t_up, t_dn, total, imbalance)
    ru, rd, rtot, rimb = ref
    print(f"    |S21|2={ru*100:.2f}% |S31|2={rd*100:.2f}% total={rtot*100:.2f}% "
          f"imb={rimb*100:.3f}%")

    labels, ups, dns = ["meep_torch"], [ru], [rd]

    print("[2] pylops drop-in ...")
    from .fdtd_pylops_2d import step_pylops
    m = SS.run_with_step(step_pylops)[1]   # (t_up, t_dn, total, imbalance)
    pu, pd, ptot, pimb = m
    print(f"    |S21|2={pu*100:.2f}% |S31|2={pd*100:.2f}% total={ptot*100:.2f}% "
          f"imb={pimb*100:.3f}%  (dTu={abs(pu-ru)*100:.3f}% dTd={abs(pd-rd)*100:.3f}%)")
    labels.append("pylops"); ups.append(pu); dns.append(pd)

    try:
        import devito  # noqa: F401
    except ImportError:
        print("[3] devito drop-in ... skipped (devito not installed)")
    else:
        print("[3] devito drop-in ...")
        from .fdtd_devito_2d import step_devito
        dm = SS.run_with_step(step_devito)[1]
        du, dd, dtot, dimb = dm
        print(f"    |S21|2={du*100:.2f}% |S31|2={dd*100:.2f}% total={dtot*100:.2f}% "
              f"imb={dimb*100:.3f}%")
        labels.append("devito"); ups.append(du); dns.append(dd)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], [u * 100 for u in ups], width=0.4, label="|S21|^2")
    ax.bar([i + 0.2 for i in x], [d * 100 for d in dns], width=0.4, label="|S31|^2")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("power [%]")
    ax.set_title(f"Y-splitter 2D drop-in integrators vs meep_torch "
                 f"(ref imb {rimb*100:.3f}%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "symbolic_splitter_2d.png", dpi=150)
    plt.close(fig)
    print(f"saved {out / 'symbolic_splitter_2d.png'}")


if __name__ == "__main__":
    main()