from typing import NamedTuple, Sequence


class LorentzPole(NamedTuple):
    """Meep-convention Lorentzian: chi(f) = sigma * fn^2 / (fn^2 - f^2 - i*f*gamma)."""

    fn: float
    gamma: float
    sigma: float


def analytic_eps(eps_inf: float, poles: Sequence[LorentzPole], f: float) -> complex:
    eps = complex(eps_inf)
    for p in poles:
        eps += p.sigma * p.fn**2 / (p.fn**2 - f**2 - 1j * f * p.gamma)
    return eps
