# orbital_telemetry/scattering.py
import math
from typing import Dict, List, Tuple
import numpy as np

def legendre_polynomial(l: int, x: float) -> float:
    return float(np.polynomial.legendre.Legendre.basis(l)(x))

def differential_cross_section(
    phase_map: Dict[int, List[Tuple[float, float]]],
    energy: float,
    k: float,
    thetas: List[float]
) -> List[Tuple[float, float]]:
    """
    Compute dσ/dΩ(θ) via partial-wave sum:
    f(θ) = (1/(2ik)) Σ (2l+1)(e^{2iδ_l} - 1) P_l(cosθ)
    dσ/dΩ = |f(θ)|^2
    Note: k = sqrt(2μE)/ħ supplied externally.
    """
    # Interpolate phase at requested energy
    # Simple nearest neighbor; replace with spline if needed
    def phase_at(l: int) -> float:
        series = phase_map[l]
        best = min(series, key=lambda t: abs(t[0] - energy))
        return best[1]

    out = []
    for th in thetas:
        x = math.cos(th)
        re_sum = 0.0
        im_sum = 0.0
        for l in phase_map.keys():
            dl = phase_at(l)
            exp_term_re = math.cos(2.0 * dl)
            exp_term_im = math.sin(2.0 * dl)
            coeff = (2*l + 1) * legendre_polynomial(l, x)
            # e^{2iδ} - 1
            term_re = coeff * (exp_term_re - 1.0)
            term_im = coeff * (exp_term_im - 0.0)
            re_sum += term_re
            im_sum += term_im
        f_re = re_sum / (2.0 * k)  # dividing by i -> swap re/im; here we keep magnitude-only focus
        f_im = im_sum / (2.0 * k)
        dsdo = f_re**2 + f_im**2
        out.append((th, dsdo))
    return out
