# orbital_telemetry/phase.py
import math
from typing import Callable, List, Tuple

HBAR = 1.0  # natural units by default
def effective_potential(V: Callable[[float], float], ell: int, mu_red: float, r: float) -> float:
    return V(r) + (ell*(ell+1) * HBAR**2) / (2.0 * mu_red * r**2)

def wkb_phase_shift(
    V: Callable[[float], float],
    ell: int,
    mu_red: float,
    E: float,
    r_min: float = 1e-3,
    r_max: float = 50.0,
    steps: int = 10000
) -> float:
    """
    Crude WKB-like estimate of the phase shift for partial wave ell at energy E.
    Integrates k(r) difference relative to a free reference.
    """
    dr = (r_max - r_min) / steps
    delta = 0.0
    for i in range(steps):
        r = r_min + i * dr
        Veff = effective_potential(V, ell, mu_red, r)
        k = math.sqrt(max(0.0, 2.0 * mu_red * max(E - Veff, 0.0))) / HBAR
        k0 = math.sqrt(max(0.0, 2.0 * mu_red * E)) / HBAR
        delta += (k - k0) * dr
    # Langer-like correction term (heuristic)
    return delta - (ell * math.pi / 2.0)

def phase_sweep(
    V: Callable[[float], float],
    ell_list: List[int],
    mu_red: float,
    energies: List[float]
) -> dict:
    """
    Compute phase shifts for multiple ell over an energy grid.
    Returns {ell: [(E, delta_ell(E)), ...], ...}
    """
    out = {}
    for ell in ell_list:
        series = []
        for E in energies:
            series.append((E, wkb_phase_shift(V, ell, mu_red, E)))
        out[ell] = series
    return out

def wigner_time_delay(deltas: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Given [(E, delta(E))], return [(E, tau(E))] with tau = 2*hbar*d(delta)/dE.
    Central differences for derivative.
    """
    if len(deltas) < 3:
        return [(E, 0.0) for (E, _) in deltas]
    result = []
    for i in range(len(deltas)):
        E, d = deltas[i]
        if 0 < i < len(deltas) - 1:
            E_prev, d_prev = deltas[i-1]
            E_next, d_next = deltas[i+1]
            dEdelta = (d_next - d_prev) / (E_next - E_prev)
        else:
            dEdelta = 0.0
        tau = 2.0 * HBAR * dEdelta
        result.append((E, tau))
    return result
