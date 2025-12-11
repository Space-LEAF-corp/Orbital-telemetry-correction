# orbital_telemetry/potentials.py
import math
from typing import Callable

def square_well(V0: float, R: float) -> Callable[[float], float]:
    """
    Simple finite square well: V(r) = -V0 for r < R, else 0.
    """
    def V(r: float) -> float:
        return -V0 if r < R else 0.0
    return V

def yukawa(g: float, mu: float) -> Callable[[float], float]:
    """
    Yukawa-like potential: V(r) = -g * exp(-mu*r) / r, regularized near r=0.
    """
    def V(r: float) -> float:
        if r < 1e-6:
            r = 1e-6
        return -g * math.exp(-mu * r) / r
    return V

def combined(*Vs: Callable[[float], float]) -> Callable[[float], float]:
    """
    Linear combination: V_total(r) = sum_i V_i(r).
    """
    def V(r: float) -> float:
        return sum(v(r) for v in Vs)
    return V
