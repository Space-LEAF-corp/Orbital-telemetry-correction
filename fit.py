# orbital_telemetry/fit.py
import math
from typing import List, Tuple, Optional

def breit_wigner_phase(E: float, E0: float, Gamma: float) -> float:
    return math.atan2(Gamma / 2.0, E0 - E)

def find_resonances(
    deltas: List[Tuple[float, float]],
    min_jump: float = 0.7,   # ~π/2 crossing indicator threshold (radians)
    min_delay: Optional[List[Tuple[float, float]]] = None,
    delay_threshold: float = 1.0
) -> List[Tuple[float, float]]:
    """
    Flag candidate resonance energies where phase changes rapidly.
    Returns list of (E*, score) where score combines phase jump and delay (if provided).
    """
    flags = []
    for i in range(1, len(deltas)):
        E_prev, d_prev = deltas[i-1]
        E, d = deltas[i]
        jump = abs(d - d_prev)
        score = jump
        if min_delay is not None:
            # align by energy index; assumes same grid
            _, tau = min_delay[i]
            score += max(0.0, tau - delay_threshold)
        if jump >= min_jump:
            flags.append((E, score))
    return flags
