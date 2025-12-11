# examples/run_demo.py
import json
import numpy as np
from orbital_telemetry.potentials import square_well, yukawa, combined
from orbital_telemetry.phase import phase_sweep, wigner_time_delay, HBAR
from orbital_telemetry.scattering import differential_cross_section
from orbital_telemetry.fit import find_resonances

# Define a demo potential (swap-in real model as needed)
V = combined(
    square_well(V0=5.0, R=1.2),
    yukawa(g=2.5, mu=1.0)
)

mu_red = 1.0
E_grid = np.linspace(0.1, 10.0, 200).tolist()
ells = [0, 1, 2]
thetas = np.linspace(0.0, np.pi, 181).tolist()

phase_map = phase_sweep(V, ells, mu_red, E_grid)
delay_map = {l: wigner_time_delay(series) for l, series in phase_map.items()}

# Find candidates per partial wave
candidates = {}
for l in ells:
    cand = find_resonances(phase_map[l], min_delay=delay_map[l], delay_threshold=0.5)
    candidates[l] = cand

# Pick an energy of interest
E_star = 3.0
k_star = np.sqrt(2.0 * mu_red * E_star) / HBAR
dsdo = differential_cross_section(phase_map, E_star, k_star, thetas)

print("Candidates (ell: [(E*, score), ...]):")
for l, c in candidates.items():
    print(l, c[:5])

# Save results
with open("demo_results.json", "w") as f:
    json.dump({
        "phase_map": {l: phase_map[l] for l in ells},
        "delay_map": {l: delay_map[l] for l in ells},
        "candidates": candidates,
        "E_star": E_star,
        "dsdo": dsdo
    }, f, indent=2)
