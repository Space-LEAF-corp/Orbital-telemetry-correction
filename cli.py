# orbital_telemetry/cli.py
import json
import math
import argparse
from .potentials import square_well, yukawa, combined
from .phase import phase_sweep, wigner_time_delay, HBAR
from .scattering import differential_cross_section

def build_potential(cfg):
    parts = []
    for p in cfg.get("potentials", []):
        if p["type"] == "square_well":
            parts.append(square_well(p["V0"], p["R"]))
        elif p["type"] == "yukawa":
            parts.append(yukawa(p["g"], p["mu"]))
        else:
            raise ValueError(f"Unknown potential type: {p['type']}")
    return combined(*parts) if parts else (lambda r: 0.0)

def main():
    parser = argparse.ArgumentParser(description="Orbital Telemetry Correction CLI")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--out", default="results.json", help="Output JSON path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    mu_red = cfg["mu_red"]
    E_grid = cfg["energies"]
    ells = cfg["ells"]
    thetas = cfg["thetas"]

    V = build_potential(cfg)

    phase_map = phase_sweep(V, ells, mu_red, E_grid)
    delay_map = {l: wigner_time_delay(series) for l, series in phase_map.items()}

    # Example: compute differential cross section at a target energy
    E_star = cfg.get("E_star", E_grid[len(E_grid)//2])
    k_star = math.sqrt(2.0 * mu_red * E_star) / HBAR
    dsdo = differential_cross_section(phase_map, E_star, k_star, thetas)

    out = {
        "phase_map": {l: list(map(list, series)) for l, series in phase_map.items()},
        "delay_map": {l: list(map(list, series)) for l, series in delay_map.items()},
        "dsdo_at_E_star": list(map(list, dsdo)),
        "E_star": E_star
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
