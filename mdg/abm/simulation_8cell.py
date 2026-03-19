"""
simulation_8cell.py — 8-cell C. elegans simulation using deformable ellipsoid model.

Extends Embryo (from simulation.py) with Embryo8Cell that runs the full
2 → 4 → 8 cell trajectory. Volumes for 8-cell daughters are extracted fresh
from Sample04_Volume.csv.

Run from project root:
    python mdg/abm/simulation_8cell.py
"""

import sys
import os
import math
import time
import itertools
import torch

# Resolve paths — data_loader lives in mdg/, physics & simulation in mdg/abm/
_ABM_DIR = os.path.dirname(os.path.abspath(__file__))
_MDG_DIR = os.path.dirname(_ABM_DIR)
_PROJ_DIR = os.path.dirname(_MDG_DIR)
for _d in [_ABM_DIR, _MDG_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from physics import (
    CellAgent, total_energy,
    SHELL_A, SHELL_B, SHELL_C, DEVICE, DTYPE,
    jkr_contact_area,
)
from simulation import Embryo, run_one_step, clamp_to_shell, DT, ETA
from data_loader import load_volumes, load_volumes_8cell, load_contact_areas


# ---------------------------------------------------------------------------
# Full lineage map — all 14 identities, mirrors physics.py _LINEAGE_MAP
# ---------------------------------------------------------------------------
LINEAGE_MAP = {
    "AB": "AB", "P1": "P",
    "ABa": "AB", "ABp": "AB", "EMS": "EMS", "P2": "P", "P": "P",
    "ABar": "AB", "ABal": "AB", "ABpr": "AB", "ABpl": "AB",
    "MS": "EMS", "E": "EMS",
    "C": "P", "P3": "P",
}

# Biologically expected contacts at 8-cell stage (from Sulston lineage)
EXPECTED_CONTACTS = {
    ("ABar", "ABpr"), ("ABal", "ABpl"),
    ("ABar", "MS"),   ("ABal", "MS"),
    ("ABpr", "C"),    ("ABpl", "C"),
    ("MS", "E"),      ("MS", "C"),
    ("E", "P3"),
}


# ---------------------------------------------------------------------------
# Embryo8Cell
# ---------------------------------------------------------------------------

class Embryo8Cell(Embryo):
    """
    Extends Embryo to run the full 2→8 cell simulation.
    run_8cell() inlines all phases — does not call super().run() to avoid
    double-recording and t-counter conflicts.
    """

    def __init__(self, volumes_4cell, volumes_8cell, params, perturbation=None):
        super().__init__(volumes_4cell, params, perturbation)
        self.volumes_8cell = volumes_8cell

        # 8-cell timing (all offsets from t=0)
        # Override parent's 4-cell equilibration: 300 steps, not 500
        self.T_EQUILIBRATE_4CELL = 300
        self.T_ABA_DIV  = self.T_P1_DIV + self.T_EQUILIBRATE_4CELL  # t=620
        self.T_ABP_DIV  = self.T_ABA_DIV + 40                        # t=660
        self.T_EMS_DIV  = self.T_ABA_DIV + 80                        # t=700
        self.T_P2_DIV   = self.T_ABA_DIV + 120                       # t=740
        self.T_EQUILIBRATE_8CELL = 500
        self.T_TOTAL = self.T_P2_DIV + self.T_EQUILIBRATE_8CELL      # t=1240

    def run_8cell(self, record_every=5, verbose=True):
        """Run complete 2→8 cell simulation and return trajectory."""

        if verbose:
            print(f"\nRunning 8-cell embryo simulation on {DEVICE}")
            print(f"Total steps: {self.T_TOTAL}")

        # --- Phase 1: 2-cell equilibration (200 steps) ---
        if verbose:
            print("\n[Phase 1] 2-cell equilibration...")
        for _ in range(self.T_EQUILIBRATE_2CELL):
            run_one_step(self.cells, self.params)
            self.t += 1
            if self.t % record_every == 0:
                self.record_frame()

        # --- Division 1: AB → ABa + ABp (Y axis) ---
        if verbose:
            print("\n[Division 1] AB → ABa + ABp (axis: Y)")
        self.divide("AB", ["ABa", "ABp"], [0.0, 1.0, 0.0], [0.5, 0.5])
        self.record_frame()
        self._print_cells()

        # --- Phase 2: 3-cell equilibration (120 steps) ---
        if verbose:
            print("\n[Phase 2] 3-cell equilibration...")
        for _ in range(self.T_P1_DIV - self.T_AB_DIV):
            run_one_step(self.cells, self.params)
            self.t += 1
            if self.t % record_every == 0:
                self.record_frame()

        # --- Division 2: P1 → P2 + EMS (X axis) ---
        if verbose:
            print("\n[Division 2] P1 → P2 + EMS (axis: X)")
        # P2 at +X (posterior), EMS at -X (anterior)
        self.divide("P1", ["P2", "EMS"], [1.0, 0.0, 0.0], [0.45, 0.55])
        self.record_frame()
        self._print_cells()

        # --- Phase 3: 4-cell equilibration (300 steps) ---
        if verbose:
            print("\n[Phase 3] 4-cell equilibration (300 steps)...")
        for _ in range(self.T_EQUILIBRATE_4CELL):
            run_one_step(self.cells, self.params)
            self.t += 1
            if self.t % record_every == 0:
                self.record_frame()

        if verbose:
            print("\n[4-cell checkpoint]")
            self._print_final_state()

        # Save a snapshot of the 4-cell trajectory endpoint for reporting
        self._trajectory_4cell_end = len(self.trajectory) - 1

        # --- Division 3: ABa → ABar + ABal (Z axis) ---
        if verbose:
            print("\n[Division 3] ABa → ABar + ABal (axis: Z)")
        self.divide_8cell("ABa", ["ABar", "ABal"], [0, 0, 1], [0.5, 0.5])
        self.record_frame()
        self._print_cells()

        # 40-step equilibration
        for _ in range(40):
            run_one_step(self.cells, self.params)
            self.t += 1
            if self.t % record_every == 0:
                self.record_frame()

        # --- Division 4: ABp → ABpr + ABpl (Z axis) ---
        if verbose:
            print("\n[Division 4] ABp → ABpr + ABpl (axis: Z)")
        self.divide_8cell("ABp", ["ABpr", "ABpl"], [0, 0, 1], [0.5, 0.5])
        self.record_frame()
        self._print_cells()

        # 40-step equilibration
        for _ in range(40):
            run_one_step(self.cells, self.params)
            self.t += 1
            if self.t % record_every == 0:
                self.record_frame()

        # --- Division 5: EMS → MS + E (X axis) ---
        if verbose:
            print("\n[Division 5] EMS → MS + E (axis: X)")
        # MS at +X, E at -X
        self.divide_8cell("EMS", ["MS", "E"], [1, 0, 0], [0.55, 0.45])
        self.record_frame()
        self._print_cells()

        # 40-step equilibration
        for _ in range(40):
            run_one_step(self.cells, self.params)
            self.t += 1
            if self.t % record_every == 0:
                self.record_frame()

        # --- Division 6: P2 → C + P3 (X axis) ---
        if verbose:
            print("\n[Division 6] P2 → C + P3 (axis: X)")
        # C at +X, P3 at -X
        self.divide_8cell("P2", ["C", "P3"], [1, 0, 0], [0.5, 0.5])
        self.record_frame()
        self._print_cells()

        # --- Phase 5: 8-cell equilibration (500 steps) ---
        if verbose:
            print("\n[Phase 5] 8-cell equilibration (500 steps)...")
        for step in range(self.T_EQUILIBRATE_8CELL):
            run_one_step(self.cells, self.params)
            self.t += 1
            if self.t % record_every == 0:
                self.record_frame()
            # NaN guard
            for c in self.cells:
                if not torch.all(torch.isfinite(c.position)):
                    raise RuntimeError(
                        f"NaN in {c.identity} position at step {self.t}"
                    )

        if verbose:
            print(f"\nSimulation complete. {len(self.trajectory)} frames recorded.")
            self._print_final_state()

        return self.trajectory

    def divide_8cell(self, mother_identity, daughter_names, axis_vector, volume_fractions):
        """
        Division using 8-cell volumes from dataset.
        Falls back to mother.V0 * fraction if a daughter is missing from dataset.
        Initialises position, axes, and quaternion for every daughter.
        """
        mother = next(
            (c for c in self.cells if c.identity == mother_identity), None
        )
        if mother is None:
            raise ValueError(f"Cell {mother_identity} not found. Current cells: "
                             f"{[c.identity for c in self.cells]}")

        mother_pos = mother.position.detach().clone()
        daughters  = []

        for idx, (name, frac) in enumerate(zip(daughter_names, volume_fractions)):
            # Volume from dataset or fallback
            V_d = self.volumes_8cell.get(name)
            if V_d is None:
                V_d = mother.V0 * frac
                print(f"  [WARNING] {name} not in 8-cell dataset — using "
                      f"mother.V0 * {frac:.2f} = {V_d:.2f} μm³")
            else:
                print(f"  {name}: V0 = {V_d:.2f} μm³ (from dataset)")

            d   = CellAgent(name, V_d)
            ax  = torch.tensor(axis_vector, dtype=DTYPE, device=DEVICE)
            sign = 1.0 if idx == 0 else -1.0
            pos  = mother_pos + sign * d.R * ax

            if self.perturbation is not None and self.perturbation > 0:
                noise = torch.randn(3, dtype=DTYPE, device=DEVICE) * self.perturbation
                pos   = pos + noise

            pos = clamp_to_shell(pos, d.R_c)
            if not torch.all(torch.isfinite(pos)):
                print(f"  [WARNING] {name} position non-finite after clamp, resetting to origin")
                pos = torch.zeros(3, dtype=DTYPE, device=DEVICE)

            d.position  = pos.clone().detach().requires_grad_(True)
            d.axes      = torch.tensor([d.R, d.R, d.R],
                                        dtype=DTYPE, device=DEVICE, requires_grad=True)
            d.quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0],
                                         dtype=DTYPE, device=DEVICE, requires_grad=True)
            daughters.append(d)

        self.cells = [c for c in self.cells if c.identity != mother_identity]
        self.cells.extend(daughters)

        self.division_log.append({
            "t":        self.t,
            "mother":   mother_identity,
            "daughters": daughter_names,
            "axis":     axis_vector,
        })

    def _print_cells(self):
        print(f"  Current cells (t={self.t}): "
              f"{[c.identity for c in self.cells]}")

    def compute_full_topology(self):
        """Return dict of all 28 pairwise contact areas at current state."""
        contacts = {}
        w = self.params["w"]
        ids = [c.identity for c in self.cells]
        for i, j in itertools.combinations(range(len(self.cells)), 2):
            ci, cj = self.cells[i], self.cells[j]
            A = jkr_contact_area(ci, cj, w, self.params)
            key = tuple(sorted([ci.identity, cj.identity]))
            contacts[key] = A.item()
        return contacts

    def print_topology_table(self, contacts):
        """Print the full 28-pair contact table."""
        cells8 = ["ABar", "ABal", "ABpr", "ABpl", "MS", "E", "C", "P3"]
        all_pairs = list(itertools.combinations(cells8, 2))

        print("\n" + "=" * 65)
        print("FULL 28-PAIR CONTACT TABLE")
        print("=" * 65)
        print(f"{'Pair':<20} {'Area (μm²)':>12}  {'Expected?':>10}  {'Status':>8}")
        print("-" * 65)

        n_nonzero = 0
        expected_found = 0
        for a, b in all_pairs:
            key = tuple(sorted([a, b]))
            area = contacts.get(key, 0.0)
            exp  = "YES" if key in {tuple(sorted(p)) for p in EXPECTED_CONTACTS} else ""
            stat = "PRESENT" if area > 5.0 else "absent"
            if area > 5.0:
                n_nonzero += 1
                if exp == "YES":
                    expected_found += 1
            print(f"  {a}-{b:<15} {area:>12.2f}  {exp:>10}  {stat:>8}")

        print("-" * 65)
        print(f"\n  N non-zero (> 5 μm²): {n_nonzero}/28")
        print(f"  Spherical baseline:   28/28 (fully degenerate)")
        print(f"  Deformable result:    {n_nonzero}/28")
        print(f"  Expected contacts present: {expected_found}/9")

        if n_nonzero < 28:
            print("\n  [RESULT] Deformable model IS restricting contacts.")
        else:
            print("\n  [RESULT] All 28 pairs non-zero — topology still degenerate.")

        print("=" * 65)
        return n_nonzero, expected_found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    # --- Load volumes ---
    print("=" * 60)
    print("LOADING VOLUMES")
    print("=" * 60)

    volumes_4cell = load_volumes()
    print("\n4-cell volumes (from dataset):")
    for name, v in volumes_4cell.items():
        print(f"  {name}: {v:.2f} μm³")

    volumes_8cell = load_volumes_8cell()
    print("\n8-cell volumes (from dataset):")
    fallbacks = []
    for name, v in volumes_8cell.items():
        if v is None:
            print(f"  {name}: MISSING — will use mother.V0/2 fallback")
            fallbacks.append(name)
        else:
            print(f"  {name}: {v:.2f} μm³")

    if fallbacks:
        print(f"\n  [WARNING] Fallback used for: {fallbacks}")

    # --- Best calibrated params from Prompt 2 ---
    best_params = {
        "gamma_AB":  0.8455,
        "gamma_EMS": 0.7610,
        "gamma_P":   0.6826,
        "w":         0.8523,
        "alpha":     0.3482,
    }
    measured_areas_4cell = load_contact_areas()

    # --- Run simulation ---
    print("\n" + "=" * 60)
    print("RUNNING 8-CELL SIMULATION")
    print("=" * 60)

    embryo = Embryo8Cell(
        volumes_4cell,
        volumes_8cell,
        best_params,
        perturbation=None,
    )
    trajectory = embryo.run_8cell(record_every=5, verbose=True)

    # --- Topology validation ---
    print("\n" + "=" * 60)
    print("TOPOLOGY VALIDATION")
    print("=" * 60)
    contacts = embryo.compute_full_topology()
    n_nonzero, expected_found = embryo.print_topology_table(contacts)

    # --- Cell shapes at 8-cell equilibrium ---
    print("\n8-CELL SHAPE SUMMARY")
    print("=" * 60)
    for c in embryo.cells:
        if c.axes is not None:
            ax = c.axes.detach().cpu().numpy()
            ratio = ax.max() / ax.min()
            print(f"  {c.identity}: a={ax[0]:.3f} b={ax[1]:.3f} c={ax[2]:.3f}  "
                  f"max/min={ratio:.4f}")

    # --- Save results ---
    print("\nSaving simulation_results_8cell.pt ...")
    out_path = os.path.join(_PROJ_DIR, "results", "simulation_results_8cell.pt")

    # Split trajectory at 4-cell end
    t4_end = embryo._trajectory_4cell_end
    trajectory_4cell = trajectory[:t4_end + 1]
    trajectory_8cell = trajectory[t4_end + 1:]

    torch.save({
        "trajectory_4cell":  trajectory_4cell,
        "trajectory_8cell":  trajectory_8cell,
        "best_params":       best_params,
        "volumes_4cell":     volumes_4cell,
        "volumes_8cell":     volumes_8cell,
        "measured_areas_4cell": {str(k): v for k, v in measured_areas_4cell.items()},
        "division_log_full": embryo.division_log,
        "topology_8cell": {
            "contact_table":            {str(k): v for k, v in contacts.items()},
            "n_nonzero":                n_nonzero,
            "expected_contacts_present": expected_found,
        },
    }, out_path)
    print(f"Saved to {out_path}")
    print(f"\nTotal runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
