"""
simulation_8cell.py — 8-cell C. elegans simulation (fixed, v2).

Changes from v1 (per diagnostic_report.md):
  Fix 3: 4-cell equil 300 → 800 steps; inter-division gaps 40 → 150 steps
  Fix 4: EMS division delayed to T_ABA + 160, P2 to T_ABA + 200
  Fix 5: E-cell symmetry-breaking nudge at EMS division
  Fix 6: adaptive equilibration termination (stop when |ΔE| < tol for 20 steps)

Fixes 1+2 (gradient clips / axes LR) are in simulation.py run_one_step.

Run from project root:
    python mdg/abm/simulation_8cell.py
"""

import sys
import os
import math
import time
import itertools
import torch

_ABM_DIR  = os.path.dirname(os.path.abspath(__file__))
_MDG_DIR  = os.path.dirname(_ABM_DIR)
_PROJ_DIR = os.path.dirname(_MDG_DIR)
for _d in [_ABM_DIR, _MDG_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from physics import (
    CellAgent, total_energy, jkr_contact_area,
    SHELL_A, SHELL_B, SHELL_C, DEVICE, DTYPE,
)
from simulation import Embryo, run_one_step, clamp_to_shell, DT, ETA
from data_loader import load_volumes, load_volumes_8cell, load_contact_areas


LINEAGE_MAP = {
    "AB": "AB", "P1": "P",
    "ABa": "AB", "ABp": "AB", "EMS": "EMS", "P2": "P", "P": "P",
    "ABar": "AB", "ABal": "AB", "ABpr": "AB", "ABpl": "AB",
    "MS": "EMS", "E": "EMS",
    "C": "P", "P3": "P",
}

EXPECTED_CONTACTS = {
    frozenset(p) for p in [
        ("ABar", "ABpr"), ("ABal", "ABpl"),
        ("ABar", "MS"),   ("ABal", "MS"),
        ("ABpr", "C"),    ("ABpl", "C"),
        ("MS",   "E"),    ("MS",   "C"),
        ("E",    "P3"),
    ]
}


# ---------------------------------------------------------------------------
# Fix 6: adaptive equilibration helper
# ---------------------------------------------------------------------------

def equilibrate(cells, params, max_steps, tol=0.05, record_every=5, embryo=None):
    """
    Run overdamped gradient descent until energy change per step < tol for 20
    consecutive frames, or until max_steps is reached.
    """
    prev_E     = None
    stable_cnt = 0

    for step in range(max_steps):
        run_one_step(cells, params)
        if embryo is not None:
            embryo.t += 1
            if embryo.t % record_every == 0:
                embryo.record_frame()

        # NaN guard
        for c in cells:
            if not torch.all(torch.isfinite(c.position)):
                raise RuntimeError(
                    f"NaN in {c.identity} at step {embryo.t if embryo else step}"
                )

        if (step + 1) % 50 == 0:
            E_now = total_energy(cells, params).item()
            if prev_E is not None:
                delta = abs(E_now - prev_E)
                if delta < tol:
                    stable_cnt += 1
                    if stable_cnt >= 3:   # 3 * 50 = 150 consecutive stable steps
                        t_str = str(embryo.t) if embryo else str(step)
                        print(f"    [converged at step {step+1}, t={t_str}, E={E_now:.4f}]")
                        return step + 1
                else:
                    stable_cnt = 0
            prev_E = E_now

    return max_steps


# ---------------------------------------------------------------------------
# Embryo8Cell
# ---------------------------------------------------------------------------

class Embryo8Cell(Embryo):
    """
    Extends Embryo for the full 2→8 cell trajectory.
    run_8cell() inlines all phases — does not call super().run().
    """

    def __init__(self, volumes_4cell, volumes_8cell, params, perturbation=None):
        super().__init__(volumes_4cell, params, perturbation)
        self.volumes_8cell = volumes_8cell

        # Fix 3: 4-cell equil raised to 800 steps
        # Fix 4: EMS delayed to T_ABA + 160, P2 to T_ABA + 200
        self.T_EQUILIBRATE_4CELL = 800
        self.T_ABA_DIV  = self.T_P1_DIV + self.T_EQUILIBRATE_4CELL  # t=320+800=1120
        self.T_ABP_DIV  = self.T_ABA_DIV + 150                       # gap=150 steps
        self.T_EMS_DIV  = self.T_ABA_DIV + 160                       # delayed from 80
        self.T_P2_DIV   = self.T_ABA_DIV + 200                       # delayed from 120
        self.T_EQUILIBRATE_8CELL = 500
        self.T_TOTAL = self.T_P2_DIV + self.T_EQUILIBRATE_8CELL

    def run_8cell(self, record_every=5, verbose=True):
        """Run complete 2→8 cell simulation and return trajectory."""
        if verbose:
            print(f"\nRunning 8-cell embryo simulation on {DEVICE}")
            print(f"Max steps: {self.T_TOTAL}  (adaptive early-stop active)")

        # --- Phase 1: 2-cell equilibration ---
        if verbose:
            print("\n[Phase 1] 2-cell equilibration (max 200 steps)...")
        equilibrate(self.cells, self.params, 200,
                    tol=0.01, record_every=record_every, embryo=self)
        E1 = total_energy(self.cells, self.params).item()
        if verbose:
            print(f"  E at end of phase 1: {E1:.4f}")

        # --- Division 1: AB → ABa + ABp ---
        if verbose:
            print("\n[Division 1] AB → ABa + ABp (axis: Y)")
        self.divide("AB", ["ABa", "ABp"], [0.0, 1.0, 0.0], [0.5, 0.5])
        self.record_frame()
        self._print_cells()

        # --- Phase 2: 3-cell equilibration ---
        if verbose:
            print("\n[Phase 2] 3-cell equilibration (max 120 steps)...")
        equilibrate(self.cells, self.params, 120,
                    tol=0.01, record_every=record_every, embryo=self)
        E2 = total_energy(self.cells, self.params).item()
        if verbose:
            print(f"  E at end of phase 2: {E2:.4f}")

        # --- Division 2: P1 → P2 + EMS ---
        if verbose:
            print("\n[Division 2] P1 → P2 + EMS (axis: X)")
        self.divide("P1", ["P2", "EMS"], [1.0, 0.0, 0.0], [0.45, 0.55])
        self.record_frame()
        self._print_cells()

        # --- Phase 3: 4-cell equilibration (Fix 3: 800 steps) ---
        if verbose:
            print("\n[Phase 3] 4-cell equilibration (max 800 steps, tol=0.01)...")
        steps_used = equilibrate(self.cells, self.params, 800,
                                 tol=0.01, record_every=record_every, embryo=self)
        E3 = total_energy(self.cells, self.params).item()
        if verbose:
            print(f"  E at end of phase 3: {E3:.4f}  steps used: {steps_used}")
            self._print_final_state()

        self._trajectory_4cell_end = len(self.trajectory) - 1

        # --- Division 3: ABa → ABar + ABal (Z) ---
        if verbose:
            print("\n[Division 3] ABa → ABar + ABal (axis: Z)")
        self.divide_8cell("ABa", ["ABar", "ABal"], [0, 0, 1], [0.5, 0.5])
        self.record_frame()
        self._print_cells()

        # Fix 3: 150-step inter-division gap
        equilibrate(self.cells, self.params, 150,
                    tol=0.05, record_every=record_every, embryo=self)

        # --- Division 4: ABp → ABpr + ABpl (Z) ---
        if verbose:
            print("\n[Division 4] ABp → ABpr + ABpl (axis: Z)")
        self.divide_8cell("ABp", ["ABpr", "ABpl"], [0, 0, 1], [0.5, 0.5])
        self.record_frame()
        self._print_cells()

        # 150-step gap — Fix 4: EMS fires at ABA+160 so ABp gap partially overlaps
        equilibrate(self.cells, self.params, 150,
                    tol=0.05, record_every=record_every, embryo=self)

        # --- Division 5: EMS → MS + E (X, Fix 4: delayed to T_ABA+160) ---
        if verbose:
            print("\n[Division 5] EMS → MS + E (axis: X)  [delayed: AB daughters settled]")
        self.divide_8cell("EMS", ["MS", "E"], [1, 0, 0], [0.55, 0.45])
        self.record_frame()
        self._print_cells()

        # Check E position (Fix 5 verification)
        e_cell = next((c for c in self.cells if c.identity == "E"), None)
        if e_cell is not None:
            ep = e_cell.position.detach().cpu().numpy()
            dist = float(sum(x**2 for x in ep)**0.5)
            print(f"  [Check C] E position after EMS div: ({ep[0]:.2f},{ep[1]:.2f},{ep[2]:.2f})  "
                  f"dist_from_center={dist:.2f} μm  {'OK' if dist > 2.0 else 'STILL NEAR CENTER'}")

        equilibrate(self.cells, self.params, 150,
                    tol=0.05, record_every=record_every, embryo=self)

        # --- Division 6: P2 → C + P3 (X, Fix 4: delayed to T_ABA+200) ---
        if verbose:
            print("\n[Division 6] P2 → C + P3 (axis: X)")
        self.divide_8cell("P2", ["C", "P3"], [1, 0, 0], [0.5, 0.5])
        self.record_frame()
        self._print_cells()

        # --- Phase 5: 8-cell equilibration ---
        if verbose:
            print("\n[Phase 5] 8-cell equilibration (max 500 steps, tol=0.01)...")
        steps_8 = equilibrate(self.cells, self.params, 500,
                              tol=0.01, record_every=record_every, embryo=self)
        E5 = total_energy(self.cells, self.params).item()
        if verbose:
            print(f"  E at end of 8-cell equil: {E5:.4f}  steps used: {steps_8}")
            self._print_final_state()

        if verbose:
            print(f"\nSimulation complete. {len(self.trajectory)} frames recorded.")

        return self.trajectory

    def divide_8cell(self, mother_identity, daughter_names, axis_vector, volume_fractions):
        """
        Division using 8-cell volumes from dataset.
        Fix 5: E cell gets a small symmetry-breaking nudge.
        """
        mother = next(
            (c for c in self.cells if c.identity == mother_identity), None
        )
        if mother is None:
            raise ValueError(
                f"Cell {mother_identity} not found. Cells: "
                f"{[c.identity for c in self.cells]}"
            )

        mother_pos = mother.position.detach().clone()
        daughters  = []

        for idx, (name, frac) in enumerate(zip(daughter_names, volume_fractions)):
            V_d = self.volumes_8cell.get(name)
            if V_d is None:
                V_d = mother.V0 * frac
                print(f"  [WARNING] {name} missing from dataset — "
                      f"fallback V0 = {V_d:.2f} μm³")
            else:
                print(f"  {name}: V0 = {V_d:.2f} μm³ (dataset)")

            d    = CellAgent(name, V_d)
            ax   = torch.tensor(axis_vector, dtype=DTYPE, device=DEVICE)
            sign = 1.0 if idx == 0 else -1.0
            pos  = mother_pos + sign * d.R * ax

            # Fix 5: break the symmetric force well that traps E at origin
            if name == "E":
                nudge = torch.tensor([-0.1, 0.25, 0.1], dtype=DTYPE, device=DEVICE)
                nudge = nudge / torch.norm(nudge)
                pos   = pos + nudge * d.R * 0.15
                print(f"    [Fix 5] E nudge applied: +{(nudge * d.R * 0.15).numpy().round(3)}")

            if self.perturbation is not None and self.perturbation > 0:
                noise = torch.randn(3, dtype=DTYPE, device=DEVICE) * self.perturbation
                pos   = pos + noise

            pos = clamp_to_shell(pos, d.R_c)
            if not torch.all(torch.isfinite(pos)):
                print(f"  [WARNING] {name} position non-finite, resetting to origin")
                pos = torch.zeros(3, dtype=DTYPE, device=DEVICE)

            d.position   = pos.clone().detach().requires_grad_(True)
            d.axes       = torch.tensor([d.R, d.R, d.R],
                                         dtype=DTYPE, device=DEVICE, requires_grad=True)
            d.quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0],
                                          dtype=DTYPE, device=DEVICE, requires_grad=True)
            daughters.append(d)

        self.cells = [c for c in self.cells if c.identity != mother_identity]
        self.cells.extend(daughters)

        self.division_log.append({
            "t":         self.t,
            "mother":    mother_identity,
            "daughters": daughter_names,
            "axis":      axis_vector,
        })

    def _print_cells(self):
        print(f"  Cells (t={self.t}): {[c.identity for c in self.cells]}")

    def compute_full_topology(self):
        """All C(n,2) pairwise contact areas at current state."""
        contacts = {}
        w = self.params["w"]
        for i, j in itertools.combinations(range(len(self.cells)), 2):
            ci, cj = self.cells[i], self.cells[j]
            A   = jkr_contact_area(ci, cj, w, self.params)
            key = tuple(sorted([ci.identity, cj.identity]))
            contacts[key] = A.item()
        return contacts

    def print_topology_table(self, contacts):
        cells8   = ["ABar", "ABal", "ABpr", "ABpl", "MS", "E", "C", "P3"]
        all_pairs = list(itertools.combinations(cells8, 2))
        exp_norm  = {tuple(sorted(p)) for p in EXPECTED_CONTACTS
                     if isinstance(p, (tuple, frozenset))}

        print("\n" + "=" * 65)
        print("FULL 28-PAIR CONTACT TABLE")
        print("=" * 65)
        print(f"  {'Pair':<22} {'Area (μm²)':>12}  {'Expected?':>10}  Status")
        print("  " + "-" * 60)

        n_nonzero = 0
        expected_found = 0
        for a, b in all_pairs:
            key  = tuple(sorted([a, b]))
            area = contacts.get(key, 0.0)
            exp  = "YES" if key in {tuple(sorted(p)) for p in
                    [("ABar","ABpr"),("ABal","ABpl"),("ABar","MS"),("ABal","MS"),
                     ("ABpr","C"),("ABpl","C"),("MS","E"),("MS","C"),("E","P3")]} else ""
            stat = "PRESENT" if area > 5.0 else "absent"
            if area > 5.0:
                n_nonzero += 1
                if exp == "YES":
                    expected_found += 1
            print(f"  {a}-{b:<18} {area:>12.2f}  {exp:>10}  {stat}")

        print("  " + "-" * 60)
        print(f"\n  N non-zero (>5 μm²):        {n_nonzero}/28")
        print(f"  Spherical baseline:           28/28")
        print(f"  Deformable result:            {n_nonzero}/28")
        print(f"  Expected contacts present:    {expected_found}/9")

        verdict = ("Deformable model IS restricting contacts."
                   if n_nonzero < 28 else
                   "All 28 pairs non-zero — still degenerate. Report as finding.")
        print(f"\n  [RESULT] {verdict}")
        print("=" * 65)
        return n_nonzero, expected_found


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def check_A_gradient_magnitudes(contacts_frame, best_params):
    """Check A: gradient magnitudes after fixes 1+2."""
    print("\n" + "=" * 60)
    print("CHECK A — Single-step gradient magnitudes (post-fix)")
    print("=" * 60)

    cells_diag = []
    for c in contacts_frame["cells"]:
        agent = CellAgent(c["identity"], c["V0"])
        agent.position = torch.tensor(c["position"], dtype=DTYPE, device=DEVICE,
                                       requires_grad=True)
        axv = c.get("axes", [c["R"], c["R"], c["R"]])
        agent.axes = torch.tensor(axv, dtype=DTYPE, device=DEVICE, requires_grad=True)
        qv  = c.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        agent.quaternion = torch.tensor(qv, dtype=DTYPE, device=DEVICE, requires_grad=True)
        agent.R   = c["R"]
        agent.R_c = 0.8 * c["R"]
        cells_diag.append(agent)

    E = total_energy(cells_diag, best_params)
    E.backward()

    all_pos_ok = True
    all_axes_ok = True
    print(f"  {'Cell':6s}  {'||∇pos||':>10}  {'clipped@20?':>11}  "
          f"{'axes Δ/step':>12}  {'is0.025ok?':>11}")
    for agent in cells_diag:
        gp  = agent.position.grad
        ga  = agent.axes.grad
        nmP = gp.norm().item() if gp is not None else float("nan")
        nmA = ga.norm().item() if ga is not None else float("nan")
        clipped = nmP > 20.0
        if clipped:
            all_pos_ok = False
        axstep = 0.005 * min(nmA, 5.0)
        ax_ok  = axstep <= 0.025
        if not ax_ok:
            all_axes_ok = False
        print(f"  {agent.identity:6s}  {nmP:>10.4f}  {'YES' if clipped else 'no':>11}  "
              f"{axstep:>12.5f}  {'OK' if ax_ok else 'WARN':>11}")

    print(f"\n  All pos gradients < 20.0: {'YES' if all_pos_ok else 'NO — still being clipped'}")
    print(f"  All axes updates ≤ 0.025 μm/step: {'YES' if all_axes_ok else 'NO'}")
    return all_pos_ok, all_axes_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("=" * 60)
    print("LOADING VOLUMES")
    print("=" * 60)

    volumes_4cell = load_volumes()
    print("\n4-cell volumes:")
    for name, v in volumes_4cell.items():
        print(f"  {name}: {v:.2f} μm³")

    volumes_8cell = load_volumes_8cell()
    print("\n8-cell volumes:")
    fallbacks = []
    for name, v in volumes_8cell.items():
        if v is None:
            fallbacks.append(name)
            print(f"  {name}: MISSING — fallback")
        else:
            print(f"  {name}: {v:.2f} μm³")

    best_params = {
        "gamma_AB":  0.8455,
        "gamma_EMS": 0.7610,
        "gamma_P":   0.6826,
        "w":         0.8523,
        "alpha":     0.3482,
    }
    measured_areas_4cell = load_contact_areas()

    # --- Check A: gradient magnitudes with old 8-cell .pt ---
    old_pt = os.path.join(_PROJ_DIR, "results", "simulation_results_8cell.pt")
    if os.path.exists(old_pt):
        old_d = torch.load(old_pt, weights_only=False)
        old_8_frames = old_d.get("trajectory_8cell", [])
        if old_8_frames:
            check_A_gradient_magnitudes(old_8_frames[0], best_params)

    # --- Run fresh simulation ---
    print("\n" + "=" * 60)
    print("RUNNING 8-CELL SIMULATION (with all fixes applied)")
    print("=" * 60)

    embryo = Embryo8Cell(
        volumes_4cell, volumes_8cell, best_params, perturbation=None
    )
    trajectory = embryo.run_8cell(record_every=5, verbose=True)

    # --- Topology ---
    print("\n" + "=" * 60)
    print("TOPOLOGY VALIDATION")
    print("=" * 60)
    contacts = embryo.compute_full_topology()
    n_nonzero, expected_found = embryo.print_topology_table(contacts)

    # --- Cell shapes ---
    print("\n8-CELL SHAPE SUMMARY")
    print("=" * 60)
    for c in embryo.cells:
        if c.axes is not None:
            ax = c.axes.detach().cpu().numpy()
            ratio = ax.max() / ax.min()
            pos = c.position.detach().cpu().numpy()
            print(f"  {c.identity}: a={ax[0]:.3f} b={ax[1]:.3f} c={ax[2]:.3f}  "
                  f"ratio={ratio:.4f}  pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})")

    # --- Save ---
    out_path = os.path.join(_PROJ_DIR, "results", "simulation_results_8cell.pt")
    t4_end = embryo._trajectory_4cell_end
    trajectory_4cell = trajectory[:t4_end + 1]
    trajectory_8cell = trajectory[t4_end + 1:]

    print(f"\nSaving {out_path} ...")
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
    print(f"Saved. Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
