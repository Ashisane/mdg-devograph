"""
test_deformable.py — Test suite for the deformable ellipsoid cell model.

Tests 1-6 are unit tests of the new geometry and physics primitives.
Tests 7-8 are integration tests: full 4-cell simulation with deformable cells.

Run from the project root:
    python test/test_deformable.py
"""

import sys
import os
import math
import torch

# Both physics and simulation live in mdg/abm/
_ABM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mdg", "abm")
_MDG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mdg")
for _d in [_ABM_DIR, _MDG_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from physics import (
    CellAgent, DEVICE, DTYPE,
    quaternion_to_rotation, effective_radius, ellipsoid_contact_distance,
    shape_energy, total_energy,
    SHELL_A, SHELL_B, SHELL_C, K_ELASTIC
)

# Calibrated best params from Prompt 2 validation
BEST_PARAMS = {
    'gamma_AB':  0.8455,
    'gamma_EMS': 0.7610,
    'gamma_P':   0.6826,
    'w':         0.8523,
    'alpha':     0.3482,
}

RESULTS = {}


# ---------------------------------------------------------------------------

def test_1_backward_compat():
    """
    TEST 1: Backward compatibility
    Run test_physics.py's 6 original tests. They must all pass now that
    run_inner_loop only optimises positions and the fallback path is active.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Backward Compatibility (original 6 tests)")
    print("=" * 60)

    # Inline the original 6 checks so this file is self-contained
    import test_physics as tp
    r1 = tp.test_1_jkr_formula()
    r2 = tp.test_2_shell_confinement()
    r3 = tp.test_3_overlap_repulsion()
    r4 = tp.test_4_volume_elasticity()
    r5 = tp.test_5_cortical_flow()
    r6 = tp.test_6_inner_loop_convergence()

    passed = all([r1, r2, r3, r4, r5, r6])
    print(f"\n  TEST 1: {'PASS' if passed else 'FAIL'}")
    return passed


def test_2_shape_init():
    """
    TEST 2: Shape initialisation
    After set_position, axes == [R,R,R] and quaternion == [1,0,0,0].
    """
    print("\n" + "=" * 60)
    print("TEST 2: Shape Initialisation")
    print("=" * 60)

    V0   = (4 / 3) * math.pi * 8.0**3
    cell = CellAgent("ABa", V0)
    cell.set_position(0.0, 0.0, 0.0)

    axes_ok = (
        cell.axes is not None and
        cell.axes.shape == (3,) and
        torch.allclose(cell.axes, torch.tensor([cell.R, cell.R, cell.R], dtype=DTYPE), atol=1e-9)
    )
    quat_ok = (
        cell.quaternion is not None and
        cell.quaternion.shape == (4,) and
        torch.allclose(cell.quaternion, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE), atol=1e-9)
    )
    norm_ok = abs(torch.norm(cell.quaternion).item() - 1.0) < 1e-9

    print(f"  axes shape == (3,): {cell.axes.shape} — {'PASS' if axes_ok else 'FAIL'}")
    print(f"  axes == [R,R,R]: {'PASS' if axes_ok else 'FAIL'}")
    print(f"  quaternion shape == (4,): {cell.quaternion.shape} — {'PASS' if quat_ok else 'FAIL'}")
    print(f"  quaternion == [1,0,0,0]: {'PASS' if quat_ok else 'FAIL'}")
    print(f"  ||quaternion|| ≈ 1: {torch.norm(cell.quaternion).item():.10f} — {'PASS' if norm_ok else 'FAIL'}")

    passed = axes_ok and quat_ok and norm_ok
    print(f"\n  TEST 2: {'PASS' if passed else 'FAIL'}")
    return passed


def test_3_effective_radius_sphere():
    """
    TEST 3: effective_radius returns R for a sphere in any direction.
    """
    print("\n" + "=" * 60)
    print("TEST 3: effective_radius == R for sphere")
    print("=" * 60)

    V0   = (4 / 3) * math.pi * 8.0**3
    cell = CellAgent("ABa", V0)
    cell.set_position(0.0, 0.0, 0.0)

    directions = [
        torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),
        torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE),
        torch.nn.functional.normalize(torch.tensor([1.0, 1.0, 1.0], dtype=DTYPE), dim=0),
        torch.nn.functional.normalize(torch.tensor([1.0, -2.0, 3.0], dtype=DTYPE), dim=0),
    ]

    passed = True
    with torch.no_grad():
        for d in directions:
            r = effective_radius(cell, d).item()
            ok = abs(r - cell.R) < 1e-4
            print(f"  direction {d.numpy().round(3)}: r_eff={r:.6f}, R={cell.R:.6f} — {'PASS' if ok else 'FAIL'}")
            passed = passed and ok

    print(f"\n  TEST 3: {'PASS' if passed else 'FAIL'}")
    return passed


def test_4_shape_energy_zero_at_rest():
    """
    TEST 4: shape_energy == 0 when axes = [R, R, R].
    """
    print("\n" + "=" * 60)
    print("TEST 4: shape_energy == 0 at rest")
    print("=" * 60)

    V0   = (4 / 3) * math.pi * 8.0**3
    cell = CellAgent("ABa", V0)
    cell.set_position(0.0, 0.0, 0.0)

    with torch.no_grad():
        E = shape_energy(cell).item()

    print(f"  shape_energy = {E:.2e}")
    passed = E < 1e-6
    print(f"\n  TEST 4: {'PASS' if passed else 'FAIL'}")
    return passed


def test_5_shape_energy_positive_deformed():
    """
    TEST 5: shape_energy > 0 for a volume-conserving deformation.
    axes = [R*1.5, R*0.9, R*0.74] satisfies a*b*c ≈ R³ (approx volume-conserving).
    """
    print("\n" + "=" * 60)
    print("TEST 5: shape_energy > 0 when deformed")
    print("=" * 60)

    V0   = (4 / 3) * math.pi * 8.0**3
    cell = CellAgent("ABa", V0)
    cell.set_position(0.0, 0.0, 0.0)

    # 1.5 * 0.9 * 0.74 ≈ 0.999 — close to volume-conserving, but shape differs from sphere
    with torch.no_grad():
        cell.axes = torch.tensor(
            [cell.R * 1.5, cell.R * 0.9, cell.R * 0.74],
            dtype=DTYPE, device=DEVICE, requires_grad=True
        )
        E = shape_energy(cell).item()

    print(f"  axes = [{cell.R*1.5:.3f}, {cell.R*0.9:.3f}, {cell.R*0.74:.3f}]")
    print(f"  shape_energy = {E:.6f}")
    passed = E > 0
    print(f"\n  TEST 5: {'PASS' if passed else 'FAIL'}")
    return passed


def test_6_differentiable():
    """
    TEST 6: ellipsoid_contact_distance is differentiable — .backward() works,
    all gradients are finite.
    """
    print("\n" + "=" * 60)
    print("TEST 6: Differentiability of ellipsoid geometry")
    print("=" * 60)

    V0_a = (4 / 3) * math.pi * 8.0**3
    V0_b = (4 / 3) * math.pi * 7.0**3

    a = CellAgent("ABa", V0_a)
    b = CellAgent("ABp", V0_b)
    a.set_position(-5.0, 0.0, 0.0)
    b.set_position( 5.0, 0.0, 0.0)

    r_i, r_j, d_vec = ellipsoid_contact_distance(a, b)
    loss = r_i + r_j
    loss.backward()

    grads_a = [a.position.grad, a.axes.grad, a.quaternion.grad]
    grads_b = [b.position.grad, b.axes.grad, b.quaternion.grad]

    passed = True
    for name, g in zip(
        ["a.position", "a.axes", "a.quaternion", "b.position", "b.axes", "b.quaternion"],
        grads_a + grads_b
    ):
        if g is None:
            print(f"  {name}: grad is None — FAIL")
            passed = False
        elif not torch.all(torch.isfinite(g)):
            print(f"  {name}: non-finite gradient — FAIL")
            passed = False
        else:
            print(f"  {name}: grad OK  {g.numpy().round(5)}")

    print(f"\n  TEST 6: {'PASS' if passed else 'FAIL'}")
    return passed


def _run_4cell_sim():
    """Run one 4-cell simulation with deformable cells. Returns Embryo instance."""
    # Import here to avoid circular at module load time
    import sys, os  # already imported at top, but be explicit for clarity
    _sim_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mdg", "abm")
    if _sim_dir not in sys.path:
        sys.path.insert(0, _sim_dir)

    from simulation import Embryo
    from data_loader import load_volumes

    volumes = load_volumes()
    embryo  = Embryo(volumes, BEST_PARAMS, perturbation=0.0)
    embryo.run(record_every=50, verbose=False)
    return embryo


def test_7_4cell_sim_no_nan():
    """
    TEST 7: Full 4-cell simulation completes without NaN.
    ABa-P2 contact < 10 μm². All axes remain positive.
    """
    print("\n" + "=" * 60)
    print("TEST 7: 4-cell simulation — no NaN, correct topology")
    print("=" * 60)

    print("  Running simulation (this takes ~30-60s) ...")
    embryo = _run_4cell_sim()

    # Check for NaN in any trajectory frame
    nan_found = False
    for frame in embryo.trajectory:
        for c in frame["cells"]:
            for v in c["position"]:
                if not math.isfinite(v):
                    nan_found = True
            if "axes" in c:
                for v in c["axes"]:
                    if not math.isfinite(v):
                        nan_found = True

    check_nan = not nan_found
    print(f"  NaN in trajectory: {'none — PASS' if check_nan else 'FOUND — FAIL'}")

    # Check ABa-P2 contact in last frame
    final_contacts = embryo._compute_contacts()
    aba_p2 = final_contacts.get("ABa-P2", final_contacts.get("P2-ABa", 0.0))
    check_topo = aba_p2 < 10.0
    print(f"  ABa-P2 contact = {aba_p2:.4f} μm² (< 10): {'PASS' if check_topo else 'FAIL'}")

    # Check all axes positive in final state
    axes_ok = True
    for c in embryo.cells:
        if c.axes is not None:
            axes_vals = c.axes.detach().cpu().numpy()
            if any(v <= 0 for v in axes_vals):
                axes_ok = False
                print(f"  {c.identity} axes not positive: {axes_vals} — FAIL")
            else:
                print(f"  {c.identity} axes: {axes_vals.round(3)} — OK")

    print(f"  All axes > 0: {'PASS' if axes_ok else 'FAIL'}")

    passed = check_nan and check_topo and axes_ok
    print(f"\n  TEST 7: {'PASS' if passed else 'FAIL'}")

    # Store embryo for test 8
    RESULTS["_embryo"] = embryo
    return passed


def test_8_cells_deform():
    """
    TEST 8: At least one cell in each contacting pair has max/min axes ratio > 1.005.
    The 4-cell equilibrium consistently produces 1.009-1.043 across all pairs.
    This confirms the model produces measurable deformation at contact interfaces.
    """
    print("\n" + "=" * 60)
    print("TEST 8: Cells deform under contact (axis asymmetry > 1.005)")
    print("=" * 60)

    embryo = RESULTS.get("_embryo")
    if embryo is None:
        print("  [SKIP] No embryo from TEST 7 — re-running simulation...")
        embryo = _run_4cell_sim()

    CONTACTING_PAIRS = [
        ("ABa", "ABp"), ("ABa", "EMS"), ("ABp", "EMS"), ("ABp", "P2"), ("EMS", "P2")
    ]

    cell_axes = {}
    for c in embryo.cells:
        if c.axes is not None:
            ax = c.axes.detach().cpu()
            cell_axes[c.identity] = ax

    print(f"\n  Final axes per cell:")
    for ident, ax in cell_axes.items():
        ratio = (ax.max() / ax.min()).item()
        print(f"    {ident}: a={ax[0]:.4f} b={ax[1]:.4f} c={ax[2]:.4f}  max/min={ratio:.4f}")

    passed = True
    for c1, c2 in CONTACTING_PAIRS:
        ax1 = cell_axes.get(c1)
        ax2 = cell_axes.get(c2)
        if ax1 is None or ax2 is None:
            continue

        r1 = (ax1.max() / ax1.min()).item()
        r2 = (ax2.max() / ax2.min()).item()
        pair_ok = r1 > 1.005 or r2 > 1.005

        if not pair_ok:
            passed = False
        print(f"  {c1}-{c2}: r1={r1:.4f}  r2={r2:.4f}  deformed: {'PASS' if pair_ok else 'FAIL'}")

    print(f"\n  TEST 8: {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("DEFORMABLE CELL MODEL TEST SUITE")
    print("=" * 60)

    RESULTS["TEST 1: Backward Compat"]     = test_1_backward_compat()
    RESULTS["TEST 2: Shape Init"]          = test_2_shape_init()
    RESULTS["TEST 3: effective_radius"]    = test_3_effective_radius_sphere()
    RESULTS["TEST 4: shape_energy rest"]   = test_4_shape_energy_zero_at_rest()
    RESULTS["TEST 5: shape_energy deform"] = test_5_shape_energy_positive_deformed()
    RESULTS["TEST 6: Differentiable"]      = test_6_differentiable()
    RESULTS["TEST 7: 4-cell sim"]          = test_7_4cell_sim_no_nan()
    RESULTS["TEST 8: Cell deformation"]    = test_8_cells_deform()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, result in RESULTS.items():
        if name.startswith("_"):
            continue
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_pass = False

    print(f"\nALL TESTS: {'PASS' if all_pass else 'FAIL'}")
