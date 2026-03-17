"""
test_physics.py — Test suite for the biophysical simulation engine.

Runs 6 tests and prints PASS/FAIL for each.
All 6 tests must pass before Prompt 2 begins.
"""

import math
import torch

from physics import (
    CellAgent, DEVICE, DTYPE,
    shell_energy, volume_energy, overlap_repulsion,
    jkr_contact_area, adhesion_energy, cortical_flow_energy,
    total_energy, run_inner_loop,
    SHELL_A, SHELL_B, SHELL_C
)

DEFAULT_PARAMS = {
    'gamma_AB': 1.0,
    'gamma_EMS': 0.7,
    'gamma_P': 0.4,
    'w': 0.5,
    'alpha': 0.1,
}


def test_1_jkr_formula():
    """
    TEST 1: JKR formula
      Two cells R=8μm, γ=1.0, w=0.5, distance=14μm
      Assert A_contact > 0 and finite
      Assert A_contact decreases as distance increases
    """
    print("\n" + "=" * 60)
    print("TEST 1: JKR Formula")
    print("=" * 60)

    V0 = (4.0 / 3.0) * math.pi * 8.0 ** 3

    cell_a = CellAgent("ABa", V0)
    cell_b = CellAgent("ABp", V0)
    params = DEFAULT_PARAMS.copy()

    cell_a.set_position(0.0, 0.0, 0.0)
    cell_b.set_position(14.0, 0.0, 0.0)

    with torch.no_grad():
        A14 = jkr_contact_area(cell_a, cell_b, params['w'], params)
        A14_val = A14.item()

    print(f"  R = {cell_a.R:.2f} μm")
    print(f"  R_c = {cell_a.R_c:.2f} μm")
    print(f"  R_c_i + R_c_j = {cell_a.R_c + cell_b.R_c:.2f} μm")
    print(f"  Distance = 14.0 μm")
    print(f"  A_contact (d=14) = {A14_val:.6f} μm²")

    check1 = A14_val > 0 and math.isfinite(A14_val)
    print(f"  A_contact > 0 and finite: {'PASS' if check1 else 'FAIL'}")

    cell_b.set_position(17.0, 0.0, 0.0)
    with torch.no_grad():
        A17 = jkr_contact_area(cell_a, cell_b, params['w'], params)
        A17_val = A17.item()

    print(f"  A_contact (d=17) = {A17_val:.6f} μm²")
    check2 = A14_val > A17_val
    print(f"  A_contact decreases with distance: {'PASS' if check2 else 'FAIL'}")

    passed = check1 and check2
    print(f"\n  TEST 1: {'PASS' if passed else 'FAIL'}")
    return passed


def test_2_shell_confinement():
    """
    TEST 2: Shell confinement
      Cell at position [30, 0, 0] (outside shell) → E_shell > 0
      Cell at [0, 0, 0] (center) → E_shell == 0
    """
    print("\n" + "=" * 60)
    print("TEST 2: Shell Confinement")
    print("=" * 60)

    V0 = (4.0 / 3.0) * math.pi * 5.0 ** 3
    cell = CellAgent("ABa", V0)

    cell.set_position(30.0, 0.0, 0.0)
    with torch.no_grad():
        E_out = shell_energy(cell).item()
    print(f"  E_shell at [30, 0, 0] = {E_out:.6f}")
    check1 = E_out > 0
    print(f"  E_shell > 0 (outside): {'PASS' if check1 else 'FAIL'}")

    cell.set_position(0.0, 0.0, 0.0)
    with torch.no_grad():
        E_center = shell_energy(cell).item()
    print(f"  E_shell at [0, 0, 0] = {E_center:.6f}")
    check2 = abs(E_center) < 1e-10
    print(f"  E_shell == 0 (center): {'PASS' if check2 else 'FAIL'}")

    passed = check1 and check2
    print(f"\n  TEST 2: {'PASS' if passed else 'FAIL'}")
    return passed


def test_3_overlap_repulsion():
    """
    TEST 3: Overlap repulsion
      Two cells at distance < R_c_i + R_c_j → E_rep > 0
      Two cells at distance > R_c_i + R_c_j → E_rep == 0
    """
    print("\n" + "=" * 60)
    print("TEST 3: Overlap Repulsion")
    print("=" * 60)

    V0 = (4.0 / 3.0) * math.pi * 8.0 ** 3
    cell_a = CellAgent("ABa", V0)
    cell_b = CellAgent("ABp", V0)
    threshold = cell_a.R_c + cell_b.R_c

    close_d = threshold - 2.0
    cell_a.set_position(0.0, 0.0, 0.0)
    cell_b.set_position(close_d, 0.0, 0.0)
    with torch.no_grad():
        E_close = overlap_repulsion(cell_a, cell_b).item()
    print(f"  R_c_i + R_c_j = {threshold:.2f} μm")
    print(f"  E_rep at d={close_d:.2f} = {E_close:.6f}")
    check1 = E_close > 0
    print(f"  E_rep > 0 (overlapping): {'PASS' if check1 else 'FAIL'}")

    far_d = threshold + 5.0
    cell_b.set_position(far_d, 0.0, 0.0)
    with torch.no_grad():
        E_far = overlap_repulsion(cell_a, cell_b).item()
    print(f"  E_rep at d={far_d:.2f} = {E_far:.6f}")
    check2 = abs(E_far) < 1e-10
    print(f"  E_rep == 0 (separated): {'PASS' if check2 else 'FAIL'}")

    passed = check1 and check2
    print(f"\n  TEST 3: {'PASS' if passed else 'FAIL'}")
    return passed


def test_4_volume_elasticity():
    """
    TEST 4: Volume elasticity
      Cell with V_eff = V0 (no neighbors) → E_volume ≈ 0
    """
    print("\n" + "=" * 60)
    print("TEST 4: Volume Elasticity")
    print("=" * 60)

    V0 = (4.0 / 3.0) * math.pi * 8.0 ** 3
    cell = CellAgent("ABa", V0)
    cell.set_position(0.0, 0.0, 0.0)

    with torch.no_grad():
        E_vol = volume_energy(cell, [cell]).item()
    print(f"  V0 = {V0:.2f} μm³")
    print(f"  R = {cell.R:.2f} μm")
    print(f"  E_volume (no neighbors) = {E_vol:.10f}")
    check = abs(E_vol) < 1e-6
    print(f"  E_volume ≈ 0: {'PASS' if check else 'FAIL'}")

    print(f"\n  TEST 4: {'PASS' if check else 'FAIL'}")
    return check


def test_5_cortical_flow():
    """
    TEST 5: Cortical flow
      P2 cell: verify gradient at x=-5 points in -x direction (force toward +x)
      AB cell → E_cortical == 0
    """
    print("\n" + "=" * 60)
    print("TEST 5: Cortical Flow")
    print("=" * 60)

    alpha = 0.1
    V0 = (4.0 / 3.0) * math.pi * 5.0 ** 3

    p2 = CellAgent("P2", V0)
    p2.set_position(-5.0, 0.0, 0.0)
    with torch.no_grad():
        E_p2 = cortical_flow_energy(p2, alpha).item()
    print(f"  E_cortical (P2, x=-5) = {E_p2:.6f}")

    # E = -alpha * x, so at x=-5, E=+0.5. The force F = -dE/dx = +alpha > 0,
    # which pushes P2 toward the posterior (+x).
    E_p2_tensor = cortical_flow_energy(p2, alpha)
    E_p2_tensor.backward()
    grad_x = p2.position.grad[0].item()
    print(f"  Gradient at x=-5: dE/dx = {grad_x:.6f}")
    print(f"  Force direction: toward {'+ x (posterior)' if grad_x < 0 else '- x (anterior)'}")

    check1 = grad_x < 0
    print(f"  P2 pushed toward +x: {'PASS' if check1 else 'FAIL'}")

    ab = CellAgent("ABa", V0)
    ab.set_position(-5.0, 0.0, 0.0)
    with torch.no_grad():
        E_ab = cortical_flow_energy(ab, alpha).item()
    print(f"  E_cortical (ABa, x=-5) = {E_ab:.6f}")
    check2 = abs(E_ab) < 1e-10
    print(f"  E_cortical == 0 (AB): {'PASS' if check2 else 'FAIL'}")

    passed = check1 and check2
    print(f"\n  TEST 5: {'PASS' if passed else 'FAIL'}")
    return passed


def test_6_inner_loop_convergence():
    """
    TEST 6: Inner loop convergence
      Two cells at random positions inside shell
      Run inner loop
      Assert converged (|ΔE| < threshold)
      Assert both cells inside shell at convergence
    """
    print("\n" + "=" * 60)
    print("TEST 6: Inner Loop Convergence")
    print("=" * 60)

    V0_a = (4.0 / 3.0) * math.pi * 7.0 ** 3
    V0_b = (4.0 / 3.0) * math.pi * 6.0 ** 3

    cell_a = CellAgent("ABa", V0_a)
    cell_b = CellAgent("EMS", V0_b)

    cell_a.set_position(-5.0, 2.0, 1.0)
    cell_b.set_position(5.0, -2.0, -1.0)

    print(f"  Initial: {cell_a}")
    print(f"  Initial: {cell_b}")

    params = DEFAULT_PARAMS.copy()
    final_E, steps = run_inner_loop([cell_a, cell_b], params, verbose=True)

    print(f"  Final E = {final_E:.8f}")
    print(f"  Steps taken = {steps}")

    check1 = steps < 5000
    print(f"  Converged: {'PASS' if check1 else 'FAIL'}")

    def is_inside_shell(cell):
        with torch.no_grad():
            x, y, z = cell.position.tolist()
        f = (x / SHELL_A) ** 2 + (y / SHELL_B) ** 2 + (z / SHELL_C) ** 2
        return f <= 1.05

    inside_a = is_inside_shell(cell_a)
    inside_b = is_inside_shell(cell_b)
    print(f"  Cell A inside shell: {'PASS' if inside_a else 'FAIL'}")
    print(f"  Cell B inside shell: {'PASS' if inside_b else 'FAIL'}")

    with torch.no_grad():
        print(f"  Final positions:")
        print(f"    {cell_a}")
        print(f"    {cell_b}")

    passed = check1 and inside_a and inside_b
    print(f"\n  TEST 6: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    print("=" * 60)
    print("PHYSICS ENGINE TEST SUITE")
    print("=" * 60)

    results = {}

    results["TEST 1: JKR Formula"] = test_1_jkr_formula()
    if not results["TEST 1: JKR Formula"]:
        print("\n!!! TEST 1 FAILED — STOPPING !!!")
        exit(1)

    results["TEST 2: Shell Confinement"] = test_2_shell_confinement()
    results["TEST 3: Overlap Repulsion"] = test_3_overlap_repulsion()
    results["TEST 4: Volume Elasticity"] = test_4_volume_elasticity()
    results["TEST 5: Cortical Flow"] = test_5_cortical_flow()
    results["TEST 6: Inner Loop Convergence"] = test_6_inner_loop_convergence()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print(f"\nALL TESTS: {'PASS' if all_pass else 'FAIL'}")
