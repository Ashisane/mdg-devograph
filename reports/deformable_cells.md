# Report 5 — Deformable Ellipsoid Cell Model

## 1. Test Results

| # | Test | Result |
|---|------|--------|
| 1 | Backward compatibility (original 6 tests) | **PASS** |
| 2 | Shape initialisation (axes, quaternion) | **PASS** |
| 3 | `effective_radius` returns R for sphere | **PASS** |
| 4 | `shape_energy` == 0 at rest | **PASS** |
| 5 | `shape_energy` > 0 when deformed | **PASS** |
| 6 | Ellipsoid geometry is differentiable | **PASS** |
| 7 | 4-cell simulation — no NaN, correct topology | **PASS** |
| 8 | Cells deform under contact (axis ratio > 1.005) | **PASS** |

**Key numbers from TEST 7:**
- ABa-P2 contact = 0.0000 μm² ✓ (forbidden edge correctly absent)
- No NaN in any trajectory frame ✓
- All axes remained positive throughout ✓

---

## 2. Equilibrium Cell Shapes

Final axes (a, b, c) at 4-cell equilibrium with calibrated params
(γ_AB=0.8455, γ_EMS=0.7610, γ_P=0.6826, w=0.8523, α=0.3482):

| Cell | a (μm) | b (μm) | c (μm) | max/min ratio |
|------|--------|--------|--------|--------------|
| ABa  | 8.882  | 8.802  | 8.862  | 1.009        |
| ABp  | 8.882  | 8.802  | 8.862  | 1.009        |
| EMS  | 7.845  | 7.965  | 8.185  | 1.043        |
| P2   | 7.487  | 7.467  | 7.467  | 1.003        |

EMS shows the strongest deformation (4.3%) because it is sandwiched between three other
cells (ABa, ABp, P2) — the forces from multiple contact interfaces flatten it
asymmetrically. ABa/ABp deform symmetrically (~0.9%) because they contact each other
and EMS with comparable forces.

---

## 3. K_elastic Value Used

**K_elastic = 0.5** (tuned down from the default 2.0)

At K_elastic=2.0, axis ratios were only ~0.009% — too stiff for detectable asymmetry
within the 820-step simulation window. At 0.5, cells deform at the contact interface
while still recovering toward R when contact is removed.

The axes learning rate in `run_one_step` is set to 5× the position LR (0.05 vs 0.01)
because the contact gradient through `effective_radius` is ~10× smaller in magnitude
than the position gradient. This compensates without changing the physics — it just
respects the different gradient scales between positional and shape DOFs.

---

## 4. 4-Cell Topology

The forbidden ABa-P2 contact (= 0.0000 μm²) and the 3+1 diamond topology are preserved
identically under deformable cell dynamics. The deformation shifts contact areas
slightly compared to the rigid sphere model (effective radii now vary with contact
direction) but the qualitative topology is robust.

---

## 5. Bugs Found and Fixed

During implementation, two bugs in the existing codebase were discovered and fixed:

1. **Broken import chain**: `data_loader.py` used `__file__`-relative path expecting
   `datasets/` at `mdg/datasets/`, but the datasets were moved to the project root.
   Fixed: path now walks up to project root before joining `datasets/`.

2. **Daughter cell axes not initialised**: `Embryo.divide()` set daughter `.position`
   via direct tensor assignment (bypassing `set_position()`), leaving `.axes` and
   `.quaternion` as `None`. This caused a silent failure in TEST 8 (cell_axes dict
   was empty, all pairs skipped). Fixed: `divide()` now explicitly initialises all
   three DOFs for each daughter.

---

## 6. Known Limitations

1. **Small deformation amplitude.** At 820-step equilibration, cells reach ~1-4% axis
   asymmetry. True biological deformation at cell contacts is ~10-30%. Longer
   simulations or higher axes LR would increase this.

2. **Shape inheritance.** Daughter cells start as spheres after division regardless of
   mother cell shape. The deformed configuration is lost at each division event.

3. **Orientation decoupling.** Quaternion torque from autograd is very small (~0 for
   near-spherical cells). Orientation only matters significantly when axis ratio > 1.2.

4. **Effective radius approximation.** Uses the support function formula for ellipsoids,
   which gives the correct surface point in a given direction. This is a first-order
   approximation; Minkowski-sum contact geometry would be more accurate.

---

## 7. Next Steps

To enable 8-cell simulation:
- Add vortex/cytoplasmic rotation term (previously identified as missing)
- Specify 3rd-division axes from CShaper lineage data
- May need K_elastic ≤ 0.2 at 8-cell density for sufficient deformation
