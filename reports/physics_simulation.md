# Report 1 — C. elegans Biophysical ABM

## 1. DATASET SCHEMA

| File | Format | Shape | Separator | Content |
|------|--------|-------|-----------|---------|
| CDSample04.txt | Text | 22,945 lines | Whitespace | Cell lineage tracking: Cell, Time, Z, X, Y (816 unique cell names) |
| Sample04_Volume.csv | CSV | 150 × 804 | Comma | Rows = timepoints, Cols = cell names, Values = volume in voxels |
| Sample04_Stat.csv | CSV | 151 × 10,253 | Comma | Contact area matrix. Row 0 = cell2 partner header. Rows 1+ = timepoint data |
| Sample04_Surface.csv | CSV | 150 × 804 | Comma | Same structure as Volume; total surface area per cell per timepoint |

**Cell identity encoding**: Cell name strings (e.g. "ABa", "ABp", "EMS", "P2") as column headers.  
**Timepoint indexing**: Integer index (0-based in Volume/Surface, 1-based in Stat).  
**Units**: Volume in voxels (0.09 × 0.09 × 1.0 μm³/voxel). Contact areas in voxel-surface units.

## 2. EXTRACTED VALUES

### V₀ per cell (averaged over 4-cell stage: timepoints 0–1)

| Cell | Mean Voxels | V₀ (μm³) | R (μm) | R_c (μm) |
|------|-------------|-----------|--------|----------|
| ABa  | 343,550     | 2,782.75  | 8.72   | 6.98     |
| ABp  | 371,354     | 3,007.97  | 8.96   | 7.16     |
| EMS  | 282,828     | 2,290.91  | 8.17   | 6.54     |
| P2   | 199,470     | 1,615.71  | 7.29   | 5.83     |

### Contact areas (mean over 4-cell stage, raw voxel-surface units)

| Pair     | Mean Area |
|----------|-----------|
| ABa-ABp  | 5,776.69  |
| ABa-EMS  | 4,307.77  |
| ABp-EMS  | 4,245.28  |
| ABp-P2   | 4,297.64  |
| EMS-P2   | 2,434.78  |
| ABa-P2   | **ABSENT** |

## 3. BIOLOGICAL CONSTRAINTS

| Constraint | Result |
|------------|--------|
| V_ABa ≈ V_ABp (within 15%) | **PASS** (7.5% difference) |
| min(V_ABa, V_ABp) > V_EMS | **PASS** (2783 > 2291) |
| V_EMS > V_P2 | **PASS** (2291 > 1616) |
| ABa-P2 contact absent | **PASS** |

## 4. DEVICE

```
Device: cpu (PyTorch 2.10.0, CUDA not available)
```

## 5. TEST RESULTS

| Test | Description | Result |
|------|-------------|--------|
| TEST 1 | JKR formula: A_contact=293.89 μm² at d=14, ≈0 at d=17, decreases | **PASS** |
| TEST 2 | Shell confinement: E > 0 outside, E = 0 at center | **PASS** |
| TEST 3 | Overlap repulsion: E > 0 overlapping, E = 0 separated | **PASS** |
| TEST 4 | Volume elasticity: E ≈ 0 when V_eff = V0 | **PASS** |
| TEST 5 | Cortical flow: P2 force toward +x, AB E = 0 | **PASS** |
| TEST 6 | Inner loop convergence: converged step 139, E=-108.51, cells inside shell | **PASS** |

**ALL 6 TESTS PASS**

## 6. JKR TEST VALUES

```
Test 1 setup:
  R_i = R_j = 8.00 μm
  γ_i = γ_j = 1.0 pN/μm (AB lineage)
  w = 0.5 mJ/m²
  R_c_i + R_c_j = 12.80 μm
  R_i + R_j = 16.00 μm (gate threshold)

  A_contact (d=14 μm) = 293.89 μm²   ← adhesion-driven contact, no compression (d > R_c sum)
  A_contact (d=17 μm) = 0.000001 μm²  ← cells genuinely not touching (d > R sum)
  A_contact decreases with distance: confirmed
```

At d=14 μm the compressive force F=0 (since d > R_c_sum=12.8), but the JKR formula correctly
produces ~294 μm² of adhesion-only contact area. The gate fires because d < R_sum=16.
At d=17 μm > R_sum, the gate correctly suppresses the contact area to ≈0.

**Gate fix applied**: `sigmoid(20·(R_i + R_j - d))` instead of `sigmoid(20·(R_c_i + R_c_j - d))`.
This decouples contact detection (cell surfaces touching) from compression (cell cores overlapping).

## 7. ASSUMPTIONS

1. **4-cell stage identification**: Defined as timepoints where ABa, ABp, EMS, and P2 all have non-null volume entries (timepoints 0–1 in Volume CSV; rows 1–2 in Stat CSV).
2. **Contact area units**: Stored in raw dataset units (voxel surface area). Exact μm² conversion depends on the 3D segmentation algorithm and contact-face orientation; not applied here.
3. **Volume ordering**: ABp (3008 μm³) is slightly larger than ABa (2783 μm³) — 7.5% difference. The spec states "V_ABa ≈ V_ABp" — they are approximately equal, which is biologically consistent.
4. **Lineage assignment**: ABa/ABp → "AB", EMS → "EMS", P2 → "P" as specified.

## 8. BLOCKERS

**None.** All tests pass.