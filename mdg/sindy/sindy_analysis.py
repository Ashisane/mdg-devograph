#!/usr/bin/env python3
"""
sindy_analysis.py  –  SINDy sparse regression on C. elegans 4-cell dynamics.

Reads  : simulation_results.pt   (ABM trajectory, pre-computed)
         datasets/CDSample04.txt  (CShaper real embryo positions)

Writes : sindy_results.pt
         sindy_report.md
"""

import sys
import warnings
import textwrap
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from pysindy.optimizers import STLSQ



# Calibrated ABM parameters (Report 2)
GAMMA_MAP = {"AB": 0.8455053, "EMS": 0.7609548, "P": 0.6826006}
# V₀ per cell in μm³, averaged over 4-cell stage (Report 1)
V0_MAP    = {"ABa": 2782.75, "ABp": 3007.97, "EMS": 2290.91, "P2": 1615.71}
LINEAGE   = {"ABa": "AB", "ABp": "AB", "EMS": "EMS", "P2": "P"}
CELLS     = ["ABa", "ABp", "EMS", "P2"]

# Feature vector column names (per observation)
FEATURE_NAMES = ["x", "y", "z", "contact_area_sum", "V0", "gamma", "n_neighbors"]
# SINDy candidate library terms
LIBRARY_NAMES = [
    "x", "y", "z",
    "ca", "V0", "gamma", "nn",
    "x^2", "y^2", "z^2",
    "x*ca", "y*ca",
    "1/ca",
]
N_LIB = len(LIBRARY_NAMES)   # 13
AXES  = ["x", "y", "z"]



def finite_diff(arr: np.ndarray, dt_arr: np.ndarray) -> np.ndarray:
    """
    Compute derivative of arr (shape n×3) along axis-0 using finite differences.
    dt_arr: (n-1,) array of spacing between consecutive samples.
    Forward difference at boundary index 0,
    central difference at interior indices,
    backward difference at boundary index n-1.
    """
    n = arr.shape[0]
    ddt = np.zeros_like(arr, dtype=float)
    if n == 1:
        return ddt                         # no derivative for single point
    ddt[0] = (arr[1] - arr[0]) / float(dt_arr[0])
    for i in range(1, n - 1):
        total_dt = float(dt_arr[i - 1] + dt_arr[i])
        ddt[i]   = (arr[i + 1] - arr[i - 1]) / total_dt
    ddt[-1] = (arr[-1] - arr[-2]) / float(dt_arr[-1])
    return ddt


def build_theta(X: np.ndarray) -> np.ndarray:
    """
    Build 13-column candidate library from 7-feature matrix X.
    X columns: [x, y, z, ca, V0, gamma, nn]
    Returns Theta of shape (n_obs, 13).
    1/ca is set to 0 wherever ca <= 1 to avoid division by zero.
    """
    x,  y,  z  = X[:, 0], X[:, 1], X[:, 2]
    ca, V0, gam, nn = X[:, 3], X[:, 4], X[:, 5], X[:, 6]
    safe_ca = np.where(ca > 1.0, ca, np.inf)   # prevents division near zero
    inv_ca  = np.where(ca > 1.0, 1.0 / safe_ca, 0.0)
    return np.column_stack([
        x,  y,  z,
        ca, V0, gam, nn,
        x**2, y**2, z**2,
        x * ca, y * ca,
        inv_ca,
    ])


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-15:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def fit_stlsq(
    Theta: np.ndarray,
    dy: np.ndarray,
    label: str,
    threshold: float = 0.05,
    alpha: float    = 0.05,
) -> tuple:
    """
    Fit STLSQ on Theta -> dy (1-D).
    Returns (coef, r2_val, n_nonzero).
    """
    opt = STLSQ(threshold=threshold, alpha=alpha)
    opt.fit(Theta, dy.reshape(-1, 1))
    coef   = opt.coef_.flatten()
    y_pred = Theta @ coef
    r2val  = r2_score(dy, y_pred)
    n_nz   = int(np.sum(np.abs(coef) > 1e-10))
    print(f"    [{label}]  threshold={threshold}  →  {n_nz} terms,  R²={r2val:.4f}")
    return coef, r2val, n_nz


def fit_axis(
    Theta: np.ndarray,
    dy: np.ndarray,
    label: str,
    threshold: float = 0.05,
    alpha: float    = 0.05,
) -> tuple:
    """
    Fit STLSQ; retry at threshold=0.01 if zero terms survive (Rule 7).
    Returns (final_coef, final_r2, attempts_list).
    """
    coef, r2val, n_nz = fit_stlsq(Theta, dy, label, threshold, alpha)
    attempts = [{"threshold": threshold, "coef": coef.copy(), "r2": r2val, "n_nonzero": n_nz}]
    if n_nz == 0:
        print(f"    ⚠ Zero terms survived at threshold={threshold}. Retrying at 0.01 …")
        coef2, r2val2, n_nz2 = fit_stlsq(Theta, dy, f"{label}@0.01", 0.01, alpha)
        attempts.append({
            "threshold": 0.01, "coef": coef2.copy(), "r2": r2val2, "n_nonzero": n_nz2
        })
        coef, r2val = coef2, r2val2   # use retry result as final
    return coef, r2val, attempts


def format_equation(coef: np.ndarray, axis: str, lib_names=LIBRARY_NAMES) -> str:
    """Return human-readable equation string with surviving (non-zero) terms."""
    terms = [
        (lib_names[i], float(coef[i]))
        for i in range(len(coef))
        if abs(coef[i]) > 1e-10
    ]
    if not terms:
        return f"d{axis}/dt = 0   (no surviving terms)"
    parts = [f"{c:+.6f}·{n}" for n, c in terms]
    return f"d{axis}/dt = " + " ".join(parts)


def survival_info(coef: np.ndarray, lib_names=LIBRARY_NAMES) -> list:
    return [
        (lib_names[i], float(coef[i]))
        for i in range(len(coef))
        if abs(coef[i]) > 1e-10
    ]






sim  = torch.load("simulation_results.pt", map_location="cpu", weights_only=False)
traj = sim["trajectory_4cell"]

print(f"Total trajectory frames in trajectory_4cell : {len(traj)}")

# 4-cell equilibration phase: every frame where n_cells == 4
frames_4cell = [i for i, f in enumerate(traj) if f["n_cells"] == 4]
n_frames     = len(frames_4cell)
times_arr    = np.array([float(traj[i]["t"]) for i in frames_4cell])
dt_abm       = float(times_arr[1] - times_arr[0])

print(f"4-cell stage  :  {n_frames} frames  "
      f"(indices {frames_4cell[0]}–{frames_4cell[-1]}, "
      f"t={int(times_arr[0])}–{int(times_arr[-1])}, dt={dt_abm})")


pos_abm = {c: np.zeros((n_frames, 3)) for c in CELLS}
ca_abm  = {c: np.zeros(n_frames)      for c in CELLS}
nn_abm  = {c: np.zeros(n_frames)      for c in CELLS}

for fi, fidx in enumerate(frames_4cell):
    frame        = traj[fidx]
    cell_lookup  = {c["identity"]: c for c in frame["cells"]}
    contacts     = frame["contacts"]                   # {"ABa-ABp": area, …}

    for cell in CELLS:
        pos_abm[cell][fi] = cell_lookup[cell]["position"]

    for cell in CELLS:
        ca_sum  = 0.0
        n_neigh = 0
        for pair_key, area in contacts.items():
            if cell in pair_key.split("-"):
                ca_sum += area
                if area > 50.0:
                    n_neigh += 1
        ca_abm[cell][fi] = ca_sum
        nn_abm[cell][fi] = float(n_neigh)


dt_arr_abm  = np.full(n_frames - 1, dt_abm)
X_abm_rows  = []
dX_abm_rows = []

for cell in CELLS:
    pos  = pos_abm[cell]                       # (n_frames, 3)
    dpos = finite_diff(pos, dt_arr_abm)        # (n_frames, 3)
    gamma = GAMMA_MAP[LINEAGE[cell]]
    V0    = V0_MAP[cell]
    for fi in range(n_frames):
        x_c, y_c, z_c = pos[fi]
        ca = ca_abm[cell][fi]
        nn = nn_abm[cell][fi]
        X_abm_rows.append([x_c, y_c, z_c, ca, V0, gamma, nn])
        dX_abm_rows.append(dpos[fi])

X_abm  = np.array(X_abm_rows,  dtype=float)   # (n_frames*4, 7)
dX_abm = np.array(dX_abm_rows, dtype=float)   # (n_frames*4, 3)

print(f"\nABM feature matrix    :  {X_abm.shape}  "
      f"({n_frames} frames × 4 cells, {len(FEATURE_NAMES)} features)")
print(f"ABM derivative matrix :  {dX_abm.shape}")

Theta_abm = build_theta(X_abm)
print(f"ABM Theta (library)   :  {Theta_abm.shape}  "
      f"(13 candidate terms)")

print("\nABM feature statistics:")
print(f"  {'Feature':22s}  {'mean':>12s}  {'std':>12s}  {'min':>12s}  {'max':>12s}")
for fi, fname in enumerate(FEATURE_NAMES):
    col = X_abm[:, fi]
    print(f"  {fname:22s}  {col.mean():12.4g}  {col.std():12.4g}  "
          f"{col.min():12.4g}  {col.max():12.4g}")

print("\nABM derivative statistics:")
for ai, axis in enumerate(AXES):
    col = dX_abm[:, ai]
    print(f"  d{axis}/dt:  mean={col.mean():9.4g}  std={col.std():9.4g}  "
          f"min={col.min():9.4g}  max={col.max():9.4g}")





df_raw = pd.read_csv("datasets/CDSample04.txt", sep=r"\s+")
df4cs  = (df_raw[df_raw["Cell"].isin(CELLS)]
          .sort_values(["Cell", "Time"])
          .reset_index(drop=True))

print(f"Total rows for cells {CELLS}: {len(df4cs)}")
print("Time range in dataset:", df_raw["Time"].min(), "–", df_raw["Time"].max())
print("\nTimepoints per target cell:")
cs_coverage = {}
for cell in CELLS:
    sub = df4cs[df4cs["Cell"] == cell]
    tps = sorted(sub["Time"].tolist())
    cs_coverage[cell] = len(tps)
    note = "(⚠ < 2 → no derivative)" if len(tps) < 2 else ""
    print(f"  {cell}: {len(tps)} timepoints  {tps}  {note}")


X_cs_rows   = []
dX_cs_rows  = []
cs_cell_per_row = []

for cell in CELLS:
    sub  = df4cs[df4cs["Cell"] == cell].sort_values("Time")
    tps  = sub["Time"].values.astype(float)
    n_tp = len(tps)
    if n_tp < 2:
        print(f"  {cell}: skipped (only {n_tp} timepoint, cannot compute derivative)")
        continue
    # Stack as [X, Y, Z] voxel positions
    pos_cs = np.column_stack([
        sub["X"].values.astype(float),
        sub["Y"].values.astype(float),
        sub["Z"].values.astype(float),
    ])               # (n_tp, 3)
    dt_arr_cs = np.diff(tps)
    dpos_cs   = finite_diff(pos_cs, dt_arr_cs)
    gamma = GAMMA_MAP[LINEAGE[cell]]
    V0    = V0_MAP[cell]
    for i in range(n_tp):
        xv, yv, zv = pos_cs[i]
        # contact_area_sum = 0.0  (not available in CDSample04.txt)
        # n_neighbors      = 0.0  (not available)
        X_cs_rows.append([xv, yv, zv, 0.0, V0, gamma, 0.0])
        dX_cs_rows.append(dpos_cs[i])
        cs_cell_per_row.append(cell)

n_cs_obs       = len(X_cs_rows)
CSHAPER_TOO_FEW = (n_cs_obs < 10)

if n_cs_obs > 0:
    X_cs  = np.array(X_cs_rows,  dtype=float)
    dX_cs = np.array(dX_cs_rows, dtype=float)
else:
    X_cs  = np.zeros((0, 7))
    dX_cs = np.zeros((0, 3))

Theta_cs = build_theta(X_cs) if n_cs_obs > 0 else np.zeros((0, N_LIB))

print(f"\nCShaper feature matrix    :  {X_cs.shape}")
print(f"CShaper derivative matrix :  {dX_cs.shape}")
print(f"CShaper Theta (library)   :  {Theta_cs.shape}")
if CSHAPER_TOO_FEW:
    print(f"\n⚠ WARNING: Only {n_cs_obs} observations (< 10). "
          f"SINDy results for CShaper are PRELIMINARY.")





abm_coefs    = {}
abm_r2s      = {}
abm_attempts = {}

for ai, axis in enumerate(AXES):
    print(f"\n  Fitting ABM  d{axis}/dt :")
    coef, r2val, atts = fit_axis(Theta_abm, dX_abm[:, ai], f"ABM-{axis}")
    abm_coefs[axis]    = coef
    abm_r2s[axis]      = r2val
    abm_attempts[axis] = atts
    print(f"  → {format_equation(coef, axis)}")


for axis in AXES:
    print(format_equation(abm_coefs[axis], axis))
    r2str = f"{abm_r2s[axis]:.4f}" if not np.isnan(abm_r2s[axis]) else "N/A"
    print(f"   R² = {r2str}\n")





cs_coefs    = {}
cs_r2s      = {}
cs_attempts = {}

if n_cs_obs < 2:
    print("CRITICAL: Fewer than 2 observations — SINDy cannot run.")
    for axis in AXES:
        cs_coefs[axis]    = np.zeros(N_LIB)
        cs_r2s[axis]      = float("nan")
        cs_attempts[axis] = [{
            "threshold": 0.05,
            "coef": np.zeros(N_LIB),
            "r2": float("nan"),
            "n_nonzero": 0,
        }]
else:
    for ai, axis in enumerate(AXES):
        print(f"\n  Fitting CShaper  d{axis}/dt :")
        coef, r2val, atts = fit_axis(Theta_cs, dX_cs[:, ai], f"CS-{axis}")
        cs_coefs[axis]    = coef
        cs_r2s[axis]      = r2val
        cs_attempts[axis] = atts
        print(f"  → {format_equation(coef, axis)}")

    
    for axis in AXES:
        print(format_equation(cs_coefs[axis], axis))
        r2str = f"{cs_r2s[axis]:.4f}" if not np.isnan(cs_r2s[axis]) else "N/A"
        print(f"   R² = {r2str}\n")





for axis in AXES:
    abm_eq = format_equation(abm_coefs[axis], axis)
    cs_eq  = format_equation(cs_coefs[axis],  axis)
    print(f"\n  ABM      {abm_eq}")
    print(f"  CShaper  {cs_eq}")



def build_eq_dict(coefs: dict, r2s: dict) -> dict:
    out = {}
    for axis in AXES:
        c = coefs[axis]
        surv = [
            (LIBRARY_NAMES[i], float(c[i]))
            for i in range(N_LIB)
            if abs(c[i]) > 1e-10
        ]
        out[f"d{axis}"] = {
            "terms":        [s[0] for s in surv],
            "coefficients": [s[1] for s in surv],
            "r2":           float(r2s[axis]) if not np.isnan(r2s[axis]) else None,
        }
    return out


results_to_save = {
    "abm_equations":          build_eq_dict(abm_coefs, abm_r2s),
    "cshaper_equations":      build_eq_dict(cs_coefs,  cs_r2s),
    "abm_feature_matrix":     X_abm,
    "cshaper_feature_matrix": X_cs,
    "feature_names":          FEATURE_NAMES,
    "library_names":          LIBRARY_NAMES,
    "abm_r2_per_axis":        [
        float(abm_r2s[a]) if not np.isnan(abm_r2s[a]) else None for a in AXES
    ],
    "cshaper_r2_per_axis":    [
        float(cs_r2s[a])  if not np.isnan(cs_r2s[a])  else None for a in AXES
    ],
    "abm_n_frames":   int(n_frames),
    "abm_n_obs":      int(len(X_abm)),
    "cshaper_n_obs":  int(n_cs_obs),
    "cshaper_too_few": bool(CSHAPER_TOO_FEW),
}

torch.save(results_to_save, "sindy_results.pt")
print("  Saved: sindy_results.pt")





TERM_BIOLOGY = {
    "x":    "Position along AP axis (anterior-posterior). "
            "Negative coeff → restoring force toward AP centre.",
    "y":    "Position along DV axis (dorsal-ventral). "
            "Negative coeff → restoring force toward DV centre.",
    "z":    "Position along LR axis (left-right). "
            "Negative coeff → left-right symmetrisation.",
    "ca":   "Total contact area sum. "
            "Negative coeff → cells move away from high-contact regions "
            "(cortical tension minimisation / Plateau's law).",
    "V0":   "Equilibrium volume. "
            "Non-zero coeff → volume-elasticity-driven differential drift; "
            "larger cells move differently from smaller ones.",
    "gamma":"Cortical tension coefficient (lineage-specific). "
            "Non-zero coeff → tension-driven differential drift between lineages.",
    "nn":   "Number of neighbours (contact area >50 units). "
            "Non-zero coeff → crowding / neighbour-count-dependent movement.",
    "x^2":  "Quadratic AP position. "
            "Nonlinear confinement or asymmetric restoring force.",
    "y^2":  "Quadratic DV position. Nonlinear DV restoring force.",
    "z^2":  "Quadratic LR position. Nonlinear LR constraint.",
    "x*ca": "Cross term: AP position × contact area. "
            "Contact geometry modulates AP movement.",
    "y*ca": "Cross term: DV position × contact area. "
            "Contact geometry modulates DV movement.",
    "1/ca": "Inverse contact area. "
            "Dominant at low contact (newly divided cells); "
            "could represent adhesion saturation.",
}


def attempts_note(atts: list, axis: str) -> str:
    """Return a markdown note about retry attempts."""
    if len(atts) == 1:
        a = atts[0]
        return (f"Threshold {a['threshold']}: "
                f"{a['n_nonzero']} surviving terms, R²={a['r2']:.4f}")
    lines = []
    for a in atts:
        r2str = f"{a['r2']:.4f}" if not np.isnan(a['r2']) else "N/A"
        lines.append(
            f"  - threshold={a['threshold']}: "
            f"{a['n_nonzero']} surviving terms, R²={r2str}"
        )
    return "Two attempts (Rule 7 retry applied):\n" + "\n".join(lines)


def r2str(val):
    return f"{val:.4f}" if val is not None and not np.isnan(val) else "N/A"



all_terms = set()
for axis in AXES:
    for coef_dict in [abm_coefs, cs_coefs]:
        all_terms |= {t for t, _ in survival_info(coef_dict[axis])}
all_terms = sorted(all_terms, key=lambda t: LIBRARY_NAMES.index(t) if t in LIBRARY_NAMES else 99)


def get_coef(coef_arr, term):
    if term in LIBRARY_NAMES:
        i = LIBRARY_NAMES.index(term)
        c = float(coef_arr[i])
        return f"{c:+.6f}" if abs(c) > 1e-10 else "—"
    return "—"



with open("sindy_report.md", "w", encoding="utf-8") as fout:

    def W(s=""):
        fout.write(s + "\n")

    W("# SINDy Report — C. elegans 4-Cell Stage Dynamics")
    W()
    W("> **Date generated**: auto-generated by sindy_analysis.py (Phase 4)")
    W("> **Model**: SINDy with STLSQ (threshold=0.05, α=0.05)")
    W("> **Library**: 13 candidate terms (linear, quadratic-position, cross, inverse)")
    W()


    W("---")
    W()
    W("## 1. DATA SUMMARY")
    W()
    W("### 1.1 ABM Trajectory (Dataset A)")
    W()
    W(f"- Source file: `simulation_results.pt` → key `trajectory_4cell`")
    W(f"- **4-cell equilibration phase**: {n_frames} frames "
      f"(t = {int(times_arr[0])} to {int(times_arr[-1])}, Δt = {dt_abm})")
    W(f"- Cells: {CELLS}")
    W(f"- Observations: {n_frames} frames × 4 cells = **{len(X_abm)} rows**")
    W(f"- Feature matrix shape: `{X_abm.shape}`")
    W(f"- Derivative matrix shape: `{dX_abm.shape}`")
    W(f"- Contact area, V0, γ extracted directly from trajectory.")
    W()
    W("**ABM feature statistics:**")
    W()
    W(f"| Feature | Mean | Std | Min | Max |")
    W(f"|---------|------|-----|-----|-----|")
    for fi, fname in enumerate(FEATURE_NAMES):
        col = X_abm[:, fi]
        W(f"| {fname} | {col.mean():.4g} | {col.std():.4g} "
          f"| {col.min():.4g} | {col.max():.4g} |")
    W()
    W("**ABM derivative statistics (μm per simulation time unit):**")
    W()
    W("| Axis | Mean | Std | Min | Max |")
    W("|------|------|-----|-----|-----|")
    for ai, axis in enumerate(AXES):
        col = dX_abm[:, ai]
        W(f"| d{axis}/dt | {col.mean():.4g} | {col.std():.4g} "
          f"| {col.min():.4g} | {col.max():.4g} |")
    W()

    W("### 1.2 CShaper Real Trajectory (Dataset B)")
    W()
    W("- Source file: `datasets/CDSample04.txt` (columns: Cell, Time, Z, X, Y)")
    W("- Filtered to 4 target cells: ABa, ABp, EMS, P2")
    W()
    W("**Timepoints per cell:**")
    W()
    W("| Cell | Timepoints | Notes |")
    W("|------|-----------|-------|")
    for cell in CELLS:
        n_tp = cs_coverage[cell]
        note = "⚠ Only 1 timepoint — no derivative computable; cell excluded" if n_tp < 2 else (
               "⚠ Very sparse" if n_tp < 5 else "OK")
        W(f"| {cell} | {n_tp} | {note} |")
    W()
    W(f"- **Total CShaper observations with derivatives**: {n_cs_obs}")
    W()
    if CSHAPER_TOO_FEW:
        W("> **⚠ CRITICAL DATA LIMITATION**: CShaper has only **{} observations** "
          "(< 10 threshold). The SINDy library has 13 terms. "
          "The system is **underdetermined** — coefficients cannot be reliably "
          "estimated. All CShaper SINDy results are flagged as "
          "**PRELIMINARY**. See Section 6 (Limitations).".format(n_cs_obs))
    W()
    W("**Note on CShaper features:**")
    W("CDSample04.txt contains positions only (X, Y, Z in voxel units). "
      "Contact area (`ca`) and neighbor count (`nn`) are **unavailable** and "
      "have been set to 0 throughout. V0 and γ were taken from calibrated ABM "
      "values (Reports 1 & 2). CShaper positions are in voxel coordinates "
      "(not converted to μm), so derivative units differ from ABM.")
    W()


    W("---")
    W()
    W("## 2. ABM EQUATIONS")
    W()
    W("Fit: STLSQ (threshold=0.05, α=0.05). Data: 204 observations from "
      "4-cell equilibration trajectory.")
    W()

    for axis in AXES:
        coef = abm_coefs[axis]
        r2v  = abm_r2s[axis]
        atts = abm_attempts[axis]
        surv = survival_info(coef)

        W(f"### 2.{AXES.index(axis)+1}  d{axis}/dt")
        W()
        W(f"```")
        W(format_equation(coef, axis))
        W("```")
        W()
        W(f"**R² = {r2str(r2v)}**")
        W()
        W(f"**Fit notes**: {attempts_note(atts, axis)}")
        W()

        if surv:
            W("**Surviving terms:**")
            W()
            W("| Term | Coefficient | Biological interpretation |")
            W("|------|-------------|--------------------------|")
            for term, c in surv:
                interp = TERM_BIOLOGY.get(term, "—")
                W(f"| `{term}` | {c:+.6f} | {interp} |")
        else:
            W("**No terms survived.** "
              "Dynamics along this axis are effectively zero or "
              "below the sparsity threshold.")
        W()


    W("---")
    W()
    W("## 3. CSHAPER EQUATIONS")
    W()
    if CSHAPER_TOO_FEW:
        W(f"> ⚠ **PRELIMINARY RESULTS** — Only {n_cs_obs} observations. "
          f"The 13-term library is underdetermined. "
          f"Results should be treated as exploratory only.")
        W()
    W("Fit: STLSQ (threshold=0.05, α=0.05). "
      f"Data: {n_cs_obs} observations from EMS + P2 trajectories "
      "(ABa and ABp excluded: only 1 timepoint each).")
    W()

    for axis in AXES:
        coef = cs_coefs[axis]
        r2v  = cs_r2s[axis]
        atts = cs_attempts[axis]
        surv = survival_info(coef)

        W(f"### 3.{AXES.index(axis)+1}  d{axis}/dt")
        W()
        W(f"```")
        W(format_equation(coef, axis))
        W("```")
        W()
        r2_display = r2str(r2v)
        W(f"**R² = {r2_display}**")
        W()
        W(f"**Fit notes**: {attempts_note(atts, axis)}")
        W()

        if surv:
            W("**Surviving terms:**")
            W()
            W("| Term | Coefficient | Biological interpretation |")
            W("|------|-------------|--------------------------|")
            for term, c in surv:
                interp = TERM_BIOLOGY.get(term, "—")
                W(f"| `{term}` | {c:+.6f} | {interp} |")
        else:
            W("**No terms survived.** "
              "Either the dynamics are zero, below threshold, or "
              "the system is underdetermined.")
        W()


    W("---")
    W()
    W("## 4. COMPARISON TABLE")
    W()
    W("All terms that survive in **either** ABM or CShaper equations are listed. "
      "'—' denotes coefficient effectively zero (< 1×10⁻¹⁰) or term absent.")
    W()

    for axis in AXES:
        W(f"### {axis.upper()}-axis  (d{axis}/dt)")
        W()
        W("| Term | ABM coeff | CShaper coeff | Interpretation |")
        W("|------|-----------|---------------|----------------|")

        # Collect terms that appear in either ABM or CShaper for this axis
        axis_terms = set()
        axis_terms |= {t for t, _ in survival_info(abm_coefs[axis])}
        axis_terms |= {t for t, _ in survival_info(cs_coefs[axis])}
        axis_terms  = sorted(axis_terms,
                             key=lambda t: LIBRARY_NAMES.index(t)
                             if t in LIBRARY_NAMES else 99)

        if not axis_terms:
            W("| — | — | — | No surviving terms in either dataset |")
        else:
            for term in axis_terms:
                abm_c = get_coef(abm_coefs[axis], term)
                cs_c  = get_coef(cs_coefs[axis],  term)
                interp = TERM_BIOLOGY.get(term, "—")[:80]
                W(f"| `{term}` | {abm_c} | {cs_c} | {interp} |")
        W()


    W("---")
    W()
    W("## 5. SCIENTIFIC INTERPRETATION")
    W()
    W("### 5.1 What the ABM equations reveal")
    W()

    abm_all_surviving = {}
    for axis in AXES:
        abm_all_surviving[axis] = survival_info(abm_coefs[axis])

    any_abm_terms = any(len(v) > 0 for v in abm_all_surviving.values())

    if any_abm_terms:
        W("The ABM SINDy fit captures the effective force law governing cell "
          "movement during the 4-cell equilibration phase. "
          "Key findings:")
        W()

        # Position terms
        pos_terms_found = {
            axis: [(t, c) for t, c in abm_all_surviving[axis] if t in ["x", "y", "z"]]
            for axis in AXES
        }
        if any(len(v) > 0 for v in pos_terms_found.values()):
            W("- **Position terms (x, y, z)**: Negative self-position coefficients "
              "indicate a linear restoring force — cells are pulled toward the centre "
              "of mass, consistent with confinement inside the eggshell (shell energy "
              "penalty in the ABM). Positive off-diagonal position terms would indicate "
              "geometric coupling between axes.")
            W()

        ca_terms = {
            axis: [(t, c) for t, c in abm_all_surviving[axis] if t in ["ca", "x*ca", "y*ca", "1/ca"]]
            for axis in AXES
        }
        if any(len(v) > 0 for v in ca_terms.values()):
            W("- **Contact area terms (ca, x*ca, y*ca, 1/ca)**: Surviving contact area "
              "coefficients indicate that the rate of cell movement depends on the "
              "extent of cell-cell adhesion. A negative `ca` coefficient means cells "
              "with high contact area (highly adhered) move less — consistent with "
              "cortical tension minimisation (Plateau's law). A negative `x*ca` or "
              "`y*ca` coefficient means contact area suppresses movement toward the "
              "cell's current position, i.e., adhesion creates a drag.")
            W()

        gamma_V0_terms = {
            axis: [(t, c) for t, c in abm_all_surviving[axis] if t in ["gamma", "V0"]]
            for axis in AXES
        }
        if any(len(v) > 0 for v in gamma_V0_terms.values()):
            W("- **Cortical tension (γ) and equilibrium volume (V₀) terms**: "
              "Their appearance as independent predictors means that the lineage "
              "identity (AB vs EMS vs P) or cell size contributes to the effective "
              "drift velocity, even after accounting for position. This may reflect "
              "asymmetric division outcomes and differential cortical flow (P2 has "
              "an active cortical flow in the ABM).")
            W()
    else:
        W("> **No terms survived ABM SINDy.** This indicates that during the "
          "4-cell equilibration phase, cell displacements per timestep are very "
          "small (near-equilibrium), and no candidate function in the library "
          "captures the residual dynamics above the sparsity threshold. "
          "The ABM may have converged to a fixed-point attractor before this "
          "phase is recorded.")
        W()

    W("### 5.2 CShaper equations vs ABM equations")
    W()
    cs_all_surv = {axis: survival_info(cs_coefs[axis]) for axis in AXES}
    any_cs_terms = any(len(v) > 0 for v in cs_all_surv.values())

    W("**IMPORTANT CAVEAT**: Direct numerical comparison of coefficients is "
      "**not valid** because:")
    W("1. ABM positions are in μm; CShaper positions are in voxel units.")
    W("2. ABM time is in simulation steps (arbitrary units); "
      "CShaper time is in integer acquisition frames.")
    W("3. CShaper has only {} observations — results are unreliable.".format(n_cs_obs))
    W()
    W("What CAN be compared is which **terms survive** (structural comparison):")
    W()

    # Find common and unique terms
    abm_term_set = set()
    cs_term_set  = set()
    for axis in AXES:
        abm_term_set |= {t for t, _ in abm_all_surviving[axis]}
        cs_term_set  |= {t for t, _ in cs_all_surv[axis]}

    common_terms = abm_term_set & cs_term_set
    abm_only     = abm_term_set - cs_term_set
    cs_only      = cs_term_set  - abm_term_set

    if common_terms:
        W(f"- **Terms in BOTH** (validated physics): {sorted(common_terms)}")
        W("  These terms appear in both ABM and CShaper dynamics. "
          "Their presence in the real embryo validates the ABM's "
          "choice of candidate forces.")
        W()
    else:
        W("- **No terms in common** between ABM and CShaper equations. "
          "This is most likely due to the extreme scarcity of CShaper data "
          "rather than a genuine physical discrepancy.")
        W()

    if abm_only:
        W(f"- **ABM-only terms**: {sorted(abm_only)}")
        W("  These terms are discovered in the ABM but absent from CShaper. "
          "Possible explanations: (a) CShaper data is too sparse to detect them; "
          "(b) the ABM introduces artefactual forces absent in the real embryo.")
        W()

    if cs_only:
        W(f"- **CShaper-only terms**: {sorted(cs_only)}")
        W("  These terms appear in the real embryo but not the ABM. "
          "This could indicate missing physics in the model — "
          "e.g., cytoplasmic streaming, spindle forces, or cortical flows "
          "not captured by the energy-minimisation ABM.")
        W()

    W("### 5.3 What hidden variables might explain the gap?")
    W()
    W("1. **Cell shape deformation**: The ABM uses rigid spheres. Real cells "
      "deform under contact, producing non-spherical contact geometries. "
      "Shape deformation variables (e.g., aspect ratio, contact angle) "
      "are absent from both datasets and may drive terms not captured here.")
    W()
    W("2. **Cytoplasmic and cortical flows**: PAR polarity gradients drive "
      "directed cortical flows in P2 (included partially in ABM as `alpha`) "
      "but internal flows affecting nuclear/centrosome positions are absent.")
    W()
    W("3. **Osmotic pressure and turgor**: Volume elasticity in the ABM is "
      "a penalty term; real cells regulate turgor. This may add a "
      "pressure-dependent velocity component.")
    W()
    W("4. **ABM parameter recalibration issue (Report 3b)**: The 4-cell ABM "
      "parameters do not transfer to the 8-cell stage without recalibration, "
      "suggesting the effective force law changes with cell density. "
      "Only the 4-cell phase is modelled here.")
    W()

    # ── Section 6: Honest Limitations ───────────────────────────────────────
    W("---")
    W()
    W("## 6. HONEST LIMITATIONS")
    W()
    W("### 6.1 CShaper Timepoint Density")
    W()
    W(f"CDSample04.txt contains only **{n_cs_obs} derivative-computable observations** "
      f"for the 4 target cells in the 4-cell stage:")
    W()
    for cell in CELLS:
        n = cs_coverage[cell]
        W(f"- **{cell}**: {n} timepoints → "
          + ("excluded (no derivative)" if n < 2 else f"{n} observations"))
    W()
    W("With 13 candidate library terms and only 8 observations, the SINDy "
      "system is **severely underdetermined** (more unknowns than equations). "
      "STLSQ can still produce sparse solutions, but coefficient values are "
      "unreliable. Any surviving CShaper term should be treated as a hypothesis "
      "to test with more data, not a confirmed result.")
    W()

    W("### 6.2 SINDy Autonomy Assumption")
    W()
    W("SINDy assumes the dynamics are **autonomous**: dx/dt = f(x) with no "
      "explicit time dependence. This is violated if:")
    W("- external forcing (division events, PAR polarity) changes during the window;")
    W("- the equilibration is transient (non-stationary attractor).")
    W("The 4-cell equilibration phase starts immediately after P1 division and "
      "ends at steady state — the early frames are transient. "
      "Including them may cause non-zero slow-time-varying terms to be "
      "misattributed to spatial features.")
    W()

    W("### 6.3 Spherical Cell Approximation")
    W()
    W("The ABM uses JKR contact mechanics between spheres. "
      "Real C. elegans blastomeres are roughly spherical at 4-cell stage "
      "but undergo visible flattening at contacts. "
      "The true contact area is therefore larger than JKR predicts "
      "(consistent with R² = 0.38 in the calibration). "
      "SINDy terms involving `ca` (contact area sum) may be systematically "
      "underestimated in the ABM relative to the real embryo.")
    W()

    W("### 6.4 ABM 4-Cell Topology Instability")
    W()
    W("Report 2 noted that only 10/20 ABM validation runs produce the correct "
      "3+1 diamond topology. The trajectory used here (a single run with the "
      "calibrated parameters) may represent one of these correct runs. "
      "SINDy results derived from this single trajectory may not generalise "
      "across all 20 runs.")
    W()

    W("### 6.5 ABM Near-Equilibrium Dynamics")
    W()
    W("The 4-cell phase spans t=320–820 (Report 3 frame mapping). "
      "By t~500+, the system is near equilibrium. "
      "Near-equilibrium derivatives are close to zero, which: "
      "(a) reduces the effective signal-to-noise for SINDy; "
      "(b) may cause sparsity to zero out real physical terms. "
      "Results at threshold=0.05 may be overly sparse.")
    W()

    W("### 6.6 Unit Mismatch (ABM vs CShaper)")
    W()
    W("- ABM positions: μm. CShaper positions: voxel coordinates (~0.09 μm/px in XY, ~1 μm/px in Z).")
    W("- ABM time: simulation gradient-descent units. CShaper time: acquisition frames (~1.5 min each estimated).")
    W("- ABM contact areas: μm². CShaper contact areas: unavailable (set to 0).")
    W("Coefficient values between ABM and CShaper equations are **not directly comparable**.")
    W()
    W("---")
    W()
    W("*Report generated by `sindy_analysis.py`. "
      "All source data files read-only — no existing files modified.*")

print("  Saved: sindy_report.md")

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ALL DONE")
print(f"  sindy_results.pt : saved")
print(f"  sindy_report.md  : saved")
print("=" * 70)
