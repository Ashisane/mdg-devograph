"""
physics.py — Biophysical simulation engine for C. elegans early embryogenesis.

Energy terms:
  1. Eggshell confinement (ellipsoidal)
  2. Volume elasticity / shape energy (deformable ellipsoid)
  3. Overlap repulsion (hard-core, ellipsoid-aware)
  4. Adhesion (JKR contact mechanics, ellipsoid-aware)
  5. Cortical flow (P-lineage posterior bias)

CellAgent supports both rigid sphere (axes=None) and deformable ellipsoid modes.
All pairwise functions fall back to spherical behavior when axes is None,
preserving backward compatibility with the original 4-cell tests.
"""

import math
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
print(f"[physics.py] Using device: {DEVICE}")

SHELL_A = 25.0   # half-axis AP (X), μm
SHELL_B = 15.0   # half-axis Y, μm
SHELL_C = 15.0   # half-axis Z, μm

K_SHELL   = 200.0
K_VOL     = 0.01
K_REP     = 50.0
K_DEFORM  = 0.5   # volume conservation — lowered to allow asymmetric contact deformation
K_ELASTIC = 0.5   # resistance to deviate from sphere


class CellAgent:
    """
    Cell agent supporting both rigid sphere and deformable ellipsoid modes.

    Rigid sphere (default after set_position):
        axes = [R, R, R],  quaternion = [1, 0, 0, 0]  (both require_grad=True)

    Deformable ellipsoid DOFs (all require_grad=True):
        position   : Tensor(3,) — center of mass in μm
        axes       : Tensor(3,) — semi-axes (a, b, c), constrained > 0
        quaternion : Tensor(4,) — unit quaternion for orientation
    """

    _LINEAGE_MAP = {
        # 2-cell
        "AB": "AB", "P1": "P",
        # 4-cell
        "ABa": "AB", "ABp": "AB", "EMS": "EMS", "P2": "P", "P": "P",
        # 8-cell
        "ABar": "AB", "ABal": "AB", "ABpr": "AB", "ABpl": "AB",
        "MS": "EMS", "E": "EMS",
        "C": "P", "P3": "P",
    }

    def __init__(self, identity: str, V0: float):
        self.identity = identity
        self.lineage  = self._LINEAGE_MAP.get(identity, identity)
        self.V0       = V0
        self.R        = (3.0 * V0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        self.R_c      = 0.8 * self.R
        self.position  = None
        self.axes      = None
        self.quaternion = None

    def set_position(self, x: float, y: float, z: float):
        """Set position and initialise shape DOFs (sphere at rest)."""
        self.position = torch.tensor(
            [x, y, z], dtype=DTYPE, device=DEVICE, requires_grad=True
        )
        self.axes = torch.tensor(
            [self.R, self.R, self.R], dtype=DTYPE, device=DEVICE, requires_grad=True
        )
        self.quaternion = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=DTYPE, device=DEVICE, requires_grad=True
        )

    def __repr__(self):
        pos_str = (
            f"({self.position[0].item():.2f}, "
            f"{self.position[1].item():.2f}, "
            f"{self.position[2].item():.2f})"
            if self.position is not None else "None"
        )
        return (f"CellAgent({self.identity}, lineage={self.lineage}, "
                f"R={self.R:.2f}, pos={pos_str})")


# ---------------------------------------------------------------------------
# Geometry helpers for deformable ellipsoids
# ---------------------------------------------------------------------------

def quaternion_to_rotation(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion q = (qw, qx, qy, qz) to a 3×3 rotation matrix.
    Differentiable — autograd flows through this cleanly.
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    R = torch.stack([
        torch.stack([1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)]),
        torch.stack([2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)]),
        torch.stack([2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)]),
    ])  # (3, 3)
    return R


def effective_radius(cell: "CellAgent", direction: torch.Tensor) -> torch.Tensor:
    """
    Effective radius of the ellipsoid in a given unit direction (world frame).

    r_eff = 1 / sqrt( (d_local[0]/a)² + (d_local[1]/b)² + (d_local[2]/c)² )

    For a sphere (a=b=c=R) this returns exactly R for any direction.
    """
    Q   = quaternion_to_rotation(cell.quaternion)   # (3, 3)
    abc = cell.axes                                  # (3,)

    # Project direction into ellipsoid's local frame
    d_local = Q.T @ direction                        # Q^T · d_hat

    inv_r_sq = (d_local[0] / abc[0])**2 + \
               (d_local[1] / abc[1])**2 + \
               (d_local[2] / abc[2])**2
    return 1.0 / torch.sqrt(inv_r_sq + 1e-12)


def ellipsoid_contact_distance(
    cell_i: "CellAgent", cell_j: "CellAgent"
) -> tuple:
    """
    Compute effective radii of two ellipsoids toward each other.

    Returns (r_eff_i, r_eff_j, d_vec) where:
        r_eff_i  — effective radius of i toward j
        r_eff_j  — effective radius of j toward i
        d_vec    — displacement vector j.pos - i.pos
    """
    d_vec = cell_j.position - cell_i.position
    d_norm = torch.norm(d_vec)
    d_safe = torch.clamp(d_norm, min=1e-10)
    d_hat  = d_vec / d_safe

    r_i = effective_radius(cell_i,  d_hat)
    r_j = effective_radius(cell_j, -d_hat)
    return r_i, r_j, d_vec


# ---------------------------------------------------------------------------
# Shape energy (new — only active when cell has ellipsoid DOFs)
# ---------------------------------------------------------------------------

def shape_energy(cell: "CellAgent") -> torch.Tensor:
    """
    Two-part deformation penalty:
      E_vol_deform = K_deform * (a·b·c - R³)²      — volume conservation
      E_elastic    = K_elastic * Σ(axis_k - R)²     — resistance to deformation

    Returns 0 (scalar tensor) if cell has no axes.
    """
    if cell.axes is None:
        return torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

    a, b, c = cell.axes[0], cell.axes[1], cell.axes[2]
    R3      = cell.R ** 3

    E_vol  = K_DEFORM  * (a * b * c - R3) ** 2
    E_elas = K_ELASTIC * ((a - cell.R)**2 + (b - cell.R)**2 + (c - cell.R)**2)
    return E_vol + E_elas


# ---------------------------------------------------------------------------
# Energy Term 1 — Eggshell Confinement
# ---------------------------------------------------------------------------

def shell_energy(cell: "CellAgent") -> torch.Tensor:
    """
    Ellipsoidal eggshell confinement. Soft penalty outside the shell, zero inside.

    E_shell = K_shell · clamp(f_eff - 1, min=0)²
    f_eff   = (x/a_eff)² + (y/b_eff)² + (z/c_eff)²,  a_eff = SHELL_A - R_c
    """
    pos   = cell.position
    a_eff = SHELL_A - cell.R_c
    b_eff = SHELL_B - cell.R_c
    c_eff = SHELL_C - cell.R_c

    f_eff     = (pos[0]/a_eff)**2 + (pos[1]/b_eff)**2 + (pos[2]/c_eff)**2
    violation = torch.clamp(f_eff - 1.0, min=0.0)
    return K_SHELL * violation ** 2


# ---------------------------------------------------------------------------
# Energy Term 2 — Volume Elasticity
# ---------------------------------------------------------------------------

def _spherical_cap_volume(cell_i: "CellAgent", cell_j: "CellAgent") -> torch.Tensor:
    """
    Volume of the cap that cell_j carves from cell_i, using effective radii
    when ellipsoid DOFs are present (falls back to scalar R otherwise).

      h_i = clamp(r_i - (d² + r_i² - r_j²) / (2d), min=0)
      V   = π h_i² (r_i - h_i/3)
    """
    if cell_i.axes is not None and cell_j.axes is not None:
        r_i, r_j, d_vec = ellipsoid_contact_distance(cell_i, cell_j)
    else:
        d_vec = cell_j.position - cell_i.position
        r_i   = torch.tensor(cell_i.R, dtype=DTYPE, device=DEVICE)
        r_j   = torch.tensor(cell_j.R, dtype=DTYPE, device=DEVICE)

    d      = torch.norm(d_vec)
    d_safe = torch.clamp(d, min=1e-10)

    h_i = torch.clamp(
        r_i - (d_safe**2 + r_i**2 - r_j**2) / (2.0 * d_safe),
        min=0.0
    )
    return math.pi * h_i**2 * (r_i - h_i / 3.0)


def volume_energy(cell: "CellAgent", all_cells: list) -> torch.Tensor:
    """
    Volume elasticity + shape penalty.

    E = K_vol · (V_eff - V0)²  +  shape_energy(cell)

    V_eff = (4/3)πR³ - Σ V_cap(cell, j)
    The shape_energy term is zero for rigid spheres.
    """
    V_full = (4.0 / 3.0) * math.pi * cell.R ** 3
    V_caps = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

    for other in all_cells:
        if other is not cell:
            V_caps = V_caps + _spherical_cap_volume(cell, other)

    V_eff = V_full - V_caps
    return K_VOL * (V_eff - cell.V0)**2 + shape_energy(cell)


# ---------------------------------------------------------------------------
# Energy Term 3 — Overlap Repulsion
# ---------------------------------------------------------------------------

def overlap_repulsion(cell_i: "CellAgent", cell_j: "CellAgent") -> torch.Tensor:
    """
    Hard-core overlap repulsion.

    Uses ellipsoid effective radii when available; falls back to R_c otherwise.
    E_rep = K_rep · clamp(R_c_contact - d, min=0)²
    """
    if cell_i.axes is not None and cell_j.axes is not None:
        r_i, r_j, d_vec = ellipsoid_contact_distance(cell_i, cell_j)
        d          = torch.norm(d_vec)
        R_c_sum    = 0.8 * (r_i + r_j)
    else:
        d       = torch.norm(cell_i.position - cell_j.position)
        R_c_sum = cell_i.R_c + cell_j.R_c

    overlap = torch.clamp(R_c_sum - d, min=0.0)
    return K_REP * overlap ** 2


# ---------------------------------------------------------------------------
# Energy Term 4 — Adhesion (JKR contact mechanics)
# ---------------------------------------------------------------------------

def _get_gamma(cell: "CellAgent", params: dict) -> float:
    """Cortical tension γ by lineage."""
    lm = {"AB": params['gamma_AB'], "EMS": params['gamma_EMS'], "P": params['gamma_P']}
    if cell.lineage not in lm:
        raise ValueError(f"Unknown lineage: {cell.lineage}")
    return lm[cell.lineage]


def jkr_contact_area(
    cell_i: "CellAgent",
    cell_j: "CellAgent",
    w: float,
    params: dict
) -> torch.Tensor:
    """
    JKR contact area between two cells.

    When ellipsoid axes are present, effective radii replace R_i/R_j in:
      - The reduced modulus K*
      - The effective radius R_eff (harmonic mean)
      - The compressive force F (via R_c sum)
      - The contact gate (sigmoid at R_contact)

    Full derivation in report. Returns contact area in μm².
    """
    gamma_i = _get_gamma(cell_i, params)
    gamma_j = _get_gamma(cell_j, params)

    if cell_i.axes is not None and cell_j.axes is not None:
        r_i, r_j, d_vec = ellipsoid_contact_distance(cell_i, cell_j)
        d = torch.norm(d_vec)
    else:
        r_i = torch.tensor(cell_i.R, dtype=DTYPE, device=DEVICE)
        r_j = torch.tensor(cell_j.R, dtype=DTYPE, device=DEVICE)
        d   = torch.norm(cell_i.position - cell_j.position)

    E_star_i = 2.0 * gamma_i / (r_i + 1e-12)
    E_star_j = 2.0 * gamma_j / (r_j + 1e-12)
    K_star   = (4.0 / 3.0) * (E_star_i * E_star_j) / (E_star_i + E_star_j + 1e-12)

    R_eff = (r_i * r_j) / (r_i + r_j + 1e-12)

    R_c_sum = 0.8 * (r_i + r_j)
    F = K_REP * torch.clamp(R_c_sum - d, min=0.0)

    term1    = 6.0 * math.pi * w * R_eff * F
    term2    = (3.0 * math.pi * w * R_eff) ** 2
    interior = torch.clamp(term1 + term2, min=0.0)

    a_cubed = (R_eff / (K_star + 1e-12)) * (
        F + 3.0 * math.pi * w * R_eff + torch.sqrt(interior)
    )
    a_cubed  = torch.clamp(a_cubed, min=0.0)
    a        = a_cubed ** (1.0 / 3.0)

    # Gate: cells are "in contact" when surfaces meet, not just when compressed
    R_contact = r_i + r_j
    gate      = torch.sigmoid(20.0 * (R_contact - d))
    return math.pi * a**2 * gate


def adhesion_energy(
    cell_i: "CellAgent",
    cell_j: "CellAgent",
    w: float,
    params: dict
) -> torch.Tensor:
    """E_adh = -w · A_contact"""
    return -w * jkr_contact_area(cell_i, cell_j, w, params)


# ---------------------------------------------------------------------------
# Energy Term 5 — Cortical Flow
# ---------------------------------------------------------------------------

def cortical_flow_energy(cell: "CellAgent", alpha: float) -> torch.Tensor:
    """
    P-lineage posterior bias: E = -α·x  (force pushes cell toward +x).
    Zero for AB and EMS lineages.
    """
    if cell.lineage == "P":
        return -alpha * cell.position[0]
    return torch.tensor(0.0, dtype=DTYPE, device=DEVICE)


# ---------------------------------------------------------------------------
# Total Energy
# ---------------------------------------------------------------------------

def total_energy(cells: list, params: dict) -> torch.Tensor:
    """
    E_total = Σ_i [E_shell + E_volume + E_cortical]
            + Σ_{i<j} [E_rep + E_adh]

    shape_energy is embedded in volume_energy — no separate call needed here.
    """
    w     = params['w']
    alpha = params['alpha']

    E = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

    for cell in cells:
        E = E + shell_energy(cell)
        E = E + volume_energy(cell, cells)
        E = E + cortical_flow_energy(cell, alpha)

    n = len(cells)
    for i in range(n):
        for j in range(i + 1, n):
            E = E + overlap_repulsion(cells[i], cells[j])
            E = E + adhesion_energy(cells[i], cells[j], w, params)

    return E


# ---------------------------------------------------------------------------
# Inner Loop — Overdamped Gradient Flow (position-only, for test_physics.py)
# ---------------------------------------------------------------------------

def run_inner_loop(
    cells: list,
    params: dict,
    verbose: bool = False,
    max_steps: int = 5000,
    lr: float = 0.01,
    convergence_threshold: float = 1e-8,
    convergence_window: int = 30,
) -> tuple:
    """
    Overdamped gradient descent on position only (used by test_6).

    dx/dt = -(1/η)∇E  →  pos(t+1) = pos(t) - lr·∇E

    Convergence: |ΔE| < threshold for `window` consecutive steps.
    Returns (final_E: float, steps_taken: int).
    """
    positions     = [cell.position for cell in cells]
    optimizer     = torch.optim.SGD(positions, lr=lr, momentum=0.0)
    prev_E        = None
    converge_count = 0

    for step in range(max_steps):
        optimizer.zero_grad()
        E = total_energy(cells, params)
        E.backward()

        torch.nn.utils.clip_grad_norm_(positions, max_norm=5.0)
        optimizer.step()

        current_E = E.item()

        if prev_E is not None:
            delta = abs(current_E - prev_E)
            if delta < convergence_threshold:
                converge_count += 1
            else:
                converge_count = 0

            if converge_count >= convergence_window:
                if verbose:
                    print(f"  Converged at step {step + 1}, E = {current_E:.8f}")
                return current_E, step + 1

        prev_E = current_E

        if verbose and (step + 1) % 500 == 0:
            print(f"  Step {step + 1}: E = {current_E:.8f}")

    if verbose:
        print(f"  Did not converge after {max_steps} steps, E = {current_E:.8f}")
    return current_E, max_steps
