"""
physics.py — Biophysical simulation engine for C. elegans early embryogenesis.

Energy terms:
  1. Eggshell confinement (ellipsoidal)
  2. Volume elasticity (spherical cap overlap)
  3. Overlap repulsion (hard-core)
  4. Adhesion (full JKR contact mechanics)
  5. Cortical flow (P-lineage posterior bias)
"""

import math
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
print(f"[physics.py] Using device: {DEVICE}")

# Inital config
SHELL_A = 25.0   # half-axis AP (X), μm
SHELL_B = 15.0   # half-axis Y, μm
SHELL_C = 15.0   # half-axis Z, μm


K_SHELL = 200.0
K_VOL   = 0.01
K_REP   = 50.0


class CellAgent:
    """
    Autonomous cell agent with physical properties.

    Properties:
        identity : str   — "ABa", "ABp", "EMS", "P2", etc.
        lineage  : str   — "AB", "EMS", or "P"
        V0       : float — target volume in μm³ (from dataset)
        R        : float — volumetric radius = (3·V0/4π)^(1/3)
        R_c      : float — contact radius = 0.8 × R
        position : Tensor(3,) — (x, y, z) in μm, requires_grad=True
    """

    # Lineage mapping
    _LINEAGE_MAP = {
        "ABa": "AB", "ABp": "AB", "AB": "AB",
        "EMS": "EMS",
        "P2": "P", "P1": "P", "P": "P",
    }

    def __init__(self, identity: str, V0: float):
        """
        Args:
            identity: Cell name (e.g. "ABa", "ABp", "EMS", "P2")
            V0: Target volume in μm³
        """
        self.identity = identity
        self.lineage = self._LINEAGE_MAP.get(identity, identity)
        self.V0 = V0
        self.R = (3.0 * V0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        self.R_c = 0.8 * self.R

        # Position is set externally
        self.position = None

    def set_position(self, x: float, y: float, z: float):
        """Set cell position, requires_grad=True"""
        self.position = torch.tensor(
            [x, y, z], dtype=DTYPE, device=DEVICE, requires_grad=True
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


# ═══════════════════════════════════════════════════════════════════════════
# Energy Term 1 — Eggshell Confinement
# ═══════════════════════════════════════════════════════════════════════════
def shell_energy(cell: CellAgent) -> torch.Tensor:
    """
    Ellipsoidal eggshell confinement energy.Added a soft penalty for being outside the shell, zero inside.

    E_shell = K_shell · clamp(f_eff - 1.0, min=0)²

    where f_eff = (x/a_eff)² + (y/b_eff)² + (z/c_eff)²
    and a_eff = a - R_c, b_eff = b - R_c, c_eff = c - R_c
    """
    pos = cell.position
    a_eff = SHELL_A - cell.R_c
    b_eff = SHELL_B - cell.R_c
    c_eff = SHELL_C - cell.R_c

    f_eff = (pos[0] / a_eff) ** 2 + (pos[1] / b_eff) ** 2 + (pos[2] / c_eff) ** 2
    violation = torch.clamp(f_eff - 1.0, min=0.0)
    return K_SHELL * violation ** 2


# ═══════════════════════════════════════════════════════════════════════════
# Energy Term 2 — Volume Elasticity
# ═══════════════════════════════════════════════════════════════════════════
def _spherical_cap_volume(cell_i: CellAgent, cell_j: CellAgent) -> torch.Tensor:
    """
    Compute the spherical cap overlap volume that cell_j removes from cell_i.

    V_cap_i = π · h_i² · (R_i - h_i/3)  when d < R_i + R_j, else 0
    h_i = clamp(R_i - (d² + R_i² - R_j²) / (2·d), min=0)
    """
    d = torch.norm(cell_i.position - cell_j.position)
    R_i = cell_i.R
    R_j = cell_j.R

    # If cells don't overlap, V_cap = 0
    # Use a soft check: if d >= R_i + R_j, h_i clamps to 0 naturally
    # Add small epsilon to avoid division by zero
    d_safe = torch.clamp(d, min=1e-10)

    h_i = torch.clamp(
        R_i - (d_safe ** 2 + R_i ** 2 - R_j ** 2) / (2.0 * d_safe),
        min=0.0
    )

    V_cap = math.pi * h_i ** 2 * (R_i - h_i / 3.0)
    return V_cap


def volume_energy(cell: CellAgent, all_cells: list) -> torch.Tensor:
    """
    Volume elasticity energy for a single cell.

    E_volume = K_vol · (V_eff - V0)²
    V_eff = (4/3)π·R³ - Σ V_cap(cell, j)
    """
    V_full = (4.0 / 3.0) * math.pi * cell.R ** 3
    V_caps = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

    for other in all_cells:
        if other is not cell:
            V_caps = V_caps + _spherical_cap_volume(cell, other)

    V_eff = V_full - V_caps
    return K_VOL * (V_eff - cell.V0) ** 2


# ═══════════════════════════════════════════════════════════════════════════
# Energy Term 3 — Overlap Repulsion
# ═══════════════════════════════════════════════════════════════════════════
def overlap_repulsion(cell_i: CellAgent, cell_j: CellAgent) -> torch.Tensor:
    """
    Hard-core overlap repulsion.

    E_rep = K_rep · clamp(R_c_i + R_c_j - d, min=0)²
    """
    d = torch.norm(cell_i.position - cell_j.position)
    overlap = torch.clamp(cell_i.R_c + cell_j.R_c - d, min=0.0)
    return K_REP * overlap ** 2


# ═══════════════════════════════════════════════════════════════════════════
# Energy Term 4 — Adhesion (Full JKR Contact Mechanics)
# ═══════════════════════════════════════════════════════════════════════════
def _get_gamma(cell: CellAgent, params: dict) -> float:
    """Get cortical tension γ for a cell based on its lineage."""
    lineage = cell.lineage
    if lineage == "AB":
        return params['gamma_AB']
    elif lineage == "EMS":
        return params['gamma_EMS']
    elif lineage == "P":
        return params['gamma_P']
    else:
        raise ValueError(f"Unknown lineage: {lineage}")


def jkr_contact_area(
    cell_i: CellAgent,
    cell_j: CellAgent,
    w: float,
    params: dict
) -> torch.Tensor:
    """
    Compute JKR contact area between two cells.

    Steps:
      1. E*_i = 2·γ_i / R_i (Laplace law)
         K* = (4/3) · (E*_i · E*_j) / (E*_i + E*_j)
      2. R_eff = (R_i · R_j) / (R_i + R_j)
      3. F = K_rep · clamp(R_c_i + R_c_j - d, min=0)
      4. a³ = (R_eff/K*) · [F + 3π·w·R_eff + sqrt(clamp(6π·w·R_eff·F + (3π·w·R_eff)², min=0))]
         a = clamp(a³, min=0)^(1/3)
      5. gate = sigmoid(20 · (R_i + R_j - d))
         A_contact = π · a² · gate

    Returns:
        Scalar tensor: contact area in μm²
    """
    d = torch.norm(cell_i.position - cell_j.position)

    R_i = cell_i.R
    R_j = cell_j.R
    gamma_i = _get_gamma(cell_i, params)
    gamma_j = _get_gamma(cell_j, params)

    # Step 1: Effective modulus
    E_star_i = 2.0 * gamma_i / R_i
    E_star_j = 2.0 * gamma_j / R_j
    K_star = (4.0 / 3.0) * (E_star_i * E_star_j) / (E_star_i + E_star_j)

    # Step 2: Effective radius
    R_eff = (R_i * R_j) / (R_i + R_j)

    # Step 3: Compressive force
    F = K_REP * torch.clamp(cell_i.R_c + cell_j.R_c - d, min=0.0)

    # Step 4: JKR contact radius
    term1 = 6.0 * math.pi * w * R_eff * F
    term2 = (3.0 * math.pi * w * R_eff) ** 2
    interior = torch.clamp(term1 + term2, min=0.0)

    a_cubed = (R_eff / K_star) * (
        F + 3.0 * math.pi * w * R_eff + torch.sqrt(interior)
    )
    a_cubed = torch.clamp(a_cubed, min=0.0)
    a = a_cubed ** (1.0 / 3.0)

    # Step 5: Differentiable gate at volumetric radius sum (not R_c)
    # Cells are "in contact" when surfaces touch (d < R_i + R_j),
    # not when they are compressed (d < R_c_i + R_c_j).
    gate = torch.sigmoid(20.0 * (R_i + R_j - d))
    A_contact = math.pi * a ** 2 * gate

    return A_contact


def adhesion_energy(
    cell_i: CellAgent,
    cell_j: CellAgent,
    w: float,
    params: dict
) -> torch.Tensor:
    """
    Adhesion energy: E_adh = -w · A_contact
    """
    A = jkr_contact_area(cell_i, cell_j, w, params)
    return -w * A


# ═══════════════════════════════════════════════════════════════════════════
# Energy Term 5 — Cortical Flow (Phenomenological)
# ═══════════════════════════════════════════════════════════════════════════
def cortical_flow_energy(cell: CellAgent, alpha: float) -> torch.Tensor:
    """
    Cortical flow energy for P-lineage cells.

    E_cortical = -α · pos[0]   (only P lineage)
                = 0             (AB and EMS lineage)
    """
    if cell.lineage == "P":
        return -alpha * cell.position[0]
    else:
        return torch.tensor(0.0, dtype=DTYPE, device=DEVICE)


# ═══════════════════════════════════════════════════════════════════════════
# Total Energy
# ═══════════════════════════════════════════════════════════════════════════
def total_energy(cells: list, params: dict) -> torch.Tensor:
    """
    Compute total system energy.

    E_total = Σ_i [E_shell_i + E_volume_i + E_cortical_i]
             + Σ_{i<j} [E_rep_ij + E_adh_ij]

    Args:
        cells: list of CellAgent
        params: dict with keys: gamma_AB, gamma_EMS, gamma_P, w, alpha

    Returns:
        Scalar tensor: total energy
    """
    w = params['w']
    alpha = params['alpha']

    E = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

    # Single-cell terms
    for cell in cells:
        E = E + shell_energy(cell)
        E = E + volume_energy(cell, cells)
        E = E + cortical_flow_energy(cell, alpha)

    # Pairwise terms
    n = len(cells)
    for i in range(n):
        for j in range(i + 1, n):
            E = E + overlap_repulsion(cells[i], cells[j])
            E = E + adhesion_energy(cells[i], cells[j], w, params)

    return E


# ═══════════════════════════════════════════════════════════════════════════
# Inner Loop — Overdamped Gradient Flow
# ═══════════════════════════════════════════════════════════════════════════
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
    Run overdamped gradient descent (Forward Euler = SGD with zero momentum).

    dx/dt = -(1/η)·∇E  →  pos(t+1) = pos(t) - lr·∇E
    lr = dt/η = 0.01

    Convergence: |ΔE| < threshold for `window` consecutive steps.

    Args:
        cells: list of CellAgent with positions set
        params: physics parameters
        verbose: print progress
        max_steps: maximum iterations
        lr: learning rate (dt/η)
        convergence_threshold: |ΔE| threshold
        convergence_window: consecutive steps needed

    Returns:
        (final_E: float, steps_taken: int)
    """
    # Collect position parameters
    positions = [cell.position for cell in cells]
    optimizer = torch.optim.SGD(positions, lr=lr, momentum=0.0)

    prev_E = None
    converge_count = 0

    for step in range(max_steps):
        optimizer.zero_grad()
        E = total_energy(cells, params)
        E.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(positions, max_norm=5.0)

        optimizer.step()

        current_E = E.item()

        # Check convergence
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
