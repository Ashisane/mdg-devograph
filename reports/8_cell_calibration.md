# Report 3b: 8-Cell Stage Calibration & Topology Evaluation

## Objective
Extend the 4-cell active matter simulation to the 8-cell stage to capture higher-order cell-cell contacts and extract features for SINDy, specifically addressing the geometry of the expanding embryo.

## Calibration Findings and Parameter Transferability
The calibration process yielded an important scientific finding regarding the transferability of physics parameters across developmental stages in spherical cell agent-based models (ABMs).

*   **4-Cell Parameters**: `gamma_AB ≈ 0.85, gamma_EMS ≈ 0.76, gamma_P ≈ 0.68, w ≈ 0.85`
*   **8-Cell Required Parameters**: The 8-cell stage required dramatically softer gammas (`gamma_AB = 0.10, gamma_EMS = 0.09, gamma_P = 0.08`) and higher relative adhesion/flow (`w = 10.0, alpha = 10.0`) to produce correct topology.

**Conclusion on Parameter Transfer**: The physics parameters that calibrate the 4-cell stage are **not physically meaningful** for predicting the 8-cell stage without recalibration. Because the model uses rigid spherical interactions (JKR) rather than deformable polyhedra, smaller cells packed at higher density in the 8-cell stage require fundamentally "softer" proxy parameters to allow for the same relative surface contact. 

**Scientific Implication**: The spherical cell abstraction requires stage-specific recalibration. This suggests a missing dimension in the physics model—specifically explicit cell shape deformation—that becomes critical as cell counts increase and packing density rises.

## Simulation Validation: Topology Emergence

Despite the need for stage-specific parameters, the model successfully captures the core emergent topology of the 8-cell stage.

Validation over 10 independent runs (with positional perturbation) yields:
*   **Topology Success Rate**: **10/10 correct**
    *   *Criterion*: All expected 19 contacts are present, and the structurally impossible ABa-P2 contact is strictly absent (checked across all runs).
*   **Quantitative Fit (R²)**: While topology is strictly correct, R² remains low (parallel to the 4-cell stage findings). This limitation is diagnosed identically: the uniform spherical geometry forces contact areas to be uniform, preventing the model from matching the wide variance in biologically measured contact areas.

The primary requirement for SINDy—topologically correct adjacency graphs—is fully met.