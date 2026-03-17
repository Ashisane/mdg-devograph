# Report 3 -- C. elegans ABM Animation

## 1. ANIMATION DETAILS

- **Total frames**: 300
- **FPS**: 30
- **Duration**: 10.0 seconds
- **Resolution**: 18x10 inches at 150 DPI
- **File size**: 3.7 MB
- **Render time**: 160.1 seconds

## 2. FRAME MAPPING

- **AB division**: animation frame 73 / 300
- **P1 division**: animation frame 119 / 300
- **2-cell stage**: frames 0 to 72
- **3-cell stage**: frames 73 to 118
- **4-cell stage**: frames 119 to 299

## 3. EMERGENCE SHOWN

The animation visually demonstrates:
- **Cell division**: AB divides perpendicular to AP axis (Y), 
  P1 divides parallel to AP axis (X)
- **3+1 diamond topology**: After 4-cell equilibration, 
  ABa and P2 are NOT in contact (forbidden edge shown as dashed red X)
- **Contact graph emergence**: All 5 expected contacts form naturally 
  from physics without hardcoding topology
- **Energy minimization**: System energy decreases to equilibrium 
  after each division event

## 4. KEY FRAMES

- **simulation_frame_2cell.png** (frame 63): 
  Two founding cells (AB and P1) in equilibrium within the eggshell
- **simulation_frame_3cell.png** (frame 78): 
  Just after AB division into ABa + ABp, 3 cells rearranging
- **simulation_frame_4cell.png** (frame 129): 
  Just after P1 division into EMS + P2, 4 cells equilibrating
- **simulation_frame_final.png** (frame 299): 
  Final 4-cell equilibrium showing the emergent diamond topology

## 5. PROBLEMS FACED AND HOW SOLVED

_To be filled after rendering._

## 6. FINAL CHECKLIST

- [x] simulation.mp4 generated
- [x] simulation_frame_2cell.png saved
- [x] simulation_frame_3cell.png saved
- [x] simulation_frame_4cell.png saved
- [x] simulation_frame_final.png saved
- [x] ABa-P2 forbidden edge visible throughout 4-cell stage
- [x] Division flash visible at both division events
- [x] DevoGraph builds correctly
- [x] Energy curve shows decrease to equilibrium
- [x] Camera rotates
- [x] Measured contact area lines shown in bottom center