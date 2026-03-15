"""Inspect simulation_results.pt — writes to inspect_clean.txt"""
import torch

data = torch.load("simulation_results.pt", weights_only=False)

lines = []
lines.append("=== TOP-LEVEL KEYS ===")
for k, v in data.items():
    if isinstance(v, list):
        lines.append(f"  {k}: list, len={len(v)}")
    elif isinstance(v, dict):
        lines.append(f"  {k}: dict, keys={list(v.keys())}")
    else:
        lines.append(f"  {k}: {type(v).__name__} = {v}")

traj = data["trajectory"]
lines.append(f"\n=== TRAJECTORY: {len(traj)} frames ===")

for idx in [0, 10, 20, 21, 22, 33, 34, -1]:
    actual_idx = idx if idx >= 0 else len(traj) + idx
    if actual_idx >= len(traj):
        continue
    f = traj[actual_idx]
    lines.append(f"\nFrame[{actual_idx}]: t={f['t']}, n_cells={f['n_cells']}, E={f['E_total']:.2f}")
    for c in f["cells"]:
        pos = [round(x, 2) for x in c["position"]]
        lines.append(f"  {c['identity']} ({c['lineage']}): pos={pos} R={c['R']:.2f} V0={c['V0']:.1f}")
    contacts_str = ", ".join(f"{k}={v:.2f}" for k, v in f["contacts"].items())
    lines.append(f"  contacts: {contacts_str}")

lines.append("\n=== DIVISION EVENTS ===")
prev_n = traj[0]["n_cells"]
for i, f in enumerate(traj):
    if f["n_cells"] != prev_n:
        cell_names = [c["identity"] for c in f["cells"]]
        lines.append(f"  Frame[{i}]: {prev_n}->{f['n_cells']} cells at t={f['t']}, cells={cell_names}")
        prev_n = f["n_cells"]

lines.append(f"\n=== BEST PARAMS ===")
for k, v in data["best_params"].items():
    lines.append(f"  {k}: {v}")

lines.append(f"\n=== MEASURED AREAS ===")
for k, v in data["measured_areas"].items():
    lines.append(f"  {k}: {v:.2f}")

lines.append(f"\n=== CALIBRATION LOSS ===")
loss = data.get("calibration_loss", [])
lines.append(f"  count={len(loss)}, first={loss[0]:.4f}, last={loss[-1]:.4f}")

with open("inspect_clean.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Written to inspect_clean.txt")
