"""
animation.py — Visualization for C. elegans ABM simulation.


"""

import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D


SHELL_A = 25.0  # half-axis AP (X)
SHELL_B = 15.0  # half-axis Y
SHELL_C = 15.0  # half-axis Z

BG_COLOR = "#0D1117"
SLATE = "#94A3B8"

CELL_COLORS = {
    "AB":  "#93C5FD",  # pre-division AB
    "P1":  "#FCA5A5",  # pre-division P1
    "ABa": "#60A5FA",  # AB lineage
    "ABp": "#60A5FA",
    "EMS": "#FB923C",  # orange
    "P2":  "#F87171",  # red
}

CONTACT_LINE_COLORS = {
    "ABa-ABp": "#60A5FA",
    "ABa-EMS": "#93C5FD",
    "ABp-EMS": "#818CF8",
    "ABp-P2":  "#C084FC",
    "EMS-P2":  "#FB923C",
    "ABa-P2":  "#EF4444",
}

PARAM_COLORS = {
    "gamma_AB":  "#60A5FA",
    "gamma_EMS": "#FB923C",
    "gamma_P":   "#F87171",
    "w":         "#A78BFA",
    "alpha":     "#34D399",
}

TARGET_FRAMES = 300
FPS = 30


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def avg_color(c1, c2):
    r1 = hex_to_rgb(c1)
    r2 = hex_to_rgb(c2)
    return tuple((a + b) / 2 for a, b in zip(r1, r2))



def load_and_precompute():
    print("Loading simulation_results.pt ...")
    data = torch.load("simulation_results.pt", weights_only=False)

    traj = data["trajectory"]
    best_params = data["best_params"]
    measured_areas = data["measured_areas"]
    n_traj = len(traj)

    print(f"  Trajectory frames: {n_traj}")
    print(f"  Best params: {best_params}")

    # Map 84 trajectory frames -> 300 animation frames
    # Use floor mapping (repeat frames if needed)
    frame_map = []
    for i in range(TARGET_FRAMES):
        traj_idx = int(i * (n_traj - 1) / (TARGET_FRAMES - 1))
        traj_idx = min(traj_idx, n_traj - 1)
        frame_map.append(traj_idx)

    # Find division frames in animation space
    div_frames = {"AB": None, "P1": None}
    prev_n = traj[0]["n_cells"]
    for ti, f in enumerate(traj):
        if f["n_cells"] != prev_n:
            if prev_n == 2 and f["n_cells"] == 3:
                div_frames["AB"] = ti
            elif prev_n == 3 and f["n_cells"] == 4:
                div_frames["P1"] = ti
            prev_n = f["n_cells"]

    # Map division frames to animation frames
    ab_div_anim = None
    p1_div_anim = None
    for ai, ti in enumerate(frame_map):
        if div_frames["AB"] is not None and ti >= div_frames["AB"] and ab_div_anim is None:
            ab_div_anim = ai
        if div_frames["P1"] is not None and ti >= div_frames["P1"] and p1_div_anim is None:
            p1_div_anim = ai

    print(f"  AB division: traj frame {div_frames['AB']} -> anim frame {ab_div_anim}")
    print(f"  P1 division: traj frame {div_frames['P1']} -> anim frame {p1_div_anim}")

    # Pre-compute all frame data
    frames_data = []
    all_energies = []
    all_contact_series = {k: [] for k in CONTACT_LINE_COLORS.keys()}

    for ai in range(TARGET_FRAMES):
        ti = frame_map[ai]
        f = traj[ti]
        cells = []
        for c in f["cells"]:
            cells.append({
                "identity": c["identity"],
                "lineage": c["lineage"],
                "position": np.array(c["position"]),
                "R": c["R"],
                "V0": c["V0"],
            })

        contacts = f["contacts"]
        energy = f["E_total"]
        n_cells = f["n_cells"]

        # Division flash: 3 frames before division
        flash_cell = None
        if ab_div_anim is not None and ab_div_anim - 3 <= ai < ab_div_anim:
            flash_cell = "AB"
        if p1_div_anim is not None and p1_div_anim - 3 <= ai < p1_div_anim:
            flash_cell = "P1"

        frames_data.append({
            "cells": cells,
            "contacts": contacts,
            "energy": energy,
            "n_cells": n_cells,
            "flash_cell": flash_cell,
        })

        all_energies.append(energy)

        # Build contact series
        for pair_key in CONTACT_LINE_COLORS.keys():
            val = contacts.get(pair_key, 0.0)
            if val == 0.0:
                # Try reversed key
                parts = pair_key.split("-")
                rev_key = f"{parts[1]}-{parts[0]}"
                val = contacts.get(rev_key, 0.0)
            all_contact_series[pair_key].append(val)

    energies_arr = np.array(all_energies)

    # Key frame indices
    # 2-cell: early frame
    key_2cell = max(0, (ab_div_anim or 20) - 10)
    # 3-cell: just after AB div
    key_3cell = min(TARGET_FRAMES - 1, (ab_div_anim or 20) + 5)
    # 4-cell: just after P1 div
    key_4cell = min(TARGET_FRAMES - 1, (p1_div_anim or 50) + 10)
    # Final: last frame
    key_final = TARGET_FRAMES - 1

    return {
        "frames_data": frames_data,
        "energies": energies_arr,
        "contact_series": all_contact_series,
        "best_params": best_params,
        "measured_areas": measured_areas,
        "ab_div_anim": ab_div_anim,
        "p1_div_anim": p1_div_anim,
        "key_frames": {
            "2cell": key_2cell,
            "3cell": key_3cell,
            "4cell": key_4cell,
            "final": key_final,
        },
    }



def create_figure(precomputed):
    fig = plt.figure(figsize=(18, 10), facecolor=BG_COLOR)

    # Left panel: 3D (40% width, 80% height)
    ax3d = fig.add_axes([0.01, 0.22, 0.42, 0.76], projection="3d",
                         facecolor="none")
    ax3d.set_axis_off()
    ax3d.set_xlim(-28, 28)
    ax3d.set_ylim(-20, 20)
    ax3d.set_zlim(-20, 20)
    ax3d.view_init(elev=25, azim=30)

    # Draw eggshell wireframe (static — draw once)
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 12)
    xs = SHELL_A * np.outer(np.cos(u), np.sin(v))
    ys = SHELL_B * np.outer(np.sin(u), np.sin(v))
    zs = SHELL_C * np.outer(np.ones_like(u), np.cos(v))
    ax3d.plot_wireframe(xs, ys, zs, color="#334155", alpha=0.2, linewidth=0.3)

    # Right panel: DevoGraph (2D, 40% width)
    ax_devo = fig.add_axes([0.50, 0.22, 0.46, 0.76], facecolor=BG_COLOR)
    ax_devo.set_xlim(-0.05, 1.05)
    ax_devo.set_ylim(-0.05, 1.05)
    ax_devo.set_axis_off()
    ax_devo.set_title("DevoGraph", color="white", fontsize=12, pad=10)

    # Bottom Left: Energy
    ax_energy = fig.add_axes([0.05, 0.04, 0.27, 0.16], facecolor=BG_COLOR)
    ax_energy.set_title("System Energy", fontsize=9, color="white", pad=4)
    ax_energy.tick_params(colors=SLATE, labelsize=7)
    for spine in ax_energy.spines.values():
        spine.set_color(SLATE)
        spine.set_linewidth(0.5)

    # Bottom Center: Contact Areas
    ax_contacts = fig.add_axes([0.37, 0.04, 0.27, 0.16], facecolor=BG_COLOR)
    ax_contacts.set_title("Contact Areas", fontsize=9, color="white", pad=4)
    ax_contacts.tick_params(colors=SLATE, labelsize=7)
    for spine in ax_contacts.spines.values():
        spine.set_color(SLATE)
        spine.set_linewidth(0.5)

    # Bottom Right: Calibrated Parameters (static bar chart)
    ax_params = fig.add_axes([0.70, 0.04, 0.27, 0.16], facecolor=BG_COLOR)
    ax_params.set_title("Calibrated Params", fontsize=9, color="white", pad=4)
    ax_params.tick_params(colors=SLATE, labelsize=7)
    for spine in ax_params.spines.values():
        spine.set_color(SLATE)
        spine.set_linewidth(0.5)

    # Draw static parameter bars
    bp = precomputed["best_params"]
    param_names = ["gamma_AB", "gamma_EMS", "gamma_P", "w", "alpha"]
    param_labels = ["g_AB", "g_EMS", "g_P", "w", "alpha"]
    param_vals = [bp.get(k, 0) for k in param_names]
    param_cols = [PARAM_COLORS[k] for k in param_names]
    y_pos = np.arange(len(param_names))
    bars = ax_params.barh(y_pos, param_vals, color=param_cols, height=0.6, alpha=0.85)
    ax_params.set_yticks(y_pos)
    ax_params.set_yticklabels(param_labels, fontsize=7, color=SLATE)
    ax_params.invert_yaxis()
    for i, (bar, val) in enumerate(zip(bars, param_vals)):
        ax_params.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                       f"{val:.3f}", va="center", fontsize=7, color=SLATE)

    # Draw static energy background
    energies = precomputed["energies"]
    frames_x = np.arange(TARGET_FRAMES)

    # Plot full energy curve dimly
    ax_energy.plot(frames_x, energies, color="#34D399", linewidth=0.5, alpha=0.3)
    ax_energy.set_xlabel("frame", fontsize=8, color=SLATE)
    ax_energy.set_ylabel("E_total", fontsize=8, color=SLATE)

    # Division dashed lines
    if precomputed["ab_div_anim"] is not None:
        ax_energy.axvline(precomputed["ab_div_anim"], color="#60A5FA",
                          linestyle="--", linewidth=0.8, alpha=0.7)
        ax_energy.text(precomputed["ab_div_anim"] + 2, energies.max() * 0.9,
                       "AB div", fontsize=6, color="#60A5FA")
    if precomputed["p1_div_anim"] is not None:
        ax_energy.axvline(precomputed["p1_div_anim"], color="#F87171",
                          linestyle="--", linewidth=0.8, alpha=0.7)
        ax_energy.text(precomputed["p1_div_anim"] + 2, energies.max() * 0.8,
                       "P1 div", fontsize=6, color="#F87171")

    # Static contact area measured lines
    measured = precomputed["measured_areas"]
    scale_val = precomputed["best_params"].get("scale", 1.0)
    # Convert measured (raw) to simulation units for comparison
    # Actually the simulation outputs are in um^2, measured are in raw voxels
    # We show contact areas in simulation units and measured/scale as reference
    for pair_key, color in CONTACT_LINE_COLORS.items():
        if pair_key == "ABa-P2":
            continue
        measured_key = pair_key.replace("-", "-")
        if measured_key in measured:
            sim_units_val = measured[measured_key] / scale_val
            ax_contacts.axhline(sim_units_val, color="gray", linestyle="--",
                                linewidth=0.5, alpha=0.4)

    ax_contacts.set_xlabel("frame", fontsize=8, color=SLATE)
    ax_contacts.set_ylabel("area (um^2)", fontsize=8, color=SLATE)

    return fig, ax3d, ax_devo, ax_energy, ax_contacts



# Pre-compute unit sphere mesh
_u_sphere = np.linspace(0, 2 * np.pi, 20)
_v_sphere = np.linspace(0, np.pi, 10)
_sx = np.outer(np.cos(_u_sphere), np.sin(_v_sphere))
_sy = np.outer(np.sin(_u_sphere), np.sin(_v_sphere))
_sz = np.outer(np.ones_like(_u_sphere), np.cos(_v_sphere))


# DevoGraph node positions
DEVO_POS = {
    "AB":  (0.25, 0.75),
    "ABa": (0.25, 0.75),
    "P1":  (0.75, 0.75),
    "ABp": (0.75, 0.75),
    "EMS": (0.30, 0.25),
    "P2":  (0.75, 0.25),
}



def make_update(fig, ax3d, ax_devo, ax_energy, ax_contacts, precomputed):
    """Create the update function for FuncAnimation."""
    frames_data = precomputed["frames_data"]
    energies = precomputed["energies"]
    contact_series = precomputed["contact_series"]
    ab_div = precomputed["ab_div_anim"]
    p1_div = precomputed["p1_div_anim"]

    # Persistent energy line and dot
    energy_line, = ax_energy.plot([], [], color="#34D399", linewidth=1.5)
    energy_dot, = ax_energy.plot([], [], "o", color="#EF4444", markersize=4)

    # Persistent contact lines
    contact_lines = {}
    for pair_key, color in CONTACT_LINE_COLORS.items():
        ls = "--" if pair_key == "ABa-P2" else "-"
        line, = ax_contacts.plot([], [], color=color, linewidth=1.5,
                                  linestyle=ls, label=pair_key)
        contact_lines[pair_key] = line

    # Add legend once
    ax_contacts.legend(fontsize=6, loc="upper left",
                        facecolor=BG_COLOR, edgecolor=SLATE,
                        labelcolor=SLATE)

    # Stage label text
    stage_text = ax3d.text2D(0.05, 0.95, "", transform=ax3d.transAxes,
                              fontsize=11, color="white", fontweight="bold")

    # Store artists that need clearing
    dynamic_3d = []
    dynamic_devo = []

    def update(frame_idx):
        nonlocal dynamic_3d, dynamic_devo

        fd = frames_data[frame_idx]
        cells = fd["cells"]
        contacts = fd["contacts"]
        flash_cell = fd["flash_cell"]
        n_cells = fd["n_cells"]

        # ─── Clear previous dynamic artists ───
        for art in dynamic_3d:
            try:
                art.remove()
            except Exception:
                pass
        dynamic_3d.clear()

        for art in dynamic_devo:
            try:
                art.remove()
            except Exception:
                pass
        dynamic_devo.clear()

        # ─── Camera rotation ───
        ax3d.view_init(elev=25, azim=30 + frame_idx * 0.8)

        # ─── Stage label ───
        stage_map = {2: "2-cell", 3: "3-cell", 4: "4-cell"}
        stage_text.set_text(stage_map.get(n_cells, f"{n_cells}-cell"))

        # ─── Draw cells as spheres ───
        cell_pos_map = {}
        for c in cells:
            pos = c["position"]
            R = c["R"]
            identity = c["identity"]
            cell_pos_map[identity] = pos

            # Color
            color = CELL_COLORS.get(identity, "#888888")
            alpha = 0.85
            draw_R = R

            # Division flash
            if flash_cell and identity == flash_cell:
                color = "#FFFFFF"
                draw_R = R * 1.1

            xs = pos[0] + draw_R * _sx
            ys = pos[1] + draw_R * _sy
            zs = pos[2] + draw_R * _sz

            surf = ax3d.plot_surface(xs, ys, zs, color=color, alpha=alpha,
                                      shade=True, linewidth=0)
            dynamic_3d.append(surf)

            # Label
            txt = ax3d.text(pos[0], pos[1], pos[2] + R + 1.5,
                            identity, ha="center", fontsize=8,
                            color="white", fontweight="bold")
            dynamic_3d.append(txt)

        # ─── Contact lines (3D) ───
        for pair_key in CONTACT_LINE_COLORS.keys():
            parts = pair_key.split("-")
            c1, c2 = parts[0], parts[1]
            if c1 not in cell_pos_map or c2 not in cell_pos_map:
                continue

            p1 = cell_pos_map[c1]
            p2 = cell_pos_map[c2]

            # Get area
            area = contacts.get(pair_key, 0.0)
            if area == 0.0:
                rev = f"{c2}-{c1}"
                area = contacts.get(rev, 0.0)

            if pair_key == "ABa-P2":
                # Always draw forbidden contact
                line = ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                color="#EF4444", linestyle="--", linewidth=1,
                                alpha=0.4)
                dynamic_3d.extend(line)
                mid = (p1 + p2) / 2
                txt = ax3d.text(mid[0], mid[1], mid[2], "X",
                                fontsize=7, color="#EF4444", ha="center")
                dynamic_3d.append(txt)
            elif area > 50:
                col1 = CELL_COLORS.get(c1, "#888888")
                col2 = CELL_COLORS.get(c2, "#888888")
                lw = 1 + area / 600
                line_color = avg_color(col1, col2)
                line = ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                color=line_color, linewidth=lw, alpha=0.6)
                dynamic_3d.extend(line)

        # ─── DevoGraph (right panel) ───
        # Draw nodes
        for c in cells:
            identity = c["identity"]
            if identity not in DEVO_POS:
                continue
            dx, dy = DEVO_POS[identity]
            color = CELL_COLORS.get(identity, "#888888")
            size = c["V0"] / 40

            # Flash
            if flash_cell and identity == flash_cell:
                color = "#FFFFFF"

            circle = ax_devo.scatter([dx], [dy], s=size, c=color,
                                      alpha=0.9, zorder=5, edgecolors="white",
                                      linewidths=0.5)
            dynamic_devo.append(circle)

            label = ax_devo.text(dx, dy - 0.08, identity, fontsize=9,
                                  color="white", ha="center", va="top")
            dynamic_devo.append(label)

        # Draw edges
        cell_ids = set(c["identity"] for c in cells)
        for pair_key in CONTACT_LINE_COLORS.keys():
            parts = pair_key.split("-")
            c1, c2 = parts[0], parts[1]
            if c1 not in cell_ids or c2 not in cell_ids:
                continue
            if c1 not in DEVO_POS or c2 not in DEVO_POS:
                continue

            p1x, p1y = DEVO_POS[c1]
            p2x, p2y = DEVO_POS[c2]

            area = contacts.get(pair_key, 0.0)
            if area == 0.0:
                rev = f"{c2}-{c1}"
                area = contacts.get(rev, 0.0)

            if pair_key == "ABa-P2":
                line = ax_devo.plot([p1x, p2x], [p1y, p2y],
                                    color="#EF4444", linestyle="--",
                                    linewidth=1, alpha=0.3)
                dynamic_devo.extend(line)
                mx, my = (p1x + p2x) / 2, (p1y + p2y) / 2
                txt = ax_devo.text(mx, my + 0.03, "forbidden", fontsize=6,
                                    color="#EF4444", ha="center")
                dynamic_devo.append(txt)
            elif area > 50:
                lw = 0.5 + area / 500
                max_area = max(contacts.values()) if contacts else 1
                alpha_val = 0.7 + 0.3 * (area / max(max_area, 1))
                color = CONTACT_LINE_COLORS.get(pair_key, "#888888")
                line = ax_devo.plot([p1x, p2x], [p1y, p2y],
                                    color=color, linewidth=lw,
                                    alpha=min(alpha_val, 1.0))
                dynamic_devo.extend(line)

        # ─── Bottom Left: Energy up to current frame ───
        x_data = np.arange(frame_idx + 1)
        energy_line.set_data(x_data, energies[:frame_idx + 1])
        energy_dot.set_data([frame_idx], [energies[frame_idx]])
        ax_energy.set_xlim(0, TARGET_FRAMES)
        e_min = energies[:frame_idx + 1].min() if frame_idx > 0 else energies[0] - 1
        e_max = energies[:frame_idx + 1].max() if frame_idx > 0 else energies[0] + 1
        margin = max(abs(e_max - e_min) * 0.1, 10)
        ax_energy.set_ylim(e_min - margin, e_max + margin)

        # ─── Bottom Center: Contact areas up to current frame ───
        for pair_key, line in contact_lines.items():
            y_data = contact_series[pair_key][:frame_idx + 1]
            line.set_data(np.arange(frame_idx + 1), y_data)
        ax_contacts.set_xlim(0, TARGET_FRAMES)
        # Compute y range from all visible contact data
        all_visible = []
        for pair_key in CONTACT_LINE_COLORS.keys():
            all_visible.extend(contact_series[pair_key][:frame_idx + 1])
        if all_visible:
            c_max = max(all_visible)
            ax_contacts.set_ylim(0, max(c_max * 1.2, 100))

        if frame_idx % 50 == 0:
            print(f"  Rendering frame {frame_idx}/{TARGET_FRAMES}")

        return []

    return update



def save_key_frame(fig, frame_idx, update_fn, filename):
    """Save a single frame as PNG."""
    update_fn(frame_idx)
    fig.savefig(filename, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    print(f"  Saved: {filename}")


def generate_report(precomputed, mp4_path, png_files, duration_s):
    """Generate report_3.md."""
    kf = precomputed["key_frames"]
    ab_div = precomputed["ab_div_anim"]
    p1_div = precomputed["p1_div_anim"]

    mp4_size = os.path.getsize(mp4_path) / (1024 * 1024) if os.path.exists(mp4_path) else 0

    lines = []
    lines.append("# Report 3 -- C. elegans ABM Animation\n")



    report_text = "\n".join(lines)
    with open("report_3.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("  Saved: report_3.md")


def main():
    print("=" * 60)
    print("PROMPT 3: ANIMATION")
    print("=" * 60)

    # Phase 1
    precomputed = load_and_precompute()
    print(f"  Pre-computed {TARGET_FRAMES} frames")

    # Phase 2
    print("\nCreating figure layout ...")
    fig, ax3d, ax_devo, ax_energy, ax_contacts = create_figure(precomputed)

    # Phase 3
    print("\nSetting up animation ...")
    update_fn = make_update(fig, ax3d, ax_devo, ax_energy, ax_contacts, precomputed)

    # Save key-frame PNGs
    print("\nSaving key-frame PNGs ...")
    kf = precomputed["key_frames"]
    png_files = [
        "simulation_frame_2cell.png",
        "simulation_frame_3cell.png",
        "simulation_frame_4cell.png",
        "simulation_frame_final.png",
    ]
    save_key_frame(fig, kf["2cell"], update_fn, png_files[0])
    save_key_frame(fig, kf["3cell"], update_fn, png_files[1])
    save_key_frame(fig, kf["4cell"], update_fn, png_files[2])
    save_key_frame(fig, kf["final"], update_fn, png_files[3])

    # Phase 4: Render MP4
    print("\nRendering simulation.mp4 ...")
    render_start = time.time()

    anim = FuncAnimation(fig, update_fn, frames=TARGET_FRAMES,
                          interval=1000 // FPS, blit=False)

    writer = FFMpegWriter(fps=FPS, bitrate=3000,
                           extra_args=["-pix_fmt", "yuv420p"])
    anim.save("simulation.mp4", writer=writer, dpi=150,
              savefig_kwargs={"facecolor": BG_COLOR})

    render_time = time.time() - render_start
    print(f"\nRendering done in {render_time:.1f}s")

    # Generate report
    print("\nGenerating report_3.md ...")
    generate_report(precomputed, "simulation.mp4", png_files, render_time)

    print("\n" + "=" * 60)
    print("PROMPT 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
