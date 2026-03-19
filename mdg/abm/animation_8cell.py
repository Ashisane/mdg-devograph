"""
animation_8cell.py — Animated visualisation of the 8-cell C. elegans simulation.

Three-panel layout:
  - Left 60%  : 3D cell view (ellipsoids, eggshell, contact edges)
  - Right 40% : DevoGraph
  - Bottom 20%: Energy timeline + contact area traces

Pre-computes all data before FuncAnimation — zero torch/file operations inside update().

Usage (from project root):
    python mdg/abm/animation_8cell.py
"""

import sys
import os
import math
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import torch

_ABM_DIR  = os.path.dirname(os.path.abspath(__file__))
_MDG_DIR  = os.path.dirname(_ABM_DIR)
_PROJ_DIR = os.path.dirname(_MDG_DIR)
for _d in [_ABM_DIR, _MDG_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from physics import SHELL_A, SHELL_B, SHELL_C

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
LINEAGE_COLORS = {
    # 8-cell AB daughters
    "ABar": "#60A5FA", "ABal": "#60A5FA",
    "ABpr": "#60A5FA", "ABpl": "#60A5FA",
    # 8-cell EMS daughters
    "MS": "#FB923C", "E": "#FB923C",
    # 8-cell P daughters
    "C":  "#F87171", "P3": "#F87171",
    # pre-division 4-cell
    "ABa": "#93C5FD", "ABp": "#93C5FD",
    "EMS": "#FDBA74",
    "P2":  "#FCA5A5",
    # 2-cell
    "AB":  "#BFDBFE", "P1": "#FECACA",
}

BG_COLOR     = "#0D1117"
SHELL_COLOR  = "#334155"

# ---------------------------------------------------------------------------
# DevoGraph node layout (fixed positions)
# ---------------------------------------------------------------------------
NODE_POS_8CELL = {
    "ABar": (0.6, 0.8), "ABal": (0.2, 0.8),
    "ABpr": (0.8, 0.6), "ABpl": (0.1, 0.6),
    "MS":   (0.4, 0.3), "E":    (0.3, 0.15),
    "C":    (0.7, 0.3), "P3":   (0.8, 0.15),
}
NODE_POS_4CELL = {
    "ABa": (0.3, 0.75), "ABp": (0.7, 0.75),
    "EMS": (0.3, 0.3),  "P2":  (0.7, 0.3),
}
NODE_POS_2CELL = {
    "AB": (0.35, 0.6), "P1": (0.65, 0.6),
}
NODE_POS_3CELL = {
    "ABa": (0.2, 0.75), "ABp": (0.5, 0.75), "P1": (0.8, 0.5),
}

# The 9 expected contacts (used for contact-area trace panel)
EXPECTED_CONTACTS = [
    ("ABar", "ABpr"), ("ABal", "ABpl"),
    ("ABar", "MS"),   ("ABal", "MS"),
    ("ABpr", "C"),    ("ABpl", "C"),
    ("MS", "E"),      ("MS", "C"),
    ("E",  "P3"),
]
EXPECTED_COLORS = [
    "#60A5FA", "#60A5FA",
    "#60A5FA", "#60A5FA",
    "#60A5FA", "#60A5FA",
    "#FB923C", "#FB923C",
    "#F87171",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stage_label(n_cells):
    return f"{n_cells}-cell"


def ellipsoid_mesh(pos, a, b, c, nu=20, nv=10):
    u = np.linspace(0, 2 * math.pi, nu)
    v = np.linspace(0, math.pi, nv)
    x = a * np.outer(np.cos(u), np.sin(v)) + pos[0]
    y = b * np.outer(np.sin(u), np.sin(v)) + pos[1]
    z = c * np.outer(np.ones_like(u), np.cos(v)) + pos[2]
    return x, y, z


def eggshell_wireframe(ax3d, n_lat=24, n_lon=12):
    u = np.linspace(0, 2 * math.pi, n_lat)
    v = np.linspace(0, math.pi, n_lon)
    for ui in u:
        xs = SHELL_A * np.cos(ui) * np.sin(v)
        ys = SHELL_B * np.sin(ui) * np.sin(v)
        zs = SHELL_C * np.cos(v)
        ax3d.plot(xs, ys, zs, color=SHELL_COLOR, alpha=0.2, lw=0.4)
    for vi in v:
        xs = SHELL_A * np.cos(u) * np.sin(vi)
        ys = SHELL_B * np.sin(u) * np.sin(vi)
        zs = SHELL_C * np.cos(vi) * np.ones_like(u)
        ax3d.plot(xs, ys, zs, color=SHELL_COLOR, alpha=0.2, lw=0.4)


def get_node_pos(cells_in_frame):
    names = {c["identity"] for c in cells_in_frame}
    # Try to match to the appropriate layout
    eight = {"ABar", "ABal", "ABpr", "ABpl", "MS", "E", "C", "P3"}
    four  = {"ABa", "ABp", "EMS", "P2"}
    two   = {"AB", "P1"}
    three = {"ABa", "ABp", "P1"}

    if names & eight:
        base = dict(NODE_POS_8CELL)
        # also add any transitional cells still present
        for name, pos in NODE_POS_4CELL.items():
            if name in names:
                base[name] = pos
        return base
    elif names >= four:
        return NODE_POS_4CELL
    elif names >= three:
        return NODE_POS_3CELL
    else:
        return NODE_POS_2CELL


# ---------------------------------------------------------------------------
# Pre-compute all frame data
# ---------------------------------------------------------------------------

def precompute(trajectory):
    """
    Walk the trajectory and extract everything needed per frame.
    Returns list of dicts (one per frame).
    """
    frames = []
    for frm in trajectory:
        cells = frm["cells"]
        contacts = frm.get("contacts", {})
        E_total  = frm.get("E_total", 0.0)

        cell_data = []
        for c in cells:
            pos = np.array(c["position"])
            R   = c["R"]
            if "axes" in c:
                a, b, cc = c["axes"]
            else:
                a = b = cc = R
            col = LINEAGE_COLORS.get(c["identity"], "#CCCCCC")
            cell_data.append({
                "identity": c["identity"],
                "pos": pos,
                "a": a, "b": b, "c": cc,
                "R": R,
                "color": col,
            })

        # Build contact edges (area > 30 μm²)
        contact_edges = []
        cell_map = {c["identity"]: c for c in cells}
        for key, area in contacts.items():
            if area > 30.0:
                parts = key.split("-")
                if len(parts) == 2 and parts[0] in cell_map and parts[1] in cell_map:
                    p0 = np.array(cell_map[parts[0]]["position"])
                    p1 = np.array(cell_map[parts[1]]["position"])
                    c0 = LINEAGE_COLORS.get(parts[0], "#CCCCCC")
                    c1 = LINEAGE_COLORS.get(parts[1], "#CCCCCC")
                    avg_col = blend_color(c0, c1)
                    lw = 1.0 + area / 500.0
                    contact_edges.append((p0, p1, avg_col, lw))

        # Expected contact areas for trace panel
        exp_areas = {}
        for (a_id, b_id) in EXPECTED_CONTACTS:
            k1 = f"{a_id}-{b_id}"
            k2 = f"{b_id}-{a_id}"
            area = contacts.get(k1, contacts.get(k2, 0.0))
            exp_areas[(a_id, b_id)] = area

        frames.append({
            "t": frm["t"],
            "n_cells": frm["n_cells"],
            "cells": cell_data,
            "contact_edges": contact_edges,
            "E_total": E_total,
            "exp_areas": exp_areas,
            "node_pos": get_node_pos(cells),
            "contacts": contacts,
        })

    return frames


def blend_color(hex1, hex2):
    def h2r(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    r1 = h2r(hex1); r2 = h2r(hex2)
    avg = tuple((a + b) / 2 for a, b in zip(r1, r2))
    return avg


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

def build_figure():
    fig = plt.figure(figsize=(18, 10), facecolor=BG_COLOR)

    # Outer grid: top 80% | bottom 20%
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[4, 1],
                              hspace=0.05)
    # Top: left 60% (3D) | right 40% (DevoGraph)
    top_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                              width_ratios=[3, 2], wspace=0.02)
    # Bottom: two subplots
    bot_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                              wspace=0.3)

    ax3d   = fig.add_subplot(top_gs[0], projection='3d')
    ax_devo = fig.add_subplot(top_gs[1])
    ax_eng  = fig.add_subplot(bot_gs[0])
    ax_con  = fig.add_subplot(bot_gs[1])

    for ax in [ax_devo, ax_eng, ax_con]:
        ax.set_facecolor(BG_COLOR)
    ax3d.set_facecolor(BG_COLOR)
    ax3d.grid(False)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("none")

    return fig, ax3d, ax_devo, ax_eng, ax_con


# ---------------------------------------------------------------------------
# Main animation function
# ---------------------------------------------------------------------------

def make_animation(pt_path, out_mp4, frame_4cell_png, frame_8cell_png):
    print("Loading simulation data...")
    data = torch.load(pt_path, weights_only=False)

    traj_4 = data["trajectory_4cell"]
    traj_8 = data["trajectory_8cell"]
    trajectory = traj_4 + traj_8

    print(f"  4-cell frames: {len(traj_4)}")
    print(f"  8-cell frames: {len(traj_8)}")
    print(f"  Total frames:  {len(trajectory)}")

    print("Pre-computing frame data...")
    frames = precompute(trajectory)
    N = len(frames)

    # Time series for energy panel
    frame_ts   = [f["t"] for f in frames]
    energies   = [f["E_total"] for f in frames]
    min_e = min(e for e in energies if e > 0 and math.isfinite(e))
    max_e = max(e for e in energies if math.isfinite(e))

    # Contact traces — build per-frame arrays
    contact_series = {pair: np.zeros(N) for pair in EXPECTED_CONTACTS}
    for fi, frm in enumerate(frames):
        for pair in EXPECTED_CONTACTS:
            contact_series[pair][fi] = frm["exp_areas"].get(pair, 0.0)

    # Division event frames
    div_frames = []
    for i in range(1, N):
        if frames[i]["n_cells"] > frames[i-1]["n_cells"]:
            div_frames.append(i)

    print(f"  Division events at frames: {div_frames}")

    # Build figure
    fig, ax3d, ax_devo, ax_eng, ax_con = build_figure()

    # Static eggshell (drawn once)
    eggshell_wireframe(ax3d)
    ax3d.set_xlim(-SHELL_A, SHELL_A)
    ax3d.set_ylim(-SHELL_B, SHELL_B)
    ax3d.set_zlim(-SHELL_C, SHELL_C)
    ax3d.view_init(elev=25, azim=0)
    ax3d.set_box_aspect([SHELL_A, SHELL_B, SHELL_C])

    # Energy panel setup (static y scale)
    ax_eng.set_facecolor(BG_COLOR)
    ax_eng.tick_params(colors='white', labelsize=6)
    ax_eng.set_xlabel("t (step)", color='white', fontsize=7)
    ax_eng.set_ylabel("E_total", color='white', fontsize=7)
    for spine in ax_eng.spines.values():
        spine.set_edgecolor('#334155')

    # Contact panel setup
    ax_con.set_facecolor(BG_COLOR)
    ax_con.tick_params(colors='white', labelsize=6)
    ax_con.set_xlabel("t (step)", color='white', fontsize=7)
    ax_con.set_ylabel("Contact area (μm²)", color='white', fontsize=7)
    for spine in ax_con.spines.values():
        spine.set_edgecolor('#334155')

    # DevoGraph axes style
    ax_devo.set_facecolor(BG_COLOR)
    ax_devo.set_xlim(0, 1); ax_devo.set_ylim(0, 1)
    ax_devo.axis('off')
    ax_devo.set_title("DevoGraph (Mechanistic)", color='white', fontsize=9, pad=4)

    # -----------------------------------------------------------------------
    # Objects whose data changes per frame
    cell_surfs   = []
    cell_labels  = []
    edge_lines   = []
    devo_nodes   = {}      # identity -> scatter
    devo_edges   = []
    stage_text   = ax3d.text2D(0.02, 0.95, "", transform=ax3d.transAxes,
                               color='white', fontsize=9, va='top')
    eng_line,    = ax_eng.plot([], [], color='#4ADE80', lw=1)
    eng_cursor,  = ax_eng.plot([], [], 'o', color='white', ms=3)
    div_vlines   = []

    # Draw energy history lines (full range)
    ax_eng.set_xlim(min(frame_ts), max(frame_ts))
    ax_eng.set_ylim(min_e * 0.9, max_e * 1.1)

    # Pre-draw vertical markers for divisions
    for df_idx in div_frames:
        vl = ax_eng.axvline(frames[df_idx]["t"], color='#F87171',
                            alpha=0.6, lw=0.8, ls='--')
        ax_con.axvline(frames[df_idx]["t"], color='#F87171',
                       alpha=0.6, lw=0.8, ls='--')

    # Pre-draw contact traces (all 9)
    con_lines = {}
    for (pair, col) in zip(EXPECTED_CONTACTS, EXPECTED_COLORS):
        label = f"{pair[0]}-{pair[1]}"
        ln, = ax_con.plot([], [], color=col, lw=0.8, alpha=0.8, label=label)
        con_lines[pair] = ln
    ax_con.set_xlim(min(frame_ts), max(frame_ts))
    ax_con.set_ylim(0, max(contact_series[p].max() for p in EXPECTED_CONTACTS) * 1.1 + 1)
    ax_con.legend(fontsize=4, loc='upper left', facecolor=BG_COLOR,
                  labelcolor='white', framealpha=0.3)

    # -----------------------------------------------------------------------
    def init():
        return []

    def update(fi):
        frm = frames[fi]
        azim = fi * 0.6

        # Clear variable elements
        for s in cell_surfs:
            s.remove()
        cell_surfs.clear()
        for lbl in cell_labels:
            lbl.remove()
        cell_labels.clear()
        for ln in edge_lines:
            ln.remove()
        edge_lines.clear()

        # Clear DevoGraph nodes/edges
        for sc in devo_nodes.values():
            sc.remove()
        devo_nodes.clear()
        for ln in devo_edges:
            ln.remove()
        devo_edges.clear()

        # --- 3D cells ---
        ax3d.view_init(elev=25, azim=azim)
        for cd in frm["cells"]:
            x, y, z = ellipsoid_mesh(cd["pos"], cd["a"], cd["b"], cd["c"])
            surf = ax3d.plot_surface(x, y, z, color=cd["color"],
                                     alpha=0.75, linewidth=0, antialiased=False,
                                     shade=True)
            cell_surfs.append(surf)
            lbl = ax3d.text(cd["pos"][0], cd["pos"][1],
                            cd["pos"][2] + cd["a"] + 0.5,
                            cd["identity"], color='white', fontsize=7,
                            ha='center')
            cell_labels.append(lbl)

        # --- Contact edges ---
        for (p0, p1, col, lw) in frm["contact_edges"]:
            ln, = ax3d.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                            color=col, lw=lw, alpha=0.6)
            edge_lines.append(ln)

        # Stage label
        stage_text.set_text(stage_label(frm["n_cells"]))

        # --- DevoGraph ---
        node_pos = frm["node_pos"]
        present  = {c["identity"] for c in frm["cells"]}
        contacts_raw = frm["contacts"]

        # Draw edges first (no overlap with nodes)
        for key, area in contacts_raw.items():
            if area > 30.0:
                parts = key.split("-")
                if len(parts) == 2:
                    id1, id2 = parts[0], parts[1]
                    if id1 in node_pos and id2 in node_pos and \
                       id1 in present and id2 in present:
                        x0, y0 = node_pos[id1]
                        x1, y1 = node_pos[id2]
                        ln, = ax_devo.plot([x0, x1], [y0, y1],
                                           color='#64748B', lw=0.8, alpha=0.5,
                                           zorder=1)
                        devo_edges.append(ln)

        # Draw nodes
        for cd in frm["cells"]:
            ident = cd["identity"]
            if ident not in node_pos:
                continue
            nx, ny = node_pos[ident]
            sz = 80 + cd["R"] * 8
            sc = ax_devo.scatter([nx], [ny], s=sz,
                                 color=cd["color"], zorder=3, alpha=0.9)
            devo_nodes[ident] = sc
            ax_devo.text(nx, ny + 0.04, ident, color='white',
                         fontsize=6, ha='center', va='bottom', zorder=4)

        # --- Energy timeline ---
        ts_so_far = frame_ts[:fi+1]
        en_so_far = energies[:fi+1]
        eng_line.set_data(ts_so_far, en_so_far)
        eng_cursor.set_data([frame_ts[fi]], [energies[fi]])

        # --- Contact traces ---
        for pair, ln in con_lines.items():
            ln.set_data(frame_ts[:fi+1], contact_series[pair][:fi+1])

        return cell_surfs + cell_labels + edge_lines

    print("Building FuncAnimation...")
    ani = FuncAnimation(fig, update, frames=N, init_func=init,
                        interval=33, blit=False)

    # Save 4-cell final frame
    idx_4cel = len(traj_4) - 1
    update(idx_4cel)
    fig.savefig(frame_4cell_png, dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
    print(f"Saved {frame_4cell_png}")

    # Save 8-cell final frame
    update(N - 1)
    fig.savefig(frame_8cell_png, dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
    print(f"Saved {frame_8cell_png}")

    # Render video
    print(f"Rendering {out_mp4} ...")
    writer = FFMpegWriter(fps=30, metadata={"title": "8-Cell C. elegans ABM"},
                          extra_args=["-pix_fmt", "yuv420p"])
    ani.save(out_mp4, writer=writer, dpi=150,
             progress_callback=lambda i, n: print(f"\r  Frame {i}/{n}", end="") if i % 50 == 0 else None)
    print(f"\nSaved {out_mp4}")
    plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PT_PATH   = os.path.join(_PROJ_DIR, "results", "simulation_results_8cell.pt")
    OUT_MP4   = os.path.join(_PROJ_DIR, "results", "simulation_8cell.mp4")
    OUT_4C    = os.path.join(_PROJ_DIR, "results", "images", "frame_4cell_final.png")
    OUT_8C    = os.path.join(_PROJ_DIR, "results", "images", "frame_8cell_final.png")

    if not os.path.exists(PT_PATH):
        print(f"ERROR: {PT_PATH} not found. Run simulation_8cell.py first.")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT_4C), exist_ok=True)
    make_animation(PT_PATH, OUT_MP4, OUT_4C, OUT_8C)
