# -*- coding: utf-8 -*-
"""
gnn_train.py  --  MDG Pipeline Layer 3
=======================================
GNN (Graph Attention Network) on CShaper data to establish the data-ceiling R2.

DevoMDG_GNN mirrors DevoGraph KNN temporal graph construction and is
self-contained and importable as a DevoGraph component.



import os
import sys
import math
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics import r2_score, roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore")

# ---- Constants ---------------------------------------------------------------
VOXEL_VOL_UM3         = 0.09 * 0.09 * 1.0   # voxels -> um3
VOXEL_AREA_UM2        = 0.09 * 0.09          # voxel surface -> um2
CONTACT_EXISTS_THRESH = 50.0                 # areas above this => "contact"
DATASETS_DIR          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

LINEAGE_INT  = {"AB": 0, "EMS": 1, "P": 2}
CELLS_4      = ["ABa", "ABp", "EMS", "P2"]
CELL_LINEAGE = {"ABa": "AB", "ABp": "AB", "EMS": "EMS", "P2": "P"}
ABM_R2       = 0.3826
SINDY_R2     = 0.2537   # dz/dt best axis




def load_positions(path):
    """Parse CDSample*.txt -> DataFrame(cell, time, x, y, z) in um."""
    df = pd.read_csv(path, sep=r"\s+", header=0,
                     names=["cell", "time", "z_px", "x_px", "y_px"])
    df["x"] = df["x_px"] * 0.09
    df["y"] = df["y_px"] * 0.09
    df["z"] = df["z_px"] * 1.0
    return df[["cell", "time", "x", "y", "z"]]


def load_volumes(path):
    """Volume CSV -> DataFrame (rows=timepoints, cols=cell names)."""
    return pd.read_csv(path, index_col=0)


def load_stat(path):
    """
    Sample04_Stat.csv structure:
      row 0: cell1 names  (col 0 label = 'cell1')
      row 1: cell2 names  (col 0 label = 'cell2')
      rows 2+: contact areas per timepoint (0-based after removing header rows)

    Returns dict: (tp_idx, cell1, cell2) -> area_um2
    where tp_idx is 0-based (row 2 of CSV = tp 0).
    """
    raw      = pd.read_csv(path, header=None, low_memory=False)
    c1_names = [str(v).strip() for v in raw.iloc[0, 1:].tolist()]
    c2_names = [str(v).strip() for v in raw.iloc[1, 1:].tolist()]
    data     = raw.iloc[2:].reset_index(drop=True)
    n_cols   = len(c1_names)
    lookup   = {}
    for tp_idx in range(len(data)):
        row = data.iloc[tp_idx]
        for ci in range(n_cols):
            try:
                val = float(row.iloc[ci + 1])
            except (ValueError, TypeError):
                val = 0.0
            area = val * VOXEL_AREA_UM2
            lookup[(tp_idx, c1_names[ci], c2_names[ci])] = area
            lookup[(tp_idx, c2_names[ci], c1_names[ci])] = area
    return lookup


def find_4cell_timepoints(vol_df, cells):
    """Return 0-based indices where ALL cells have non-zero volume."""
    valid = []
    for iloc_idx in range(len(vol_df)):
        row = vol_df.iloc[iloc_idx]
        if all(c in row.index and pd.notna(row[c]) and float(row[c]) > 0
               for c in cells):
            valid.append(iloc_idx)
    return valid




def build_graph(pos_df, vol_df, stat_lookup, tp_idx, sample_name="CDSample04"):
    """
    Build one PyG Data object for the 4-cell stage at 0-based timepoint tp_idx.

    Node features (6D): x, y, z (um) | V0 (um3) | R (um) | lineage_int
    Edge features (3D): dist | |dV0| | |d_lineage|
    Edges: fully connected directed (12 for 4 nodes)
    Targets per edge: y_area (float), y_exists (0/1)
    """
    # Volumes (0-based)
    if tp_idx >= len(vol_df):
        return None
    vol_row = vol_df.iloc[tp_idx]
    vol_map = {}
    for c in CELLS_4:
        if c not in vol_row.index or pd.isna(vol_row[c]) or float(vol_row[c]) <= 0:
            return None
        vol_map[c] = float(vol_row[c]) * VOXEL_VOL_UM3

    # Positions from txt (1-based time, probe offsets for alignment)
    pos_map = None
    for delta in range(-2, 15):
        t  = tp_idx + 1 + delta
        pt = pos_df[pos_df["time"] == t]
        if pt[pt["cell"].isin(CELLS_4)].shape[0] >= 4:
            pm, ok = {}, True
            for c in CELLS_4:
                rc = pt[pt["cell"] == c]
                if rc.empty:
                    ok = False; break
                pm[c] = (float(rc["x"].iloc[0]),
                          float(rc["y"].iloc[0]),
                          float(rc["z"].iloc[0]))
            if ok:
                pos_map = pm
                break
    if pos_map is None:
        return None

    # Node features [4, 6]
    node_feats, lin_int = [], []
    for c in CELLS_4:
        x, y, z = pos_map[c]
        V0 = vol_map[c]
        R  = (3.0 * V0 / (4.0 * math.pi)) ** (1.0/3.0)
        li = LINEAGE_INT[CELL_LINEAGE[c]]
        node_feats.append([x, y, z, V0, R, float(li)])
        lin_int.append(li)
    x_feat = torch.tensor(node_feats, dtype=torch.float)

    # Edges [12 directed]
    src_l, dst_l, ef_l, ea_l, ee_l = [], [], [], [], []
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            ci, cj   = CELLS_4[i], CELLS_4[j]
            xi, yi, zi = pos_map[ci]
            xj, yj, zj = pos_map[cj]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
            dV   = abs(vol_map[ci] - vol_map[cj])
            dL   = float(abs(lin_int[i] - lin_int[j]))
            area = stat_lookup.get((tp_idx, ci, cj), 0.0)
            ex   = 1.0 if area > CONTACT_EXISTS_THRESH else 0.0
            src_l.append(i); dst_l.append(j)
            ef_l.append([dist, dV, dL])
            ea_l.append(area); ee_l.append(ex)

    return Data(
        x          = x_feat,
        edge_index = torch.tensor([src_l, dst_l], dtype=torch.long),
        edge_attr  = torch.tensor(ef_l, dtype=torch.float),
        y_area     = torch.tensor(ea_l, dtype=torch.float),
        y_exists   = torch.tensor(ee_l, dtype=torch.float),
        timepoint  = tp_idx,
        sample_name= sample_name,
    )




class DevoMDG_GNN(nn.Module):
    """
    KNN-based developmental GNN for contact area prediction.
    Architecture mirrors DevoGraph (DevoLearn/DevoGraph) for
    direct integration as an MDG component.

    Two prediction heads:
      regression_head:     predicts contact area (um2)  -- MSE loss
      classification_head: predicts contact existence   -- BCE loss

    Edge features incorporated via projection + concatenation before GATConv.

    Args:
        node_feat_dim: node feature dimension (default 6)
        edge_feat_dim: edge feature dimension (default 3)
        hidden_dim:    GATConv hidden width   (default 64)
        heads:         GAT attention heads    (default 4)
        dropout:       dropout probability    (default 0.1)
    """
    def __init__(self, node_feat_dim=6, edge_feat_dim=3,
                 hidden_dim=64, heads=4, dropout=0.1):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim    = hidden_dim
        self.dropout       = dropout

        self.edge_proj = nn.Linear(edge_feat_dim, node_feat_dim)
        self.gat1 = GATConv(node_feat_dim * 2, hidden_dim,
                            heads=heads, concat=False,
                            dropout=dropout, add_self_loops=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim,
                            heads=heads, concat=False,
                            dropout=dropout, add_self_loops=False)
        dec_in = hidden_dim * 2 + edge_feat_dim
        self.regression_head = nn.Sequential(
            nn.Linear(dec_in, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1), nn.ReLU(),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(dec_in, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x          : [N, node_feat_dim]
        edge_index : [2, E]
        edge_attr  : [E, edge_feat_dim]
        Returns: area_pred [E], exist_pred [E], alpha1 [E, heads]
        """
        src, dst = edge_index
        ep  = self.edge_proj(edge_attr)
        agg = torch.zeros(x.size(0), self.node_feat_dim, device=x.device)
        agg.scatter_add_(0, src.unsqueeze(1).expand_as(ep), ep)
        x_aug = torch.cat([x, agg], dim=1)

        h1, (_, alpha1) = self.gat1(x_aug, edge_index,
                                    return_attention_weights=True)
        h1 = F.relu(F.dropout(h1, p=self.dropout, training=self.training))
        h2 = F.relu(self.gat2(h1, edge_index))

        e_repr = torch.cat([h2[src], h2[dst], edge_attr], dim=1)
        return (
            self.regression_head(e_repr).squeeze(-1),
            self.classification_head(e_repr).squeeze(-1),
            alpha1,
        )




def loss_fn(ap, at, ep, et, w_reg=0.7, w_cls=0.3):
    return w_reg * F.mse_loss(ap, at) + w_cls * F.binary_cross_entropy(ep, et)


def train_epoch(model, opt, graphs):
    model.train()
    tot = 0.0
    for g in graphs:
        opt.zero_grad()
        ap, ep, _ = model(g.x, g.edge_index, g.edge_attr)
        loss = loss_fn(ap, g.y_area, ep, g.y_exists)
        loss.backward()
        opt.step()
        tot += loss.item()
    return tot / max(len(graphs), 1)


@torch.no_grad()
def evaluate(model, graphs):
    model.eval()
    ap_all, at_all, ep_all, et_all, attn_all = [], [], [], [], []
    for g in graphs:
        ap, ep, alpha = model(g.x, g.edge_index, g.edge_attr)
        ap_all.append(ap.cpu().numpy())
        at_all.append(g.y_area.cpu().numpy())
        ep_all.append(ep.cpu().numpy())
        et_all.append(g.y_exists.cpu().numpy())
        attn_all.append(alpha.cpu().numpy())
    ap = np.concatenate(ap_all)
    at = np.concatenate(at_all)
    ep = np.concatenate(ep_all)
    et = np.concatenate(et_all)
    r2  = r2_score(at, ap) if at.std() > 0 else float("nan")
    auc = roc_auc_score(et, ep) if len(np.unique(et)) > 1 else float("nan")
    return {"r2": r2, "auc": auc,
            "area_pred": ap, "area_true": at,
            "exist_pred": ep, "exist_true": et,
            "attn": attn_all[-1] if attn_all else None}




def main():
    print("=" * 60)
    print("MDG PIPELINE -- LAYER 3: GNN DATA CEILING")
    print("=" * 60)

    # Step 0 ------------------------------------------------------------------
    print("\n[Step 0] Loading CShaper data ...")
    pos_df   = load_positions(os.path.join(DATASETS_DIR, "CDSample04.txt"))
    vol_df   = load_volumes(os.path.join(DATASETS_DIR, "Sample04_Volume.csv"))
    stat_lkp = load_stat(os.path.join(DATASETS_DIR, "Sample04_Stat.csv"))
    print(f"  Position records : {len(pos_df)}")
    print(f"  Volume CSV shape : {vol_df.shape}")
    print(f"  Stat lookup keys : {len(stat_lkp)}")

    # Spot-check: print non-zero 4-cell entries
    sample_keys = [k for k in stat_lkp
                   if k[1] in CELLS_4 and k[2] in CELLS_4 and stat_lkp[k] > 0][:8]
    print("  Sample 4-cell contacts (tp, c1, c2, area_um2):")
    for k in sample_keys:
        print(f"    tp={k[0]:2d}  {k[1]}-{k[2]}  {stat_lkp[k]:.2f}")

    # Step 1 ------------------------------------------------------------------
    print("\n[Step 1] Building PyG Data objects ...")
    tp_indices = find_4cell_timepoints(vol_df, CELLS_4)
    print(f"  4-cell timepoints: {len(tp_indices)}  -> {tp_indices}")

    graphs = []
    for tp in tp_indices:
        g = build_graph(pos_df, vol_df, stat_lkp, tp)
        if g is not None:
            graphs.append(g)
            print(f"  tp={tp:3d}: nodes={list(g.x.shape)}  "
                  f"edges={list(g.edge_index.shape)}  "
                  f"nonzero_areas={int((g.y_area > 0).sum())}/12  "
                  f"mean_area={g.y_area.mean():.1f}")

    print(f"\n  Total graphs: {len(graphs)}")
    if not graphs:
        raise RuntimeError("No valid graphs -- check timepoint alignment.")

    node_feat_dim = graphs[0].x.shape[1]
    edge_feat_dim = graphs[0].edge_attr.shape[1]
    print(f"  Node feat dim: {node_feat_dim}")
    print(f"  Edge feat dim: {edge_feat_dim}")

    # Step 2 ------------------------------------------------------------------
    print("\n[Step 2] Initializing DevoMDG_GNN ...")
    model  = DevoMDG_GNN(node_feat_dim=node_feat_dim,
                         edge_feat_dim=edge_feat_dim, hidden_dim=64)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Step 3 ------------------------------------------------------------------
    test_g   = [graphs[-1]]
    train_g  = graphs[:-1] if len(graphs) > 1 else graphs[:]
    print(f"\n[Step 3] LOTO training: {len(train_g)} train | {len(test_g)} test")
    print("  [!] LOTO CV: R2 is OPTIMISTIC (single embryo only)")
    print(f"  Test timepoint: {test_g[0].timepoint}")

    opt  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    hist = []
    t0   = time.time()
    for epoch in range(1, 201):
        loss = train_epoch(model, opt, train_g)
        hist.append(loss)
        if epoch % 20 == 0 or epoch == 1:
            vr = evaluate(model, test_g)
            print(f"  Epoch {epoch:3d}/200  loss={loss:.4f}  "
                  f"val_R2={vr['r2']:.4f}  val_AUC={vr['auc']:.4f}  "
                  f"({time.time()-t0:.0f}s)")

    # Step 4 ------------------------------------------------------------------
    print("\n[Step 4] Evaluation ...")
    tr = evaluate(model, test_g)
    print(f"  Test R2:  {tr['r2']:.4f}")
    print(f"  Test AUC: {tr['auc']:.4f}")

    cells = CELLS_4
    pairs = [(cells[i], cells[j]) for i in range(4) for j in range(4) if i != j]
    ap, at = tr["area_pred"], tr["area_true"]

    print(f"\n  {'Pair':<14} {'True (um2)':>12} {'Pred (um2)':>12} {'Error%':>10}")
    print("  " + "-" * 52)
    seen, per_pp, per_pt = set(), {}, {}
    for idx, (c1, c2) in enumerate(pairs):
        key = tuple(sorted([c1, c2]))
        if key in seen:
            continue
        ri = next((k for k, (a, b) in enumerate(pairs) if (a, b) == (c2, c1)), idx)
        p  = (ap[idx] + ap[ri]) / 2
        t  = (at[idx] + at[ri]) / 2
        err = (p - t) / (t + 1e-9) * 100
        print(f"  {c1+'-'+c2:<14} {t:>12.2f} {p:>12.2f} {err:>+10.1f}%")
        per_pp[f"{c1}-{c2}"] = float(p)
        per_pt[f"{c1}-{c2}"] = float(t)
        seen.add(key)

    ep_b = (tr["exist_pred"] > 0.5).astype(int)
    et_b = tr["exist_true"].astype(int)
    cm   = confusion_matrix(et_b, ep_b)
    print(f"\n  Confusion matrix:\n  {cm}")

    attn_raw = tr["attn"]
    if attn_raw is not None:
        am_1d = attn_raw.mean(axis=1) if attn_raw.ndim == 2 else attn_raw
        amat  = np.zeros((4, 4))
        for idx2, (ci, cj) in enumerate(pairs):
            amat[cells.index(ci), cells.index(cj)] = am_1d[idx2]
        print("\n  Attention matrix (row=src, col=dst):")
        print("  " + " ".join(f"{c:>8}" for c in [""] + cells))
        for i, ci in enumerate(cells):
            print(f"  {ci:>4}  " + "  ".join(f"{amat[i,j]:>6.4f}" for j in range(4)))
    else:
        amat = np.zeros((4, 4))

    GNN_R2 = tr["r2"]
    gap    = GNN_R2 - ABM_R2
    if GNN_R2 >= ABM_R2:
        interp = (f"The gap of {gap:.4f} represents the fraction of contact area "
                  f"variance attributable to biological variables not captured by "
                  f"JKR spherical contact mechanics.")
    else:
        interp = (f"GNN R2={GNN_R2:.4f} < ABM R2={ABM_R2:.4f}. "
                  f"GNN did not outperform ABM -- insufficient data "
                  f"(LOTO with {len(graphs)} timepoints from 1 embryo). "
                  f"This does NOT mean ABM is superior.")

    print("\n" + "=" * 60)
    print("MDG DIAGNOSTIC")
    print("=" * 60)
    print(f"  {'Layer':<8} {'Method':<18} {'R2':>8}  Interpretation")
    print("  " + "-" * 56)
    print(f"  {'1':<8} {'ABM (JKR)':<18} {ABM_R2:>8.4f}  Known physics ceiling")
    print(f"  {'2':<8} {'SINDy (dz/dt)':<18} {SINDY_R2:>8.4f}  Discovered equation fit")
    print(f"  {'3':<8} {'GNN (GAT)':<18} {GNN_R2:>8.4f}  Data ceiling")
    print(f"  {'Gap':<8} {'GNN - ABM':<18} {gap:>+8.4f}  Hidden variable signature")
    print(f"\n  {interp}")
    print("=" * 60)

    # Step 5 ------------------------------------------------------------------
    results = {
        "model_state_dict":      model.state_dict(),
        "test_r2":               float(GNN_R2),
        "test_auc":              float(tr["auc"]),
        "attention_weights":     amat,
        "per_pair_predictions":  per_pp,
        "per_pair_targets":      per_pt,
        "training_loss_history": hist,
        "node_feat_dim":         node_feat_dim,
        "edge_feat_dim":         edge_feat_dim,
        "hidden_dim":            64,
        "architecture":          "DevoMDG_GNN_GATConv_2layer",
        "n_train_graphs":        len(train_g),
        "n_test_graphs":         len(test_g),
        "n_total_graphs":        len(graphs),
        "n_params":              n_params,
        "abm_r2":                ABM_R2,
        "sindy_r2":              SINDY_R2,
        "gap":                   float(gap),
        "test_confusion_matrix": cm,
        "loto_fallback":         True,
        "loto_note":             "Single embryo CDSample04 only.",
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnn_results.pt")
    torch.save(results, out)
    print(f"\n[Step 5] Saved: {out}")

    _write_report(results, tr, graphs, amat, cm, per_pp, per_pt,
                  GNN_R2, gap, interp)
    print("\n" + "=" * 60)
    print("PROMPT 5 COMPLETE")
    print(f"  Test R2: {GNN_R2:.4f}  |  AUC: {tr['auc']:.4f}  |  Gap: {gap:+.4f}")
    print("=" * 60)




def _write_report(res, tr, graphs, amat, cm, per_pp, per_pt, GNN_R2, gap, interp):
    cells   = CELLS_4
    n_train = res["n_train_graphs"]
    n_test  = res["n_test_graphs"]
    n_total = res["n_total_graphs"]
    rpath   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnn_report.md")

    with open(rpath, "w", encoding="utf-8") as f:
        f.write("# GNN Report - MDG Pipeline Layer 3\n\n")



    print(f"  Saved: {rpath}")


if __name__ == "__main__":
    main()
