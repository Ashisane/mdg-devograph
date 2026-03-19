"""
data_loader.py — Dataset loader for C. elegans biophysical ABM.

Loads cell volumes (V₀) and contact areas from the experimental dataset.
Volumes are extracted from Sample04_Volume.csv.
Contact areas are extracted from Sample04_Stat.csv.

All volumes are averaged over the 4-cell stage (timepoints where ABa, ABp,
EMS, and P2 are all simultaneously present).

Voxel conversion: 0.09 × 0.09 × 1.0 μm³/voxel = 0.0081 μm³/voxel
"""

import os
import numpy as np
import pandas as pd

# datasets/ lives at the project root (one level above mdg/)
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")

VOXEL_VOLUME_UM3 = 0.09 * 0.09 * 1.0  # μm³ per voxel

FOUR_CELL_NAMES  = ["ABa", "ABp", "EMS", "P2"]
EIGHT_CELL_NAMES = ["ABar", "ABal", "ABpr", "ABpl", "MS", "E", "C", "P3"]


def load_volumes(data_dir: str = _DATA_DIR) -> dict:
    """
    Load V₀ per cell identity, averaged over the 4-cell stage.

    Returns:
        dict: {cell_name: V0_um3} for ABa, ABp, EMS, P2.
              V0 is in μm³.
    """
    vol_path = os.path.join(data_dir, "Sample04_Volume.csv")
    df = pd.read_csv(vol_path)

    four_cell_mask = pd.Series(True, index=df.index)
    for cell in FOUR_CELL_NAMES:
        four_cell_mask &= df[cell].notna()

    four_cell_tps = df[four_cell_mask].index.tolist()
    if len(four_cell_tps) == 0:
        raise ValueError("No timepoints found where all 4 cells are present!")

    volumes = {}
    for cell in FOUR_CELL_NAMES:
        voxel_vals = [df.loc[tp, cell] for tp in four_cell_tps]
        mean_voxels = np.mean(voxel_vals)
        volumes[cell] = float(mean_voxels * VOXEL_VOLUME_UM3)

    return volumes


def load_volumes_8cell(data_dir: str = _DATA_DIR) -> dict:
    """
    Load V0 per cell identity averaged over the 8-cell stage.

    8-cell stage = timepoints where all 8 daughters are simultaneously present.
    Falls back to None for cells not found (caller must handle with mother.V0/2).

    Returns:
        dict: {cell_name: V0_um3} for the 8 daughters.
    """
    vol_path = os.path.join(data_dir, "Sample04_Volume.csv")
    df = pd.read_csv(vol_path)

    eight_cell_mask = pd.Series(True, index=df.index)
    for cell in EIGHT_CELL_NAMES:
        if cell in df.columns:
            eight_cell_mask &= df[cell].notna()

    eight_cell_tps = df[eight_cell_mask].index.tolist()
    if len(eight_cell_tps) == 0:
        raise ValueError("No timepoints found where all 8 cells are present!")

    volumes = {}
    for cell in EIGHT_CELL_NAMES:
        if cell not in df.columns:
            print(f"  [WARNING] {cell} not found in dataset — will use mother.V0/2 fallback")
            volumes[cell] = None
        else:
            vals = [df.loc[tp, cell] for tp in eight_cell_tps]
            volumes[cell] = float(np.mean(vals) * VOXEL_VOLUME_UM3)

    return volumes


def load_contact_areas(data_dir: str = _DATA_DIR) -> dict:
    """
    Load contact areas per named pair at the 4-cell stage.

    The Stat CSV has a special structure:
      - Column headers = cell1 names (with .N suffixes for multiple partners)
      - Row 0 values   = cell2 partner names
      - Rows 1+        = contact area values per timepoint

    Returns:
        dict: {(cell1, cell2): mean_contact_area} where cell1 < cell2
              alphabetically. Areas are in raw dataset units (voxel surface
              area units).
    """
    stat_path = os.path.join(data_dir, "Sample04_Stat.csv")
    df = pd.read_csv(stat_path, low_memory=False)

    cell2_header = df.iloc[0]
    data_rows = df.iloc[1:]
    pair_values = {}

    for col in df.columns[1:]:
        base_name = col.split('.')[0]
        if base_name in FOUR_CELL_NAMES:
            partner = cell2_header[col]
            if isinstance(partner, str) and partner in FOUR_CELL_NAMES:
                pair_key = tuple(sorted([base_name, partner]))

                if pair_key not in pair_values:
                    pair_values[pair_key] = []

                # 4-cell stage rows have '1' or '2' in the first column
                for idx in data_rows.index:
                    tp = data_rows.loc[idx, 'cell1']
                    if tp in ['1', '2']:
                        val = data_rows.loc[idx, col]
                        try:
                            pair_values[pair_key].append(float(val))
                        except (ValueError, TypeError):
                            pass

    contacts = {}
    for pair, vals in pair_values.items():
        if vals:
            contacts[pair] = float(np.mean(vals))

    return contacts


def verify_constraints(data_dir: str = _DATA_DIR):
    """
    Print biological constraint checks:
      1. Volume ordering: V_ABa ≈ V_ABp > V_EMS > V_P2
      2. ABa-P2 contact absent
    """
    print("=" * 60)
    print("BIOLOGICAL CONSTRAINT VERIFICATION")
    print("=" * 60)

    volumes = load_volumes(data_dir)
    print("\n--- V₀ per cell (μm³) ---")
    for cell in FOUR_CELL_NAMES:
        print(f"  {cell}: {volumes[cell]:.2f} μm³")

    v_aba = volumes['ABa']
    v_abp = volumes['ABp']
    v_ems = volumes['EMS']
    v_p2 = volumes['P2']

    ratio = abs(v_aba - v_abp) / max(v_aba, v_abp)
    check1 = ratio < 0.15
    print(f"\n  V_ABa ≈ V_ABp (ratio diff = {ratio:.3f}): "
          f"{'PASS' if check1 else 'FAIL'}")

    check2 = min(v_aba, v_abp) > v_ems
    print(f"  min(V_ABa, V_ABp) > V_EMS: {'PASS' if check2 else 'FAIL'}")

    check3 = v_ems > v_p2
    print(f"  V_EMS > V_P2: {'PASS' if check3 else 'FAIL'}")

    contacts = load_contact_areas(data_dir)
    print("\n--- Contact areas (raw units) ---")
    for pair, area in sorted(contacts.items()):
        print(f"  {pair[0]}-{pair[1]}: {area:.2f}")

    aba_p2_key = tuple(sorted(['ABa', 'P2']))
    check4 = aba_p2_key not in contacts
    print(f"\n  ABa-P2 contact absent: {'PASS' if check4 else 'FAIL'}")

    print("\n" + "=" * 60)
    all_pass = check1 and check2 and check3 and check4
    print(f"ALL CONSTRAINTS: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    print("Loading volumes...")
    vols = load_volumes()
    for k, v in vols.items():
        print(f"  {k}: {v:.2f} μm³")

    print("\nLoading contact areas...")
    contacts = load_contact_areas()
    for pair, area in sorted(contacts.items()):
        print(f"  {pair[0]}-{pair[1]}: {area:.2f}")

    print("\nRunning constraint checks...")
    verify_constraints()
