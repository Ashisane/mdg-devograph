# GNN Report - MDG Pipeline Layer 3

## 1. Data Summary

| Item | Value |
|------|-------|
| Dataset | CDSample04 (LOTO cross-validation) |
| Training graphs | 1 |
| Test graphs | 1 |
| Total 4-cell timepoints | 2 |
| Node feature dim | 6 |
| Edge feature dim | 3 |
| Nodes per graph | 4 (ABa, ABp, EMS, P2) |
| Edges per graph | 12 (fully connected, directed) |

> **[!] WARNING - LOTO Fallback Active.**  
> `DevoLearn/data-science-demos` and `cao13jf/CShaper` contain no CDSample CSV data.  
> Only CDSample04 is available.  
> **R^2 is OPTIMISTIC** - within-sample LOTO, not cross-embryo.

### Node Features (6D)
| # | Feature | Source |
|---|---------|--------|
| 0 | x (um) | CDSample04.txt, px*0.09 |
| 1 | y (um) | CDSample04.txt, px*0.09 |
| 2 | z (um) | CDSample04.txt, px*1.0 |
| 3 | V0 (um3) | Sample04_Volume.csv * 0.0081 |
| 4 | R (um) | (3V0/4pi)^(1/3) |
| 5 | lineage_int | AB=0, EMS=1, P=2 |

### Edge Features (3D)
| # | Feature |
|---|--------|
| 0 | Euclidean distance (um) |
| 1 | |dV0| volume asymmetry (um3) |
| 2 | |d_lineage| lineage distance |

## 2. Model Architecture

```
DevoMDG_GNN -- mirrors DevoGraph KNN temporal graph construction
  edge_proj:           Linear(3 -> 6)
  gat1:                GATConv(12 -> 64, heads=4, concat=False)
  gat2:                GATConv(64 -> 64, heads=4, concat=False)
  regression_head:     Linear(131->64) -> ReLU -> Linear(64->1) -> ReLU
  classification_head: Linear(131->64) -> ReLU -> Linear(64->1) -> Sigmoid

Total trainable parameters: 37,658
```

## 3. Test Results

| Metric | Value |
|--------|-------|
| R^2 (contact area regression) | **-0.0212** |
| AUC-ROC (contact existence) | **1.0000** |

### Per-Pair Prediction Table

| Pair | Measured (um2) | Predicted (um2) | Error % |
|------|---------------|-----------------|--------|
| ABa-ABp | 51.02 | 31.39 | -38.5% |
| ABa-EMS | 34.35 | 25.71 | -25.1% |
| ABa-P2 | 0.00 | 12.78 | +1278110790252.7% |
| ABp-EMS | 36.57 | 17.61 | -51.8% |
| ABp-P2 | 37.51 | 14.51 | -61.3% |
| EMS-P2 | 22.38 | 16.21 | -27.6% |

### Confusion Matrix (Contact Existence)

```
True\Pred   No Contact   Contact
No Contact          10         0
Contact              2         0
```

### Attention Weight Matrix (GAT Layer 1)

```
             ABa     ABp     EMS      P2
   ABa  0.0000  0.2501  0.7500  0.2500
   ABp  0.2500  0.0000  0.0000  0.0000
   EMS  0.5000  0.4999  0.0000  0.7499
    P2  0.2500  0.2500  0.2500  0.0000
```

## 4. MDG Diagnostic Table

| Layer | Method | R^2 | Interpretation |
|-------|--------|-----|----------------|
| 1 | ABM (JKR) | 0.3826 | Known physics ceiling |
| 2 | SINDy (dz/dt) | 0.2537 | Discovered equation fit (best axis) |
| 3 | GNN (GAT) | -0.0212 | Data ceiling (LOTO, optimistic) |
| Gap | GNN - ABM | -0.4038 | Hidden variable signature |

**Interpretation:** GNN R2=-0.0212 < ABM R2=0.3826. GNN did not outperform ABM -- insufficient data (LOTO with 2 timepoints from 1 embryo). This does NOT mean ABM is superior.

## 5. Honest Limitations

- **N = 2 graphs** (all from one embryo, CDSample04). Overfitting risk is HIGH with 1 training graphs.

- **4-cell stage only.** Out-of-scope for other stages.

- **Spherical approximation.** R from V0 assumes perfect spheres; real C. elegans cells are non-spherical.

- **Cross-embryo generalization unknown.** Proper data ceiling requires N >= 10 embryos with held-out cross-validation. This remains the target once multi-sample CShaper data is available.