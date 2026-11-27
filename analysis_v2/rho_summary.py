"""Spearman rho between Quest page scores and the true dense attention mass, per setting.

  python -m analysis_v2.rho_summary results_v2/ext/*/rho.csv
"""
import csv
import sys
from collections import defaultdict

import numpy as np

for path in sys.argv[1:]:
    by_layer = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by_layer[int(r["layer"])].append(float(r["mean_rho"]))
    layer_means = np.array([np.nanmean(v) for _, v in sorted(by_layer.items())])
    print(f"{path}: layers {len(layer_means)}, mean over layers {np.nanmean(layer_means):.3f}, "
          f"min {np.nanmin(layer_means):.3f} (layer {int(np.nanargmin(layer_means))}), "
          f"max {np.nanmax(layer_means):.3f} (layer {int(np.nanargmax(layer_means))})")
