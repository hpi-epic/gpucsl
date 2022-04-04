import pandas as pd
from gpucsl.pc.pc import GaussianPC

samples = pd.read_csv("data/coolingData/coolingData.csv", header=None).to_numpy()
max_level = 3
alpha = 0.05

(
    (
        directed_graph,
        separation_sets,
        pmax,
        discover_skeleton_runtime,
        edge_orientation_runtime,
        discover_skeleton_kernel_runtime,
    ),
    pc_runtime,
) = (
    GaussianPC(samples, max_level, alpha).set_distribution_specific_options().execute()
)
