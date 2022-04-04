import pandas as pd
from gpucsl.pc.discover_skeleton_gaussian import discover_skeleton_gpu_gaussian
from gpucsl.pc.helpers import init_pc_graph
from gpucsl.pc.pc import GaussianPC

samples = pd.read_csv("data/coolingData/coolingData.csv", header=None).to_numpy()
max_level = 3
alpha = 0.05

# way 1: use the PC class

pc = GaussianPC(samples, max_level, alpha).set_distribution_specific_options()

((skeleton, separation_sets, pmax, computation_time), discovery_runtime) = pc.discover_skeleton()


# way 2: call the discover method yourself

graph = init_pc_graph(samples)
num_variables = samples.shape[1]
num_observations = samples.shape[0]

((skeleton, separation_sets, pmax, computation_time), discovery_runtime) = discover_skeleton_gpu_gaussian(
    graph,
    samples,
    None,
    alpha,
    max_level,
    num_variables,
    num_observations
)