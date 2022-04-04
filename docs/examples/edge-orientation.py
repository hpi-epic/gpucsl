import pandas as pd
from gpucsl.pc.pc import GaussianPC
import networkx as nx
from gpucsl.pc.edge_orientation.edge_orientation import orient_edges

samples = pd.read_csv("data/coolingData/coolingData.csv", header=None).to_numpy()
max_level = 3
alpha = 0.05

# you will need the skeleton and separation sets from the skeleton discovery

pc = GaussianPC(samples, max_level, alpha).set_distribution_specific_options()

((skeleton, separation_sets, _, _), _) = pc.discover_skeleton()

# do stuff

(directed_graph, edge_orientation_time) = orient_edges(
    nx.DiGraph(skeleton), separation_sets
)
