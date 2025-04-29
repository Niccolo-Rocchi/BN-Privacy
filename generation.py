
import math
from pathlib import Path
import warnings
import traceback
import numpy as np
import pandas as pd
from scipy.stats import norm
import pyagrum as gum
import yaml
import shutil
import os

from utils import *

warnings.filterwarnings('ignore')

# Create directories
bn_path = "./bns"
if os.path.exists(bn_path): shutil.rmtree(bn_path)
os.makedirs(bn_path)
data_path = "./data"
if os.path.exists(data_path): shutil.rmtree(data_path)
os.makedirs(data_path)

# Set experiments hyperparameters
n_nodes = [10, 20, 50, 70, 100]
edge_ratios = [1, 2, 3, 4]
gpop_ss = 10000

# For each configuration ...
for i, (n, r) in enumerate(product(n_nodes, edge_ratios)):

    # Generate BN
    bn_gen = gum.BNGenerator()
    bn = bn_gen.generate(n_nodes=n, n_arcs=int(n * r), n_modmax=2)
    gum.saveBN(bn, f"./bns/exp{i}.bif")
    with open("./exp_meta.txt", "a") as m: 
        m.write(f"- exp{i}. Nodes: {n} Edges: {int(n * r)} Complexity: {bn.dim()}\n")

    # Generate gpop
    data_gen = gum.BNDatabaseGenerator(bn)
    data_gen.drawSamples(gpop_ss)
    data_gen.setDiscretizedLabelModeRandom()
    gpop = data_gen.to_pandas()
    gpop.to_csv(f"./data/exp{i}.csv", index=False)

    