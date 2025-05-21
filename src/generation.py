import math
from pathlib import Path
import warnings
import traceback
import numpy as np
from numpy import random
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
res_path = "./results"
if not os.path.exists(res_path): os.makedirs(res_path)

# Set experiments hyperparameters
n_nodes = 12
gpop_ss = 2000
n_exps = 10

# Set BN (NB) structure
bn_str_gen = (f"T->X{i}" for i in range(n_nodes -1))
bn_str = "; ".join(bn_str_gen)

# Set seeds
random.seed(42)
gum.initRandom(seed=42)

# For each experiment ...
for i in range(n_exps):

    # Generate BN
    bn = gum.fastBN(bn_str)
    gum.saveBN(bn, f"./bns/exp{i}.bif")

    # Generate gpop
    data_gen = gum.BNDatabaseGenerator(bn)
    data_gen.drawSamples(gpop_ss)
    data_gen.setDiscretizedLabelModeRandom()
    gpop = data_gen.to_pandas()
    gpop.to_csv(f"./data/exp{i}.csv", index=False)

    
