from run import *
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import product
import os
import shutil

num_cores = multiprocessing.cpu_count() - 1

exp_names = [os.path.splitext(f)[0] for f in os.listdir("./data/")]

trials = [
    {"ess": 1}, 
    {"ess": 2}, 
    {"ess": 5},
    {"ess": 10}
]

confs = tqdm([i for i in product(exp_names, trials)])

if __name__ == "__main__":

    # Create results directory
    res_path = "./results"
    if os.path.exists(res_path): shutil.rmtree(res_path)
    os.makedirs(res_path)

    # Run experiments (in parallel)
    Parallel(n_jobs=num_cores)(delayed(run_idm)(conf) for conf in confs)