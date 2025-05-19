from run import *
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import product
import os
import shutil

num_cores = multiprocessing.cpu_count() - 1

exp_names = [os.path.splitext(f)[0] for f in os.listdir("./data/")]

if __name__ == "__main__":

    # Run experiments
    for exp in exp_names: 
        conf = [exp, {"ess":1}]
        run_idm(conf)

    # Run experiments (in parallel)
    # Parallel(n_jobs=num_cores)(delayed(run_idm)(conf) for conf in confs)
