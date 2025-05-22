from run import *
import multiprocessing
from joblib import Parallel, delayed
from numpy import random
import os

num_cores = multiprocessing.cpu_count() - 1

exp_names = [os.path.splitext(f)[0] for f in os.listdir("./data/")]
confs = [(exp, {"ess":1}) for exp in exp_names]

if __name__ == "__main__":

    # Set seeds
    random.seed(42)
    gum.initRandom(seed=42)

    # Find eps s.t. AUC(eps)~AUC(CN) (for any exp, in parallel)
    res = Parallel(n_jobs=num_cores)(delayed(run_idm)(conf) for conf in confs)

    # Run inferences (for any exp, in parallel)
    Parallel(n_jobs=num_cores)(delayed(run_inferences)(a, b) for a, b in res)
    

    