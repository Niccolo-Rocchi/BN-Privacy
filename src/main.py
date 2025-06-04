from run import *
import multiprocessing
from joblib import Parallel, delayed
from numpy import random
import os

num_cores = multiprocessing.cpu_count() - 1

ess = 45
eps_list = np.arange(1e-5, 1e-3, 1e-5)
exp_names = [os.path.splitext(f)[0] for f in os.listdir("./data/")]
confs = [(exp, {"ess":ess}, eps_list) for exp in exp_names]

bn = gum.loadBN("./bns/exp0.bif")
n_nodes = bn.size()

if __name__ == "__main__":

        # Set seeds and generate evidence list for inferences
        random.seed(42)
        gum.initRandom(seed=42)
        evid_list = [random_product(*((0,1) for _ in range(n_nodes - 1))) for _ in range(1000)]

        # Track
        with open("./results/exp_meta.txt", "a") as m: 
                m.write(f"- Nodes: {n_nodes} Ess: {ess} Exps: {len(exp_names)}\n Eps list: {eps_list}\n\n")

        # Find eps s.t. AUC(eps)~AUC(CN) (for any exp, in parallel)
        res = Parallel(n_jobs=num_cores)(delayed(run_idm)(conf) for conf in confs)

        # Run inferences (for any exp, in parallel)
        Parallel(n_jobs=num_cores)(delayed(run_inferences)(exp, eps, ess, evid_list) for exp, eps in res) 