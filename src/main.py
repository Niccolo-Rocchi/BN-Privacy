from run import *
import multiprocessing
from joblib import Parallel, delayed
from numpy import random
import os
from pathlib import Path
from pprint import pformat

num_cores = multiprocessing.cpu_count() - 1

# Initilize eps list for each ess
ess_dict = {
        1: np.arange(0.1, 10, 0.1), 
        10: np.arange(0.1, 10, 0.1),
        20: np.arange(0.05, 5, 0.05),
        30: np.arange(5e-4, 1e-1, 5e-4), 
        40: np.arange(1e-5, 5e-2, 1e-5), 
        50: np.arange(1e-6, 5e-3, 1e-6)
}

# Initilize hyperparameters given ess
def get_conf(ess):

        n_nodes = gum.loadBN("./bns/exp0.bif").size()
        
        return  {
                "n_nodes": n_nodes,                                             # Actual number of BN nodes
                "ess": ess,                                                     # Actual ess (or S).
                "eps_list": ess_dict[ess],                                      # Range of eps considered, related to the actual ess.
                "results_path": f"./results/results_nodes{n_nodes}_ess{ess}",   # Actual results path, related to the actual ess.
                "n_ds": 20,                                                     # Number of data subsamples for `run_idm`.
                "n_bns": 50,                                                    # Number of BNs to evaluate within the CN for `run_idm`.
                "error": np.logspace(-4, 0, 25, endpoint=False),                # Type-I errors for `run_idm`.
                "tol": 0.01,                                                    # To find eps s.t. |AUC(eps) - AUC(CN)| < tol.
        }


if __name__ == "__main__":

        for ess in ess_dict.keys():

                # Set hyperp. configuration
                conf = get_conf(ess)
                print(f"Processing ESS = {ess}.")

                # Tracking
                file = Path(f"{conf['results_path']}/exp_meta.txt")
                file.parent.mkdir(parents=True, exist_ok=True)
                with open(file, "w") as f: f.write(pformat(compact_dict(conf)) + "\n\n" + "#"*50 + "\n\n")

                # Find eps (for any exp, in parallel)
                res = Parallel(n_jobs=num_cores)(delayed(run_idm)(exp, conf) for exp in [os.path.splitext(f)[0] for f in os.listdir("./data/")])

                # Generate evidence list for inferences
                random.seed(42)
                evid_list = [random_product(*((0,1) for _ in range(conf["n_nodes"] - 1))) for _ in range(1000)]

                # Run inferences (for any exp, in parallel)
                Parallel(n_jobs=num_cores)(delayed(run_inferences)(exp, eps, ess, evid_list, conf["results_path"]) for exp, eps in res) 