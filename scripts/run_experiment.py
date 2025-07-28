from src.mia import get_eps
from src.inference import *
import multiprocessing
from joblib import Parallel, delayed
import os
from pprint import pformat

from src.config import *

if __name__ == "__main__":

    # Load config
    config = get_config("configs/config1.yaml")

    # Set number of threads for parallelization
    num_cores = eval(config["num_cores"])

    # For each ESS ...
    for ess in config["ess_dict"].keys():

            print(f"Processing ESS = {ess} ... ", end="")

            # ... create results subdirectories and metadata files, ...
            root_dir = get_root_dir()
            meta_file_path = root_dir / config["results_dir"] / f'results_nodes{config["n_nodes"]}_ess{ess}' / config["meta_file"]
            meta_file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(meta_file_path, "w") as f: 
                 f.write(pformat(compact_dict(config)) + "\n\n" + "#"*50 + "\n\n")

            # ... find eps s.t. |AUC(eps) - AUC(CN)| < tol, ...
            res = Parallel(n_jobs=num_cores)(delayed(get_eps)(exp, ess, config) for exp in [os.path.splitext(f)[0] for f in os.listdir("./data/")])            

            # ... and run inferences
            Parallel(n_jobs=num_cores)(delayed(run_inferences)(exp, eps, ess, config) for exp, eps in res) 

            print("[OK]")