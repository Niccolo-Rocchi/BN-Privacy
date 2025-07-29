import multiprocessing
from joblib import Parallel, delayed
from pprint import pformat
import gc

from src.inference import run_inferences
from src.mia import get_eps
from src.utils import compact_dict
from src.config import get_root_path

def run_experiment(config):
    
    # Set number of threads for parallelization
    num_cores = eval(config["num_cores"])

    # For each ESS ...
    for ess in config["ess_dict"].keys():

        print(f"Processing ESS = {ess} ... ")

        # ... create results subdirectories and metadata files, ...
        root_path = get_root_path()
        meta_file_path = root_path / config["results_path"] / f'results_nodes{config["n_nodes"]}_ess{ess}' / config["meta_file"]
        meta_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(meta_file_path, "w") as f: 
            f.write(pformat(compact_dict(config)) + "\n\n" + "#"*50 + "\n\n")

        # ... find eps s.t. |AUC(eps) - AUC(CN)| < tol, ...
        res = Parallel(n_jobs=num_cores)(delayed(get_eps)(exp, ess, config) for exp in [f.stem for f in (root_path / config["data_path"]).iterdir() if f.is_file()])            

        # ... and run inferences
        _ = Parallel(n_jobs=num_cores)(delayed(run_inferences)(exp, eps, ess, config) for exp, eps in res)

    # Clean
    gc.collect()
