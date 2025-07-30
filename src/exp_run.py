import multiprocessing
from joblib import Parallel, delayed
from itertools import product
import gc

from src.inference import run_inferences
from src.membership_attack import get_eps, attack_cn_bn
from src.config import get_base_path

def run_cn_vs_noisybn(config):

    # Get base path 
    base_path = get_base_path(config)
    
    # Set number of threads for parallelization
    num_cores = eval(config["num_cores"])

    # For each ESS and each model ...
    exp_vec = [f.stem for f in (base_path / config["data_path"]).iterdir() if f.is_file()]
    ess_vec = config["ess_dict"].keys()

    # ... find eps s.t. |AUC(eps) - AUC(CN)| < tol, ...
    res = Parallel(n_jobs=num_cores)(delayed(get_eps)(exp, ess, config) for exp, ess in product(exp_vec, ess_vec))            

    # ... and run inferences
    _ = Parallel(n_jobs=num_cores)(delayed(run_inferences)(exp, ess, eps, config) for exp, ess, eps in res)

    # Clean
    gc.collect()

def run_cn_privacy(config):

    # Get base path 
    base_path = get_base_path(config)
    
    # Set number of threads for parallelization
    num_cores = eval(config["num_cores"])

    # For each ESS and each model ...
    exp_vec = [f.stem for f in (base_path / config["data_path"]).iterdir() if f.is_file()]
    ess_vec = eval(config["ess_vec"])

    # ... run MIA attack on BN and CN
    Parallel(n_jobs=num_cores)(delayed(attack_cn_bn)(exp, ess, config) for exp, ess in product(exp_vec, ess_vec))

    # Clean
    gc.collect()
