import gc
import multiprocessing  # noqa: F401 # pylint: disable=unused-import
from itertools import product

from joblib import Parallel, delayed

from src.config import get_base_path, load_config
from src.inference import run_inferences
from src.mia import get_eps


def main():
    # Load config
    config = load_config("cn_vs_noisybn")

    # Get base path
    base_path = get_base_path(config)

    # Set number of threads for parallelization
    num_cores = eval(config["num_cores"])

    # For each ESS and each model ...
    exp_vec = [
        f.stem for f in (base_path / config["data_path"]).iterdir() if f.is_file()
    ]
    ess_vec = config["ess_dict"].keys()

    # ... find eps s.t. |AUC(eps) - AUC(CN)| < tol, ...
    res = Parallel(n_jobs=num_cores)(
        delayed(get_eps)(exp, ess, config) for exp, ess in product(exp_vec, ess_vec)
    )

    # ... and run inferences
    _ = Parallel(n_jobs=num_cores)(
        delayed(run_inferences)(exp, ess, eps, config) for exp, ess, eps in res
    )

    # Clean
    gc.collect()


if __name__ == "__main__":

    main()
