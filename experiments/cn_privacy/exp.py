import gc
import multiprocessing  # noqa: F401 # pylint: disable=unused-import
from itertools import product

from joblib import Parallel, delayed

from src.config import get_base_path, load_config
from src.mia import attack_cn_bn


def main():
    # Load config
    config = load_config("cn_privacy")

    # Get base path
    base_path = get_base_path(config)

    # Set number of threads for parallelization
    num_cores = eval(config["num_cores"])

    # For each ESS and each model ...
    exp_vec = [
        f.stem for f in (base_path / config["data_path"]).iterdir() if f.is_file()
    ]
    ess_vec = eval(config["ess_vec"])

    # ... run MIA attack on BN and CN
    Parallel(n_jobs=num_cores)(
        delayed(attack_cn_bn)(exp, ess, config)
        for exp, ess in product(exp_vec, ess_vec)
    )

    # Clean
    gc.collect()


if __name__ == "__main__":

    main()
