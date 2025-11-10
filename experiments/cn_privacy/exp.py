import gc
import multiprocessing  # noqa: F401 # pylint: disable=unused-import
from itertools import product

from joblib import Parallel, delayed

from src.config import get_out_path, load_config
from src.data import generate_randombn
from src.mia import attack_cn_bn


def main():

    # Load config
    config = load_config("cn_privacy")

    # Generate BNs and data
    generate_randombn(config)

    # Get base path
    out_path = get_out_path(config)

    # Set number of threads for parallelization
    num_cores = eval(config["num_cores"])

    # For each ESS and each model ...
    exp_vec = [
        f.stem for f in (out_path / config["data_path"]).iterdir() if f.is_file()
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
