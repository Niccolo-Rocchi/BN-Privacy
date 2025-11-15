import gc
import multiprocessing  # noqa: F401 # pylint: disable=unused-import

import numpy as np  # noqa: F401 # pylint: disable=unused-import
from joblib import Parallel, delayed

from src.config import (create_clean_dir, get_out_path, load_config,
                        set_global_seed)
from src.data import generate_naivebayes
from src.learning import estimate_bns


def main():

    # Init configs
    config = load_config("cn_vs_noisybn")
    out_path = get_out_path(config)
    set_global_seed(config["seed"])
    num_cores = eval(config["num_cores"])

    # Generate BNs and data
    print("#" * 5, "Generate BNs and data", "#" * 5)
    create_clean_dir(out_path / config["bns_path"] / "gt")
    create_clean_dir(out_path / config["data_path"])
    open(f'{out_path}/{config["exp_meta"]}', "a").close()
    generate_naivebayes(config)

    # Init the vectors of experiments
    exp_vec = [
        f.stem for f in (out_path / config["data_path"]).iterdir() if f.is_file()
    ]

    # Estimate BNs from rpop and pool
    print("#" * 5, "Estimate BNs from rpop and pool", "#" * 5)
    create_clean_dir(out_path / config["bns_path"] / "rpop")
    create_clean_dir(out_path / config["bns_path"] / "pool")
    _ = Parallel(n_jobs=num_cores)(
        delayed(estimate_bns)(exp, config) for exp in exp_vec
    )

    # Clean
    gc.collect()


if __name__ == "__main__":

    main()
