import gc
import multiprocessing  # noqa: F401 # pylint: disable=unused-import
import sys

import numpy as np  # noqa: F401 # pylint: disable=unused-import
from joblib import Parallel, delayed

from src.attack import attack_mechanism
from src.config import create_clean_dir, get_cur_dir, load_config, map_sys_args
from src.defense import defense_mechanism
from src.mia import mia_vs_bn, mia_vs_cn, theoretical_power


def main():

    # Init configs
    config = load_config("cn_privacy")
    cur_dir = get_cur_dir(config)
    num_cores = eval(config["num_cores"])

    # Get command-line hyperparameters
    def_mec, def_args, atk_mec, atk_args = map_sys_args(sys.argv, config)

    # Init the vectors of experiments
    exp_vec = [f.stem for f in (cur_dir / config["data_path"]).iterdir() if f.is_file()]

    # Defense mechanism
    print("#" * 5, "Defense mechanism", "#" * 5)
    create_clean_dir(cur_dir / config["cns_path"])
    _ = Parallel(n_jobs=num_cores)(
        delayed(defense_mechanism)(exp, config, def_mec, def_args) for exp in exp_vec
    )

    # Attack mechanism
    print("#" * 5, "Attack mechanism", "#" * 5)
    create_clean_dir(cur_dir / config["atk_path"])
    _ = Parallel(n_jobs=num_cores)(
        delayed(attack_mechanism)(exp, config, atk_mec, atk_args) for exp in exp_vec
    )

    # MIA vs CN
    print("#" * 5, "MIA vs CN", "#" * 5)
    create_clean_dir(cur_dir / config["results_path"] / "cns")
    _ = Parallel(n_jobs=num_cores)(delayed(mia_vs_cn)(exp, config) for exp in exp_vec)

    # MIA vs BN
    print("#" * 5, "MIA vs BN", "#" * 5)
    create_clean_dir(cur_dir / config["results_path"] / "bns")
    _ = Parallel(n_jobs=num_cores)(delayed(mia_vs_bn)(exp, config) for exp in exp_vec)

    # Compute theoretical power
    print("#" * 5, "Compute theoretical power", "#" * 5)
    _ = Parallel(n_jobs=num_cores)(
        delayed(theoretical_power)(exp, config) for exp in exp_vec
    )

    # Clean
    gc.collect()


if __name__ == "__main__":
    main()
