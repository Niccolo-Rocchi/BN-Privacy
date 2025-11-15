import gc
import multiprocessing  # noqa: F401 # pylint: disable=unused-import
import sys

import numpy as np  # noqa: F401 # pylint: disable=unused-import
import pandas as pd
from joblib import Parallel, delayed

from src.attack import attack_mechanism
from src.config import (create_clean_dir, get_out_path, load_config,
                        map_sys_args)
from src.defense import defense_mechanism
from src.inference import inferences
from src.mia import find_epsilon, mia_vs_cn


def main():

    # Init configs
    config = load_config("cn_vs_noisybn")
    out_path = get_out_path(config)
    def_mec = config["def_mec"]
    atk_mec = config["atk_mec"]
    num_cores = eval(config["num_cores"])

    # Get command-line hyperparameters
    def_mec, def_args, atk_mec, atk_args = map_sys_args(sys.argv, config)

    # Init the vectors of experiments
    exp_vec = [
        f.stem for f in (out_path / config["data_path"]).iterdir() if f.is_file()
    ]

    # Defense mechanism
    print("#" * 5, "Defense mechanism", "#" * 5)
    create_clean_dir(out_path / config["cns_path"])
    _ = Parallel(n_jobs=num_cores)(
        delayed(defense_mechanism)(exp, config, def_mec, def_args) for exp in exp_vec
    )

    # Attack mechanism
    print("#" * 5, "Attack mechanism", "#" * 5)
    create_clean_dir(out_path / config["atk_path"])
    _ = Parallel(n_jobs=num_cores)(
        delayed(attack_mechanism)(exp, config, atk_mec, atk_args) for exp in exp_vec
    )

    # MIA vs CN
    print("#" * 5, "MIA vs CN", "#" * 5)
    res = Parallel(n_jobs=num_cores)(
        delayed(mia_vs_cn)(exp, config, save_res=False) for exp in exp_vec
    )
    res = pd.DataFrame(res)
    res.to_csv(f'{out_path}/{config["auc_meta"]}', index=False)

    # Find eps s.t. |AUC(eps) - AUC(CN)| < tol
    print("#" * 5, "Get epsilon", "#" * 5)
    create_clean_dir(out_path / config["noisy_path"])
    res = Parallel(n_jobs=num_cores)(
        delayed(find_epsilon)(exp, config) for exp in exp_vec
    )
    res = pd.DataFrame(res)
    res.to_csv(f'{out_path}/{config["auc_meta"]}', index=False)

    # Run inferences
    print("#" * 5, "Run inferences", "#" * 5)
    create_clean_dir(out_path / config["results_path"] / "inferences")
    _ = Parallel(n_jobs=num_cores)(delayed(inferences)(exp, config) for exp in exp_vec)

    # Clean
    gc.collect()


if __name__ == "__main__":

    main()
