import gc
import multiprocessing  # noqa: F401 # pylint: disable=unused-import
import sys

import numpy as np  # noqa: F401 # pylint: disable=unused-import
import pandas as pd
from joblib import Parallel, delayed

from src.attack import attack_mechanism
from src.config import create_clean_dir, get_cur_dir, load_config, map_sys_args
from src.defense import defense_mechanism
from src.inference import inferences
from src.mia import find_epsilon, mia_vs_cn


def main():

    # Init configs
    config = load_config("cn_vs_noisybn")
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
    res = Parallel(n_jobs=num_cores)(
        delayed(mia_vs_cn)(exp, config, save_power_res=False) for exp in exp_vec
    )
    auc_res = pd.concat((i for i in res), axis=0)
    auc_res.to_csv(f'{cur_dir}/{config["auc_meta"]}', index=False)

    # Find eps s.t. |AUC(eps) - AUC(CN)| < tol
    print("#" * 5, "Get epsilon", "#" * 5)
    create_clean_dir(cur_dir / config["noisy_path"])
    res = Parallel(n_jobs=num_cores)(
        delayed(find_epsilon)(exp, config) for exp in exp_vec
    )
    auc_res = pd.concat((i for i in res), axis=0)
    auc_res.to_csv(f'{cur_dir}/{config["auc_meta"]}', index=False)

    # Run inferences
    print("#" * 5, "Run inferences", "#" * 5)
    create_clean_dir(cur_dir / config["results_path"] / "inferences")
    _ = Parallel(n_jobs=num_cores)(delayed(inferences)(exp, config) for exp in exp_vec)

    # Clean
    gc.collect()


if __name__ == "__main__":

    main()
