import gc
import multiprocessing  # noqa: F401 # pylint: disable=unused-import

import numpy as np  # noqa: F401 # pylint: disable=unused-import
from joblib import Parallel, delayed

from src.attack import attack_mechanism
from src.config import (create_clean_dir, get_out_path, load_config,
                        set_global_seed)
from src.data import generate_randombn
from src.defense import defense_mechanism
from src.learning import estimate_bns
from src.mia import mia_vs_bn, mia_vs_cn, theoretical_power


def main():

    # Init configs
    config = load_config("cn_privacy")
    out_path = get_out_path(config)
    set_global_seed(config["seed"])
    def_mec = config["def_mec"]
    atk_mec = config["atk_mec"]
    num_cores = eval(config["num_cores"])

    # Generate BNs and data
    print("#" * 5, "Generate BNs and data", "#" * 5)
    create_clean_dir(out_path / config["bns_path"] / "gt")
    create_clean_dir(out_path / config["data_path"])
    open(f'{out_path}/{config["exp_meta"]}', "a").close()
    generate_randombn(config)

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

    # Defense mechanism
    print("#" * 5, "Defense mechanism", "#" * 5)
    create_clean_dir(out_path / config["cns_path"])
    _ = Parallel(n_jobs=num_cores)(
        delayed(defense_mechanism)(def_mec, exp, config) for exp in exp_vec
    )

    # Attack mechanism
    print("#" * 5, "Attack mechanism", "#" * 5)
    create_clean_dir(out_path / config["atk_path"])
    _ = Parallel(n_jobs=num_cores)(
        delayed(attack_mechanism)(atk_mec, exp, config) for exp in exp_vec
    )

    # MIA vs CN
    print("#" * 5, "MIA vs CN", "#" * 5)
    create_clean_dir(out_path / config["results_path"] / "cns")
    _ = Parallel(n_jobs=num_cores)(
        delayed(mia_vs_cn)(exp, config) for exp in exp_vec
    )

    # MIA vs BN
    print("#" * 5, "MIA vs BN", "#" * 5)
    create_clean_dir(out_path / config["results_path"] / "bns")
    _ = Parallel(n_jobs=num_cores)(
        delayed(mia_vs_bn)(exp, config) for exp in exp_vec
    )

    # Compute theoretical power
    print("#" * 5, "Compute theoretical power", "#" * 5)
    _ = Parallel(n_jobs=num_cores)(
        delayed(theoretical_power)(exp, config) for exp in exp_vec
    )

    # Clean
    gc.collect()


if __name__ == "__main__":
    main()
