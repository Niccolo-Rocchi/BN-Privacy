import gc
import multiprocessing  # noqa: F401 # pylint: disable=unused-import
from itertools import product

import numpy as np  # noqa: F401 # pylint: disable=unused-import
from joblib import Parallel, delayed

from src.config import create_clean_dir, get_out_path, load_config, set_global_seed
from src.data import generate_randombn
from src.mia import (
    phase_attack_mechanism,
    phase_defense_mechanism,
    phase_estimation,
    phase_mia_vs_bn,
    phase_mia_vs_cn,
    phase_theoretical_power,
)


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
    generate_randombn(config)

    # Init the vectors of experiments
    exp_vec = [
        f.stem for f in (out_path / config["data_path"]).iterdir() if f.is_file()
    ]
    ess_vec = eval(config["ess_vec"])

    # Estimate BNs from rpop and pool
    print("#" * 5, "Estimate BNs from rpop and pool", "#" * 5)
    create_clean_dir(out_path / config["bns_path"] / "rpop")
    create_clean_dir(out_path / config["bns_path"] / "pool")
    _ = Parallel(n_jobs=num_cores)(
        delayed(phase_estimation)(exp, config) for exp in exp_vec
    )

    # Defense mechanism
    print("#" * 5, "Defense mechanism", "#" * 5)
    create_clean_dir(out_path / config["cns_path"])
    _ = Parallel(n_jobs=num_cores)(
        delayed(phase_defense_mechanism)(def_mec, exp, ess, config)
        for exp, ess in product(exp_vec, ess_vec)
    )

    # Attack mechanism
    print("#" * 5, "Attack mechanism", "#" * 5)
    create_clean_dir(out_path / config["atk_path"])
    _ = Parallel(n_jobs=num_cores)(
        delayed(phase_attack_mechanism)(atk_mec, exp, ess, config)
        for exp, ess in product(exp_vec, ess_vec)
    )

    # MIA vs CN
    print("#" * 5, "MIA vs CN", "#" * 5)
    create_clean_dir(out_path / config["results_path"] / "cns")
    _ = Parallel(n_jobs=num_cores)(
        delayed(phase_mia_vs_cn)(exp, ess, config)
        for exp, ess in product(exp_vec, ess_vec)
    )

    # MIA vs BN
    print("#" * 5, "MIA vs BN", "#" * 5)
    create_clean_dir(out_path / config["results_path"] / "bns")
    _ = Parallel(n_jobs=num_cores)(
        delayed(phase_mia_vs_bn)(exp, config) for exp in exp_vec
    )

    # Compute theoretical power
    print("#" * 5, "Compute theoretical power", "#" * 5)
    _ = Parallel(n_jobs=num_cores)(
        delayed(phase_theoretical_power)(exp, config) for exp in exp_vec
    )

    # Clean
    gc.collect()


if __name__ == "__main__":
    main()
