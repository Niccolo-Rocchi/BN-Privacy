import gc
import multiprocessing  # noqa: F401 # pylint: disable=unused-import
from itertools import product
import numpy as np # noqa: F401 # pylint: disable=unused-import

from joblib import Parallel, delayed

from src.config import get_out_path, load_config, set_global_seed
from src.data import generate_randombn
from src.mia import phase_attack_mechanism, phase_defense_mechanism, phase_estimation, phase_mia_vs_bn, phase_mia_vs_cn, phase_theoretical_power


def main():

    # Init configs
    config = load_config("cn_privacy")
    out_path = get_out_path(config)
    set_global_seed(config["seed"])
    results_path = out_path / config["results_path"]
    n_samples = config["n_samples"]
    error = eval(config["error"])
    def_mec = config["def_mec"]
    atk_mec = config["atk_mec"]

    # Init the vectors of experiments
    exp_vec = [
        f.stem for f in (out_path / config["data_path"]).iterdir() if f.is_file()
    ]
    ess_vec = eval(config["ess_vec"])

    # Generate BNs and data
    print("#"*5, "Generate BNs and data", "#"*5)
    generate_randombn(config)

    # Estimate BNs from rpop and pool
    print("#"*5, "Estimate BNs from rpop and pool", "#"*5)
    for exp in exp_vec:
        phase_estimation(exp, config)

    # Defense mechanism
    print("#"*5, "Defense mechanism", "#"*5)
    for exp, ess in product(exp_vec, ess_vec):
        phase_defense_mechanism(def_mec, exp, ess, config)

    # Attack mechanism
    print("#"*5, "Attack mechanism", "#"*5)
    for exp, ess in product(exp_vec, ess_vec):
        phase_attack_mechanism(atk_mec, exp, ess, config)

    # MIA vs CN
    print("#"*5, "MIA vs CN", "#"*5)
    for exp, ess in product(exp_vec, ess_vec):
        phase_mia_vs_cn(exp, ess, config)

    # MIA vs BN (for comparison)
    print("#"*5, "MIA vs BN", "#"*5)
    for exp in exp_vec:
        phase_mia_vs_bn(exp, config)

    # Compute theoretical power
    print("#"*5, "Compute theoretical power", "#"*5)
    for exp in exp_vec:
        phase_theoretical_power(exp, config)

    # -------------------------

    # OLD! TODO: remove
    # # ... run MIA attack on BN and CN
    # Parallel(n_jobs=eval(config["num_cores"]))(
    #     delayed(attack_cn_bn)(exp, ess, config)
    #     for exp, ess in product(exp_vec, ess_vec)
    # )

    # Clean
    gc.collect()


if __name__ == "__main__":

    main()
