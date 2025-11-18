import gc
import multiprocessing  # noqa: F401 # pylint: disable=unused-import

import numpy as np  # noqa: F401 # pylint: disable=unused-import
from joblib import Parallel, delayed

from src.config import create_clean_dir, get_cur_dir, load_config
from src.data import generate_randombn
from src.learning import estimate_bns


def main():

    # Init configs
    config = load_config("cn_privacy")
    cur_dir = get_cur_dir(config)
    num_cores = eval(config["num_cores"])

    # Generate BNs and data
    print("#" * 5, "Generate BNs and data", "#" * 5)
    create_clean_dir(cur_dir / config["bns_path"])
    create_clean_dir(cur_dir / config["bns_path"] / "gt")
    create_clean_dir(cur_dir / config["data_path"])
    open(f'{cur_dir}/{config["exp_meta"]}', "w").close()
    generate_randombn(config)

    # Init the vectors of experiments
    exp_vec = [f.stem for f in (cur_dir / config["data_path"]).iterdir() if f.is_file()]

    # Estimate BNs from rpop and pool
    print("#" * 5, "Estimate BNs from rpop and pool", "#" * 5)
    create_clean_dir(cur_dir / config["bns_path"] / "rpop")
    create_clean_dir(cur_dir / config["bns_path"] / "pool")
    _ = Parallel(n_jobs=num_cores)(
        delayed(estimate_bns)(exp, config) for exp in exp_vec
    )

    # Clean
    gc.collect()


if __name__ == "__main__":
    main()
