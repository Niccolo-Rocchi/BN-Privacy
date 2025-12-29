import pandas as pd
import pyagrum as gum

from src.config import get_cur_dir, safe_assert, set_seed


# Learn BN parameters from a given BN and data
def learn_bn_params(bn, data):

    bn_copy = gum.BayesNet(bn)

    learner = gum.BNLearner(data, bn_copy)
    learner.useSmoothingPrior(1e-10)
    bn_learnt = learner.learnParameters(bn_copy)

    return bn_learnt


# Estimate BNs from rpop and pool
def estimate_bns(exp, config) -> None:

    # Get current directory
    cur_dir = get_cur_dir(config)

    # Set seed
    set_seed()

    # Read data
    gpop = pd.read_csv(f'{cur_dir / config["data_path"]}/{exp}.csv')
    bn = gum.loadBN(f'{cur_dir / config["bns_path"]}/gt/{exp}.bif')
    n_nodes = len(bn.nodes())
    gpop_ss = config["gpop_ss"]
    rpop_ss = int(gpop_ss * config["rpop_prop"])
    pool_ss = int(gpop_ss * config["pool_prop"])

    # Debug
    safe_assert(gpop_ss == gpop.shape[0])
    safe_assert(n_nodes == gpop.loc[:, ~gpop.columns.str.contains("in-")].shape[1])

    # For each data sample ...
    for sample in range(config["samples"]):

        # ... retrieve pool and rpop, ...
        pool = gpop[gpop[f"in-pool-{sample}"]].iloc[:, :n_nodes]
        rpop = gpop[gpop[f"in-rpop-{sample}"]].iloc[:, :n_nodes]

        # ... estimate BN from rpop, ...
        bn_learnt = learn_bn_params(bn, rpop)
        gum.saveBN(
            bn_learnt,
            f'{cur_dir / config["bns_path"] / "rpop"}/{f"bn_{exp}_sample{sample}"}.bif',
        )

        # ... estimate BN from pool, ...
        bn_learnt = learn_bn_params(bn, pool)
        gum.saveBN(
            bn_learnt,
            f'{cur_dir / config["bns_path"] / "pool"}/{f"bn_{exp}_sample{sample}"}.bif',
        )

        # Debug
        safe_assert(len(pool) == sum(gpop[f"in-pool-{sample}"]))
        safe_assert(len(pool) == pool_ss)
        safe_assert(len(rpop) == rpop_ss)

    return
