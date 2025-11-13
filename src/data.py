from itertools import product

import numpy as np
import pyagrum as gum
from numpy.random import randint

from src.config import create_clean_dir, get_out_path, set_global_seed
from src.utils import safe_assert, save_bn


def generate_naivebayes(config):

    # Set seed
    set_global_seed(config["seed"])

    # Set paths
    out_path = get_out_path(config)
    bns_path = out_path / config["bns_path"]
    data_path = out_path / config["data_path"]

    # Retrieve hyperparameters
    n_modmax = config["n_modmax"]
    gpop_ss = config["gpop_ss"]
    pool_ss = int(gpop_ss * config["pool_prop"])
    rpop_ss = int(gpop_ss * config["rpop_prop"])
    n_samples = config["n_samples"]

    # Set BN (naive Bayes) structure
    bn_str_gen = (
        f'{config["target_var"]}->X{i}[{randint(2, n_modmax+1)}]'
        for i in range(config["n_nodes"] - 1)
    )
    bn_str = "; ".join(bn_str_gen)

    # For each model ...
    for i in range(config["n_models"]):

        # ... generate BN, ...
        bn = gum.fastBN(bn_str)
        save_bn(bn, f"exp{i}", bns_path / "gt")

        with open(f'{out_path}/{config["exp_meta"]}', "a") as m:
            m.write(
                f'- exp{i}. Naive Bayes: {config["n_nodes"]} nodes. Complexity: {bn.dim()} Max categories: {n_modmax}\n'
            )

        # ... and generate gpop from BN
        data_gen = gum.BNDatabaseGenerator(bn)
        data_gen.drawSamples(config["gpop_ss"])
        data_gen.setDiscretizedLabelModeRandom()
        gpop = data_gen.to_pandas()

        # For any data sample ...
        for sample in range(n_samples):

            # ... sample pool and rpop
            shuffled_idx = np.random.permutation(gpop.index)

            pool_idx = shuffled_idx[:pool_ss]
            rpop_idx = shuffled_idx[pool_ss : pool_ss + rpop_ss]

            gpop[f"in-pool-{sample}"] = gpop.index.isin(pool_idx)
            gpop[f"in-rpop-{sample}"] = gpop.index.isin(rpop_idx)

            # Debug
            safe_assert(pool_ss == len(pool_idx))
            safe_assert(rpop_ss == len(rpop_idx))
            safe_assert(sum(gpop[f"in-pool-{sample}"]) == pool_ss)
            safe_assert(sum(gpop[f"in-rpop-{sample}"]) == rpop_ss)

        # Save gpop
        gpop.to_csv(f"{data_path}/exp{i}.csv", index=False)


def generate_randombn(config):

    # Set seed
    set_global_seed(config["seed"])

    # Set paths
    out_path = get_out_path(config)
    bns_path = out_path / config["bns_path"]
    data_path = out_path / config["data_path"]

    # Retrieve hyperparameters
    n_nodes_vec = eval(config["n_nodes_vec"])
    edge_ratio_vec = eval(config["edge_ratio_vec"])
    n_modmax = config["n_modmax"]
    gpop_ss = config["gpop_ss"]
    pool_ss = int(gpop_ss * config["pool_prop"])
    rpop_ss = int(gpop_ss * config["rpop_prop"])
    n_samples = config["n_samples"]

    # For each configuration ...
    for i, (n, r) in enumerate(product(n_nodes_vec, edge_ratio_vec)):

        # ... generate BN, ...
        bn_gen = gum.BNGenerator()
        bn = bn_gen.generate(n_nodes=n, n_arcs=int(n * r), n_modmax=n_modmax)
        save_bn(bn, f"exp{i}", bns_path / "gt")

        with open(f'{out_path}/{config["exp_meta"]}', "a") as m:
            m.write(
                f"- exp{i}. Nodes: {n} Edges: {int(n * r)} Complexity: {bn.dim()} Max categories: {n_modmax}\n"
            )

        # ... and generate gpop from BN
        data_gen = gum.BNDatabaseGenerator(bn)
        data_gen.drawSamples(config["gpop_ss"])
        data_gen.setDiscretizedLabelModeRandom()
        gpop = data_gen.to_pandas()

        # For any data sample ...
        for sample in range(n_samples):

            # ... sample pool and rpop
            shuffled_idx = np.random.permutation(gpop.index)

            pool_idx = shuffled_idx[:pool_ss]
            rpop_idx = shuffled_idx[pool_ss : pool_ss + rpop_ss]

            gpop[f"in-pool-{sample}"] = gpop.index.isin(pool_idx)
            gpop[f"in-rpop-{sample}"] = gpop.index.isin(rpop_idx)

            # Debug
            safe_assert(pool_ss == len(pool_idx))
            safe_assert(rpop_ss == len(rpop_idx))
            safe_assert(sum(gpop[f"in-pool-{sample}"]) == pool_ss)
            safe_assert(sum(gpop[f"in-rpop-{sample}"]) == rpop_ss)

        # Save gpop
        gpop.to_csv(f"{data_path}/exp{i}.csv", index=False)
