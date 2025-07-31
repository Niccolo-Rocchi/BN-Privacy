from itertools import product
from pprint import pformat

import pyagrum as gum

from src.config import create_clean_dir, get_base_path, set_global_seed
from src.utils import compact_dict


def generate_naivebayes(config):

    # Set seed
    set_global_seed(config["seed"])

    # Set paths
    base_path = get_base_path(config)
    bns_path = base_path / config["bns_path"]
    data_path = base_path / config["data_path"]
    results_path = base_path / config["results_path"]

    # Create empty directories
    create_clean_dir(bns_path)
    create_clean_dir(data_path)
    create_clean_dir(results_path)

    # Set BN (Naive Bayes) structure
    bn_str_gen = (f'{config["target_var"]}->X{i}' for i in range(config["n_nodes"] - 1))
    bn_str = "; ".join(bn_str_gen)

    # For each model ...
    for i in range(config["n_models"]):

        # ... generate BN, ...
        bn = gum.fastBN(bn_str)
        gum.saveBN(bn, f"{bns_path}/exp{i}.bif")

        # ... and generate gpop from BN
        data_gen = gum.BNDatabaseGenerator(bn)
        data_gen.drawSamples(config["gpop_ss"])
        data_gen.setDiscretizedLabelModeRandom()
        gpop = data_gen.to_pandas()
        gpop.to_csv(f"{data_path}/exp{i}.csv", index=False)

    # For each ESS ...
    for ess in config["ess_dict"].keys():

        # ... create results subdirectories and metadata files
        meta_file_path = (
            base_path
            / config["results_path"]
            / f'results_nodes{config["n_nodes"]}_ess{ess}'
            / config["meta_file"]
        )
        meta_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(meta_file_path, "w") as f:
            f.write(pformat(compact_dict(config)) + "\n\n" + "#" * 50 + "\n\n")


def generate_randombn(config):

    # Set seed
    set_global_seed(config["seed"])

    # Set paths
    base_path = get_base_path(config)
    bns_path = base_path / config["bns_path"]
    data_path = base_path / config["data_path"]
    results_path = base_path / config["results_path"]

    # Create empty directories
    create_clean_dir(bns_path)
    create_clean_dir(data_path)
    create_clean_dir(results_path)

    n_nodes_vec = eval(config["n_nodes_vec"])
    edge_ratio_vec = eval(config["edge_ratio_vec"])

    # For each configuration ...
    for i, (n, r) in enumerate(product(n_nodes_vec, edge_ratio_vec)):

        # ... generate BN, ...
        bn_gen = gum.BNGenerator()
        bn = bn_gen.generate(n_nodes=n, n_arcs=int(n * r), n_modmax=2)
        gum.saveBN(bn, f"{bns_path}/exp{i}.bif")

        with open(f'{results_path}/{config["meta_file"]}', "a") as m:
            m.write(
                f"- exp{i}. Nodes: {n} Edges: {int(n * r)} Complexity: {bn.dim()}\n"
            )

        # ... and generate gpop from BN
        data_gen = gum.BNDatabaseGenerator(bn)
        data_gen.drawSamples(config["gpop_ss"])
        data_gen.setDiscretizedLabelModeRandom()
        gpop = data_gen.to_pandas()
        gpop.to_csv(f"{data_path}/exp{i}.csv", index=False)
