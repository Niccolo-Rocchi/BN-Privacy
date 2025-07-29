from src.config import *
import pyagrum as gum

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
    bn_str_gen = (f"T->X{i}" for i in range(config["n_nodes"] -1))
    bn_str = "; ".join(bn_str_gen)

    # For each model ...
    for i in range(config["n_models"]):

        # ... generate BN, and ...
        bn = gum.fastBN(bn_str)
        gum.saveBN(bn, f"{bns_path}/exp{i}.bif")

        # ... generate gpop from BN
        data_gen = gum.BNDatabaseGenerator(bn)
        data_gen.drawSamples(config["gpop_ss"])
        data_gen.setDiscretizedLabelModeRandom()
        gpop = data_gen.to_pandas()
        gpop.to_csv(f"{data_path}/exp{i}.csv", index=False)



