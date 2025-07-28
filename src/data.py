from src.config import *
import pyagrum as gum

def generate_data(config):

    # Set seed
    set_global_seed(config["seed"]) 

    # Set paths
    root_dir = get_root_dir()
    bns_dir = root_dir / config["bns_dir"]
    data_dir = root_dir / config["data_dir"]
    results_dir = root_dir / config["results_dir"]
    tmp_dir = root_dir / config["tmp_dir"]

    # Create empty directories
    create_clean_dir(bns_dir)
    create_clean_dir(data_dir)
    create_clean_dir(results_dir)
    create_clean_dir(tmp_dir)

    # Set BN (Naive Bayes) structure
    bn_str_gen = (f"T->X{i}" for i in range(config["n_nodes"] -1))
    bn_str = "; ".join(bn_str_gen)

    # For each experiment ...
    for i in range(config["n_exps"]):

        # ... generate BN, and ...
        bn = gum.fastBN(bn_str)
        gum.saveBN(bn, f"{bns_dir}/exp{i}.bif")

        # ... generate gpop from BN
        data_gen = gum.BNDatabaseGenerator(bn)
        data_gen.drawSamples(config["gpop_ss"])
        data_gen.setDiscretizedLabelModeRandom()
        gpop = data_gen.to_pandas()
        gpop.to_csv(f"{data_dir}/exp{i}.csv", index=False)



