# Libraries
import yaml
from run import *
from pathlib import Path

# Main
if __name__ == "__main__":
    
    # Read all configurations
    with open('config.yaml') as f: configs = yaml.safe_load(f.read())
    inv_configs = configs["invar"]
    var_configs = configs["var"]

    # Create results directories
    results_path = Path("./results/idm")
    results_path.mkdir(parents=True, exist_ok=True)
    results_path = Path("./results/cont")
    results_path.mkdir(parents=True, exist_ok=True)

    # For each 'idm' configuration ...
    for conf in var_configs["idm"]:

        # Run experiment
        run_exp_idm(inv_configs, conf)

    # For each 'cont' configuration ...
    for conf in var_configs["cont"]:

        # Run experiment
        run_exp_cont(inv_configs, conf)