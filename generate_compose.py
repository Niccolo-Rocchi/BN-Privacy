import yaml
from itertools import product

# Set hyperparameters
names = ["cn_privacy", "cn_vs_noisybn"]
def_mecs = {"def_idm":{"ess":[1, 2, 10]}, "def_ran":{"delta":[0.2, 0.4]}}
atk_mecs = {"atk_mle":{"n_bns":[5]}}

# Initialize the `compose.yaml` file
init = {"version": "3.9"}
with open("compose.yaml", "w") as f:
    yaml.dump(init, f, default_flow_style=False)

# For any configuration ...
data = {"services": dict()}
for name, def_mec, atk_mec in product(names, def_mecs.keys(), atk_mecs.keys()):

    def_params = def_mecs[def_mec]
    atk_params = atk_mecs[atk_mec]

    # (assumption: each defense and attack mechanism has only 1 hyperparameter to be set)
    for def_par, atk_par in product(list(def_params.values())[0], list(atk_params.values())[0]):

        # ... set the related volume, ...
        volumes = [
            f"./experiments/{name}/bns:/workspace/experiments/{name}/bns",
            f"./experiments/{name}/data:/workspace/experiments/{name}/data",
            f"./experiments/{name}/output_{def_mec}_{atk_mec}_{list(def_params.keys())[0]}{def_par}:/workspace/experiments/{name}/output",
        ]

        # ... and create the experiment
        data["services"][f"{name}_{def_mec}_{atk_mec}_{list(def_params.keys())[0]}{def_par}"] = {
            "image": "bnp:2025",
            "build": ".",
            "volumes": volumes,
            "command": [
                "python",
                "-m",
                f"experiments.{name}.exp",
                f"def_mec={def_mec}",
                f"{list(def_params.keys())[0]}={def_par}",
                f"atk_mec={atk_mec}",
                f"{list(atk_params.keys())[0]}={atk_par}",            ],
        }

# Write file
with open("compose.yaml", "a") as f:
    yaml.dump(data, f, default_flow_style=False)
