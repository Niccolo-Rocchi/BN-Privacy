from itertools import product

import yaml

# Set hyperparameters
names = ["cn_privacy"]
def_mecs = {
    "def_idm": {"ess": [1, 10, 100, 1000]}, 
    "def_ran": {"delta": [0.1, 0.3, 0.5, 0.7, 1.0]}
}
atk_mecs = {
    "atk_mle": {"n_bns": [1000]},    # 1000 for cn_privacy; 100 for cn_vs_noisybn
    "atk_cen": {None: [None]},
    "atk_ran": {None: [None]},
    "atk_ent": {None: [None]},
}

# Initialize the `compose.yaml` file
init = {"version": "3.9"}
with open("compose.yaml", "w") as f:
    yaml.dump(init, f, default_flow_style=False)

# For any configuration ...
# (assumption: each defense and attack mechanism has at most 1 hyperparameter to be set)
data = {"services": dict()}
for name, def_mec, atk_mec in product(names, def_mecs.keys(), atk_mecs.keys()):

    def_params = list(def_mecs[def_mec].values())[0]
    atk_params = list(atk_mecs[atk_mec].values())[0]

    for def_par, atk_par in product(def_params, atk_params):

        # ... set the related volume, ...
        app = (
            f"_{list(def_mecs[def_mec].keys())[0]}{def_par}"
            if def_par is not None
            else ""
        )
        volumes = [
            f"./experiments/{name}/bns:/workspace/experiments/{name}/bns",
            f"./experiments/{name}/data:/workspace/experiments/{name}/data",
            f"./experiments/{name}/output_{def_mec}_{atk_mec}{app}:/workspace/experiments/{name}/output",
        ]

        # ... set the command, ...
        command = [
            "python",
            "-m",
            f"experiments.{name}.exp",
            f"def_mec={def_mec}",
            f"atk_mec={atk_mec}",
        ]

        if def_par is not None:
            command.append(f"{list(def_mecs[def_mec].keys())[0]}={def_par}")
        if atk_par is not None:
            command.append(f"{list(atk_mecs[atk_mec].keys())[0]}={atk_par}")

        # ... and create the experiment
        data["services"][f"{name}_{def_mec}_{atk_mec}{app}"] = {
            "image": "bnp:2025",
            "build": ".",
            "volumes": volumes,
            "command": command,
        }
# Print number of services
print("Number of services: ", len(data["services"]))

# Write file
with open("compose.yaml", "a") as f:
    yaml.dump(data, f, default_flow_style=False)
