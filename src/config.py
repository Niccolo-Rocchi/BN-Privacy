import os
import random
import shutil
import sys
from pathlib import Path

import pyagrum as gum
import yaml

IN_PYTEST = "pytest" in sys.modules


# Get arguments as passed from command-line for experiment
def map_sys_args(sys_args, config) -> tuple:

    # Store parameters
    params = dict([arg.split("=") for arg in sys_args if "=" in arg])
    with open(f'{config["cur_dir"]}/{config["exp_meta"]}', "a") as m:
        m.write(f"\n Defense & attack parameters: \n {params}")

    # Get defense and attack mechanisms
    def_mec = params.pop("def_mec")
    atk_mec = params.pop("atk_mec")

    # Get defense parameters
    def_args = dict()
    if def_mec == "def_idm":
        def_args["ess"] = int(params.pop("ess"))
        assert def_args["ess"] >= 0
    elif def_mec == "def_ran":
        def_args["delta"] = float(params.pop("delta"))
        assert def_args["delta"] >= 0
        assert def_args["delta"] <= 1
    else:
        raise Exception("Defense not implemented")

    # Save attack parameters
    atk_args = dict()
    if atk_mec == "atk_mle":
        atk_args["n_bns"] = int(params.pop("n_bns"))
        assert atk_args["n_bns"] >= 1
    else:
        raise Exception("Attack not implemented")

    # Exceptions
    if len(params) != 0:
        raise Exception(f"Unused parameters: {params}")

    return (def_mec, def_args, atk_mec, atk_args)


# Read configuration for experiment
def load_config(name: str):

    subdir = "test" if os.getenv("USE_TEST_CONFIG") == "1" else "experiments"

    config_path = get_root_path() / subdir / name / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


# Set global seed
def set_seed():

    random.seed(42)
    gum.initRandom(42)


# Create an empty directory
def create_clean_dir(path: Path):

    # Remove the folder if already exists
    if path.exists() and path.is_dir():
        shutil.rmtree(path)

    # Create a new folder
    path.mkdir(parents=True, exist_ok=True)


# Get output path
def get_cur_dir(config):

    root_path = get_root_path()
    cur_dir = config["cur_dir"]

    return root_path / cur_dir


# Get root (project) directory
def get_root_path():
    return Path(__file__).resolve().parents[1]


# Only perform an `assert` if code is running in `pytest`
def safe_assert(condition):
    if IN_PYTEST:
        assert condition
