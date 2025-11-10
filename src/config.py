import os
import random
import shutil
from pathlib import Path

import pyagrum as gum
import yaml


# Read configuration for experiment
def load_config(name: str):

    root = get_root_path()

    test_dir = "/tests" if os.getenv("USE_TEST_CONFIG") == "1" else ""

    config_path = root / f"configs{test_dir}" / f"{name}.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


# Set global seed
def set_global_seed(seed: int):

    random.seed(seed)
    gum.initRandom(seed)


# Get root directory
def get_root_path():
    return Path(__file__).resolve().parents[1]


# Get base path
def get_out_path(config):

    root_path = get_root_path()
    out_path = config["out_path"]

    return root_path / out_path


# Create an empty directory
def create_clean_dir(path: Path):

    # Remove the folder if already exists
    if path.exists() and path.is_dir():
        shutil.rmtree(path)

    # Create a new folder
    path.mkdir(parents=True, exist_ok=True)
