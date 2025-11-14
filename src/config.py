import os
import random
import shutil
import sys
from pathlib import Path

import pyagrum as gum
import yaml

IN_PYTEST = "pytest" in sys.modules


# Read configuration for experiment
def load_config(name: str):

    subdir = "test" if os.getenv("USE_TEST_CONFIG") == "1" else "experiments"

    config_path = get_root_path() / subdir / name / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


# Set global seed
def set_global_seed(seed: int):

    random.seed(seed)
    gum.initRandom(seed)


# Create an empty directory
def create_clean_dir(path: Path):

    # Remove the folder if already exists
    if path.exists() and path.is_dir():
        shutil.rmtree(path)

    # Create a new folder
    path.mkdir(parents=True, exist_ok=True)


# Get output path
def get_out_path(config):

    root_path = get_root_path()
    out_path = config["out_path"]

    return root_path / out_path


# Get root directory
def get_root_path():
    return Path(__file__).resolve().parents[1]


# Only perform an `assert` if code is running in `pytest`
def safe_assert(condition):
    if IN_PYTEST:
        assert condition
