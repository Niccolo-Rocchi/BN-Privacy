from src.config import get_config
from src.data import generate_data
from src.experiments import run_experiment

def test_exp():

    # Load config
    config = get_config("configs/config_test.yaml")

    # Generate BNs and data
    generate_data(config)

    # Run experiment
    run_experiment(config)