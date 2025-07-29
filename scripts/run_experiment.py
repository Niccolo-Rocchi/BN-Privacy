from src.config import get_config
from src.data import generate_data
from src.experiments import run_experiment    

if __name__ == "__main__":

    # Load config
    config = get_config("configs/config1.yaml")

    # Generate BNs and data
    generate_data(config)

    # Run experiment
    run_experiment(config)
    

            