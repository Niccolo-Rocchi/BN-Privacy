from src.config import get_config
from src.data import generate_randombn
from src.run_exp import run_cn_privacy

if __name__ == "__main__":

    # Load config
    config = get_config("experiments/cn_privacy/config.yaml")

    # Generate BNs and data
    generate_randombn(config)

    # Run experiment
    run_cn_privacy(config)
