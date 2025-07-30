from src.config import get_config
from src.data import generate_randombn
from src.exp_run import run_cn_privacy    

def test_integration():

    # Load config
    config = get_config("test/cn_privacy/config.yaml")

    # Generate BNs and data
    generate_randombn(config)

    # Run experiment
    run_cn_privacy(config)