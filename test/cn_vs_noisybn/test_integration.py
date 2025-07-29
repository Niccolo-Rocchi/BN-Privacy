from src.config import get_config
from src.data import generate_naivebayes
from src.exp_run import run_cn_vs_noisybn

def test_exp():

    # Load config
    config = get_config("test/cn_vs_noisybn/config.yaml")

    # Generate BNs and data
    generate_naivebayes(config)

    # Run experiment
    run_cn_vs_noisybn(config)