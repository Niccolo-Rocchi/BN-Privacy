from src.config import load_config
from src.data import generate_naivebayes


def main():
    # Load config
    config = load_config("cn_vs_noisybn")

    # Generate BNs and data
    generate_naivebayes(config)


if __name__ == "__main__":

    main()
