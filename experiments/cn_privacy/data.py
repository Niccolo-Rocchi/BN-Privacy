from src.config import load_config
from src.data import generate_randombn


def main():
    # Load config
    config = load_config("cn_privacy")

    # Generate BNs and data
    generate_randombn(config)


if __name__ == "__main__":

    main()
