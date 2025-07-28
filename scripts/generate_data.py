from src.data import generate_data
from src.config import get_config

if __name__ == "__main__":

    # Load config
    config = get_config("configs/config1.yaml")
    
    # Generate BNs and data
    generate_data(config)