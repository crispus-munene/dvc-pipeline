from src.utils import load_params
from src.training import train_and_log

if __name__ == "__main__":
    params = load_params("params.yaml")
    train_and_log(params)
