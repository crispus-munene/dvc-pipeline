import pandas as pd
from pandas import DataFrame

def load_data(path: str) -> DataFrame:
    """Load CSV dataset into a DataFrame"""
    return pd.read_csv(path)

if __name__ == "__main__":
    df = load_data("./data/german_credit_data.csv")
    print(df.head())
