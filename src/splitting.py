from sklearn.model_selection import train_test_split
from pandas import DataFrame

def train_test_split_data(df: DataFrame, test_size: float=0.2):
    """Split features and target into train/test sets"""
    df['Risk'] = df['Risk'].replace({'good': 1, 'bad': 0})
    X = df.drop(columns=['Risk'])
    y = df['Risk']
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)