import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset and remove customerID column if present.

    Args:
        path: Path to the CSV file

    Returns:
        DataFrame without customerID column
    """
    df = pd.read_csv(path)

    # Remove customerID as it's not a feature for prediction
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    return df