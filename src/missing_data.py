import pandas as pd

# Interpolate the dataset based on previous/next values..
def impute_interpolate(
    df: pd.DataFrame, 
    columns: list[str]
) -> pd.DataFrame:
    df[columns] = df[columns].interpolate().ffill().bfill()

    return df
