# Interpolate the dataset based on previous/next values..
def impute_interpolate(df, cols):
    df[cols] = df[cols].interpolate().ffill().bfill()
    return df
