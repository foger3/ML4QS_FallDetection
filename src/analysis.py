import pandas as pd
from DistributionBasedOutlierDetection import DistributionBasedOutlierDetection
from MissingDataInterpolation import impute_interpolate

def describe(df: pd.DataFrame)->pd.DataFrame:
    movement_num = df[df["Time difference (s)"] == 0].shape[0]
    print(f"The number of movement: {movement_num}")
    movement_start_idx = df[df["Time difference (s)"] == 0].index
    label_names = df.columns[df.columns.str.startswith('Label')]
    df_start = df.loc[movement_start_idx]
    movement_cnts = []
    for name in label_names:
        movement_cnt = df_start[df_start[name] == 1].shape[0]
        print("  - The number of {:<14}:{:>3}".format(name, movement_cnt))
        movement_cnts.append(movement_cnt)

    return pd.DataFrame([movement_num] + movement_cnts, columns = ["count"], index = ["Total"] + list(label_names))

filename = "/Users/lucat/OneDrive/Dokumente/GitHub/ML4QS_FallDetection/dataset/data_cleaned.csv"
df = pd.read_csv(filename)

# Descriptive Analysis
_ = describe(df)


# Outlier Analysis
outlier_distribution = DistributionBasedOutlierDetection()

chauvenet_df = outlier_distribution.chauvenet(df, df.columns[2:15])
chauvenet_df.sum()

mixture_df = outlier_distribution.mixture_model(df, df.columns[2:15])


# Missing Data Imputation
df = impute_interpolate(df, df.columns[2:15])