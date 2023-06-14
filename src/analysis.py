import pandas as pd

from outlier_detection import DistributionBasedOutlierDetection
from missing_data import impute_interpolate
from data_transformation import low_pass_filter
from feature_engineering import TemporalNumericalAbstraction

def describe(
    df: pd.DataFrame
) -> pd.DataFrame:
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


## ANALYSIS SECTION: Combining function from other modules ##
# Read in cleaned data
df = pd.read_csv("../dataset/data_cleaned.csv")
sensor_columns = [col for col in df.columns[2:15]]

# Descriptive Analysis
_ = describe(df)

# Outlier Analysis
outlier_distribution = DistributionBasedOutlierDetection()

chauvenet_df = outlier_distribution.chauvenet(df, sensor_columns)
chauvenet_df.sum()

mixture_df = outlier_distribution.mixture_model(df, sensor_columns)

# Missing Data Imputation
df = impute_interpolate(df, sensor_columns)

# Data Transformation
granularity = 10
df = low_pass_filter(df, sensor_columns, 1000 / granularity, 1.5)

# Feature Engineering
temporal_features = TemporalNumericalAbstraction()

# Compute the number of milliseconds covered by an instance based on the first two rows
milliseconds_per_instance = (df.index[1] - df.index[0]).microseconds/1000