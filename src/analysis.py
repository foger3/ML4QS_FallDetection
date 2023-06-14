import copy
import pandas as pd

from descriptives import describe
from outlier_detection import DistributionBasedOutlierDetection
from missing_data import impute_interpolate
from data_transformation import low_pass_filter
from feature_engineering import TemporalAbstraction, FourierTransformation


## ANALYSIS SECTION: Combining function from other modules ##
### Read in cleaned data ###
df = pd.read_csv("../dataset/data_cleaned.csv")
sensor_columns = [col for col in df.columns[2:5]] # [col for col in df.columns[2:15]]

# Initialize classes
outlier_distribution = DistributionBasedOutlierDetection()
temporal_features = TemporalAbstraction()
frequency_features = FourierTransformation()

### Descriptive Analysis ###
_ = describe(df)


### Outlier Analysis ###
chauvenet_df = outlier_distribution.chauvenet(df, sensor_columns)
chauvenet_df.sum()

mixture_df = outlier_distribution.mixture_model(df, sensor_columns)


### Missing Data Imputation ###
df = impute_interpolate(df, sensor_columns)


### Data Transformation ###
granularity = 10
filter_df = low_pass_filter(df, sensor_columns, 1000 / granularity, 1.5)


### Feature Engineering ###
# Compute number of milliseconds covered by an instance
milliseconds_per_instance = df.loc[1, "Time difference (s)"] * 1000

## Temporal Domain
# Set the window sizes to the number of instances representing 5 seconds and 30 seconds
window_sizes = [int(float(5000)/milliseconds_per_instance)]#, int(float(30000)/milliseconds_per_instance)]
num_features = ["mean", "std", "median", "min", "max", "sem"]#, "slope"] # slope takes very long

for ws in window_sizes:
    for feature in num_features:
        filter_df = temporal_features.abstract_numerical(filter_df, sensor_columns, ws, feature)
        
## Frequency Domain
fs = float(1000)/milliseconds_per_instance
ws = int(float(5000)/milliseconds_per_instance)
filter_df = frequency_features.abstract_frequency(copy.deepcopy(filter_df), sensor_columns, ws, fs)

# The percentage of overlap we allow
window_overlap = 0.9
skip_points = int((1-window_overlap) * ws)
final_df = filter_df.iloc[::skip_points,:]


### Non-temporal Predictive Modelling ###
