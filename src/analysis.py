import copy
import pandas as pd

from descriptives import describe
from outlier_detection import OutlierDetectionDistribution
from data_transformation import DataTransformation
from feature_engineering import FeatureAbstraction
from non_temporal_modelling import ClassificationProcedure, ClassificationEvaluation


## ANALYSIS SECTION: Combining function from other modules ##
### Read in cleaned data and defined reoccruing objects ###
df = pd.read_csv("../dataset/data_cleaned.csv")
sensor_columns = [
    col for col in df.columns[3:16] if "Linear" not in col
]  # [col for col in df.columns[2:15]]
label_columns = [col for col in df.columns[16:]]
milliseconds_per_instance = (
    df.loc[1, "Time difference (s)"] * 1000
)  # Compute number of milliseconds covered by an instance
binary = True # Binarize Classification(?)

### Descriptive Analysis ###
_ = describe(df)

# Reduce df to relevant data (cut out time and linear acceleration - essentially same as accelerometer)
df = df[["ID"] + sensor_columns + label_columns]
# [col for col in df.columns]


### Noise Handling ###
## Outlier Analysis
outlier = OutlierDetectionDistribution(df, sensor_columns)
chauvenet_df = outlier.chauvenet(C=2)
# outlier.visualize_chauvenet_outlier(chauvenet_df)
mixture_df = outlier.mixture_model(n_components=3)

## Missing & General Data Transformation
transform = DataTransformation()
# df = transform.impute_interpolate(df, sensor_columns)

granularity = 10
df = transform.low_pass_filter(
    df, sensor_columns, sampling_frequency=(1000 / granularity), cutoff_frequency=1.5
)


### Feature Engineering ###
# Initialize the window sizes to the number of instances representing 5 seconds
features = FeatureAbstraction(
    window_size=int(float(5000) / milliseconds_per_instance),
    sampling_rate=float(1000) / milliseconds_per_instance,
)  # important for frequency domain

## Temporal Domain
for feature in [
    "mean",
    "std",
    "median",
    "min",
    "max",
    "sem",
]:  # , "slope"] # slope takes very long
    df = features.abstract_numerical(df, sensor_columns, feature)

## Frequency Domain
df = features.abstract_frequency(copy.deepcopy(df), sensor_columns)

## Overlap: The percentage of overlap we allow: 95%
window_overlap = 0.95
skip_points = int((1 - window_overlap) * int(float(5000) / milliseconds_per_instance))
final_df = df.iloc[::skip_points, :].reset_index(drop=True)

## PCA
final_df = features.abstract_features_with_pca(final_df, label_columns, n_components=10)


### Feature Selection ###
class_fall = ClassificationProcedure(final_df, label_columns, binary)
selected_features, _, _ = class_fall.forward_selection(max_features=2)

selected_features = [
    "Accelerometer Y (m/s^2)_temp_min_ws_500",
    "Accelerometer X (m/s^2)_temp_min_ws_500",
    "Accelerometer Y (m/s^2)_freq_0.0_Hz_ws_500",
    "Accelerometer Z (m/s^2)_temp_max_ws_500",
    "Magnetometer Z (µT)_temp_median_ws_500",
    "Accelerometer Z (m/s^2)_temp_min_ws_500",
    "Accelerometer X (m/s^2)",
    "Accelerometer Z (m/s^2)_temp_median_ws_500",
    "Magnetometer Z (µT)_temp_min_ws_500",
    "Accelerometer X (m/s^2)_temp_mean_ws_500",
    "Magnetometer X (µT)_freq_48.6_Hz_ws_500",
    "Accelerometer Y (m/s^2)_freq_2.6_Hz_ws_500",
    "Gyroscope X (rad/s)_freq_5.8_Hz_ws_500",
    "Accelerometer X (m/s^2)_freq_3.4_Hz_ws_500",
    "Accelerometer X (m/s^2)_freq_24.4_Hz_ws_500",
]


### Non-temporal Predictive Modelling ###
class_pro = ClassificationProcedure(final_df, label_columns, binary, selected_features)
class_eval = ClassificationEvaluation()

performance_tr_nn, performance_te_nn = 0, 0
performance_tr_rf, performance_te_rf = 0, 0
performance_tr_svm, performance_te_svm = 0, 0

n_cv_rep = 1
for repeat in range(n_cv_rep):
    print("Training NeuralNetwork run {} / {} ... ".format(repeat, n_cv_rep))
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = class_pro.feedforward_neural_network(gridsearch=True)

    performance_tr_nn += class_eval.accuracy(class_pro.train_y, class_train_y)
    performance_te_nn += class_eval.accuracy(class_pro.test_y, class_test_y)

    print("Training RandomForest run {} / {} ... ".format(repeat, n_cv_rep))
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = class_pro.random_forest(gridsearch=True)

    performance_tr_rf += class_eval.accuracy(class_pro.train_y, class_train_y)
    performance_te_rf += class_eval.accuracy(class_pro.test_y, class_test_y)

    print("Training SVM run {} / {} ... ".format(repeat, n_cv_rep))
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = class_pro.support_vector_machine(gridsearch=True)

    performance_tr_svm += class_eval.accuracy(class_pro.train_y, class_train_y)
    performance_te_svm += class_eval.accuracy(class_pro.test_y, class_test_y)

overall_performance_tr_nn = performance_tr_nn / n_cv_rep
overall_performance_te_nn = performance_te_nn / n_cv_rep
overall_performance_tr_rf = performance_tr_rf / n_cv_rep
overall_performance_te_rf = performance_te_rf / n_cv_rep
overall_performance_tr_svm = performance_tr_svm / n_cv_rep
overall_performance_te_svm = performance_te_svm / n_cv_rep

class_eval.confusion_matrix(class_pro.test_y, class_test_y, class_train_prob_y.columns)