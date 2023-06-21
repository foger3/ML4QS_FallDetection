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

# Define whether to use binary classification / temporal split
matching = "like" # "like" for multi-class classification
temporal = True # False for non-temporal train/test split


### Descriptive Analysis ###
_ = describe(df)

# Reduce df to relevant data (cut out time and linear acceleration - essentially same as accelerometer)
df = df[["ID"] + sensor_columns + label_columns]
# df = df[sensor_columns + label_columns]


### Noise Handling ###
## Outlier Analysis
outlier = OutlierDetectionDistribution(df, sensor_columns)
chauvenet_df = outlier.chauvenet(C=10)
# outlier.chauvenet_visualize(chauvenet_df)
mixture_df = outlier.mixture_model(n_components=3)

## Missing & General Data Transformation
transform = DataTransformation(df, sensor_columns)
df = transform.impute_interpolate(df, sensor_columns)

granularity = 10
df = transform.low_pass_filter(df, sensor_columns, sampling_frequency=(1000 / granularity), cutoff_frequency=1.5)
# transform.low_pass_filter_visualize(df, label_columns)

### Feature Engineering ###
# Initialize the window sizes to the number of instances representing 5 seconds
features = FeatureAbstraction(
    window_size=int(float(5000) / milliseconds_per_instance),
    sampling_rate=(float(1000) / milliseconds_per_instance),
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
# df = pd.read_pickle("c:\\Users\\lucat\\OneDrive\\Dokumente\\Uni Amsterdam\\ml4qs_full_df.pkl")

## Overlap: The percentage of overlap we allow: 90%
window_overlap = 0.9
skip_points = int((1 - window_overlap) * int(float(5000) / milliseconds_per_instance))
final_df = df.iloc[::skip_points, :].reset_index(drop=True)
# Further measure to account for overlap when randomizing the dataset
chunk_size = 100
final_df["ID"]= (final_df.index / chunk_size + 1).astype(int)

## PCA
final_df, pca_cols = features.abstract_features_with_pca(final_df, label_columns, n_components=370)


### Feature Selection ###
class_feature = ClassificationProcedure(
    final_df, label_columns, matching, temporal
)
selected_features, _, _ = class_feature.forward_selection(max_features=30)

selected_features = ['Accelerometer Y (m/s^2)_temp_min_ws_500', 
                     'Accelerometer Z (m/s^2)_temp_mean_ws_500', 
                     'Accelerometer Y (m/s^2)_freq_0.0_Hz_ws_500', 
                     'Accelerometer Y (m/s^2)_temp_max_ws_500', 
                     'Magnetometer Z (µT)_temp_mean_ws_500', 
                     'Accelerometer Z (m/s^2)_freq_26.6_Hz_ws_500', 
                     'Accelerometer Y (m/s^2)_freq_47.0_Hz_ws_500',
                     'Magnetometer X (µT)_freq_32.8_Hz_ws_500',
                     'Magnetometer Z (µT)_temp_max_ws_500',
                     'Barometer X (hPa)_freq_6.0_Hz_ws_500',
                     'Gyroscope Z (rad/s)_freq_2.0_Hz_ws_500', 
                     'Accelerometer Z (m/s^2)_freq_43.0_Hz_ws_500', 
                     'Magnetometer Y (µT)_freq_20.4_Hz_ws_500', 
                     'Magnetometer X (µT)_freq_45.0_Hz_ws_500', 
                     'Magnetometer Y (µT)_freq_49.0_Hz_ws_500', 
                     'Accelerometer Z (m/s^2)_freq_39.0_Hz_ws_500', 
                     'Magnetometer X (µT)_freq_2.6_Hz_ws_500', 
                     'PCA_Component_138', 
                     'Magnetometer Z (µT)_freq_50.0_Hz_ws_500', 
                     'Barometer X (hPa)_freq_34.0_Hz_ws_500', 
                     'Magnetometer X (µT)_freq_24.4_Hz_ws_500', 
                     'Magnetometer Z (µT)_freq_46.0_Hz_ws_500', 
                     'Accelerometer X (m/s^2)_freq_7.2_Hz_ws_500', 
                     'Accelerometer Z (m/s^2)_freq_33.0_Hz_ws_500',
                     'Gyroscope Z (rad/s)_freq_48.4_Hz_ws_500', 
                     'Barometer X (hPa)_freq_20.8_Hz_ws_500', 
                     'Barometer X (hPa)_freq_39.4_Hz_ws_500', 
                     'Magnetometer Z (µT)_freq_7.4_Hz_ws_500', 
                     'Magnetometer X (µT)_freq_41.6_Hz_ws_500', 
                     'Accelerometer Z (m/s^2)_freq_36.0_Hz_ws_500']


### Non-temporal Predictive Modelling ###
class_pro = ClassificationProcedure(
    final_df, label_columns, matching, temporal, selected_features
)
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

cm = class_eval.confusion_matrix(class_pro.test_y, class_test_y, class_train_prob_y.columns)
class_eval.confusion_matrix_visualize(cm, [col.split(" ")[1] for col in class_train_prob_y.columns], "./cm_rf.png")


# Old selected features
# selected_features = ["Accelerometer Y (m/s^2)_temp_min_ws_500",
#                      "Magnetometer Z (µT)",
#                      "Magnetometer Z (µT)_temp_std_ws_500",
#                      "Magnetometer Z (µT)_temp_mean_ws_500",
#                      "Barometer X (hPa)_freq_0.2_Hz_ws_500",
#                      "Accelerometer Y (m/s^2)_pse",
#                      "Magnetometer Y (µT)_freq_43.2_Hz_ws_500",
#                      "Magnetometer Z (µT)_freq_35.0_Hz_ws_500",
#                      "Accelerometer Z (m/s^2)_freq_47.0_Hz_ws_500",
#                      "Accelerometer Y (m/s^2)_freq_43.0_Hz_ws_500",
#                      "Barometer X (hPa)_freq_4.2_Hz_ws_500",
#                      "Magnetometer Z (µT)_freq_5.4_Hz_ws_500",
#                      "Barometer X (hPa)_freq_6.6_Hz_ws_500",
#                      "Magnetometer Y (µT)_freq_40.6_Hz_ws_500",
#                      "Accelerometer X (m/s^2)_freq_44.0_Hz_ws_500",
#                      "Gyroscope X (rad/s)_freq_21.4_Hz_ws_500",
#                      "Magnetometer Y (µT)_freq_8.4_Hz_ws_500",
#                      "Accelerometer Y (m/s^2)_freq_12.8_Hz_ws_500",
#                      "Gyroscope Z (rad/s)_freq_17.0_Hz_ws_500",
#                      "PCA_Component_1",
#                      "Accelerometer Z (m/s^2)_freq_2.2_Hz_ws_500",
#                      "Magnetometer X (µT)_freq_14.6_Hz_ws_500",
#                      "Accelerometer Y (m/s^2)_freq_44.0_Hz_ws_500",
#                      "Gyroscope X (rad/s)_freq_32.2_Hz_ws_500",
#                      "Gyroscope X (rad/s)_freq_12.8_Hz_ws_500",
#                      "Magnetometer Z (µT)_freq_11.8_Hz_ws_500",
#                      "Gyroscope Z (rad/s)_freq_11.2_Hz_ws_500",
#                      "Barometer X (hPa)_freq_48.6_Hz_ws_500",
#                      "Accelerometer Y (m/s^2)_freq_35.0_Hz_ws_500",
#                      "Accelerometer X (m/s^2)_freq_34.4_Hz_ws_500",]