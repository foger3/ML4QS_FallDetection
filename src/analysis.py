import copy
import pandas as pd

from descriptives import describe
from outlier_detection import OutlierDetectionDistribution
from data_transformation import DataTransformation
from feature_engineering import FeatureAbstraction
from non_temporal_modelling import ClassificationPrepareData, ClassificationFeatureSelection, ClassificationEvaluation, ClassificationAlgorithms


## ANALYSIS SECTION: Combining function from other modules ##
### Read in cleaned data and defined reoccruing objects ###
df = pd.read_csv("../dataset/data_cleaned.csv")
sensor_columns = [col for col in df.columns[2:15] if "Linear" not in col] # [col for col in df.columns[2:15]]
label_columns = [col for col in df.columns[15:]] 
milliseconds_per_instance = df.loc[1, "Time difference (s)"] * 1000 # Compute number of milliseconds covered by an instance

# Initialize classes
prepare = ClassificationPrepareData()
feature_select = ClassificationFeatureSelection()
eval = ClassificationEvaluation()
learner = ClassificationAlgorithms()

### Descriptive Analysis ###
_ = describe(df)


### Noise Handling ###
## Outlier Analysis
outlier = OutlierDetectionDistribution(df, sensor_columns)
chauvenet_df = outlier.chauvenet(C=2)
# outlier.visualize_chauvenet_outlier(chauvenet_df)
mixture_df = outlier.mixture_model(n_components=3)

## Missing & General Data Transformation
transform = DataTransformation(df, sensor_columns)
# df = transform.impute_interpolate()

granularity = 10
df = transform.low_pass_filter(sampling_frequency=(1000 / granularity), 
                               cutoff_frequency=1.5,
                               order=10,
                               phase_shift=True)
# transform.visualize_low_pass()

### Feature Engineering ###
# Initialize the window sizes to the number of instances representing 5 seconds
features = FeatureAbstraction(window_size=int(float(5000)/milliseconds_per_instance),
                              sampling_rate=float(1000)/milliseconds_per_instance) # important for frequency domain

## Temporal Domain
for feature in ["mean", "std", "median", "min", "max", "sem"]: #, "slope"] # slope takes very long
    df = features.abstract_numerical(df, sensor_columns, feature)

## Frequency Domain
df = features.abstract_frequency(copy.deepcopy(df), sensor_columns)

# The percentage of overlap we allow: 95%
window_overlap = 0.95
skip_points = int((1-window_overlap) * int(float(5000)/milliseconds_per_instance))
final_df = df.iloc[::skip_points,:].reset_index(drop=True)


### Non-temporal Predictive Modelling ###
# Prepare Classification Dataset for Training###
train_X, test_X, train_y, test_y = prepare.split_classification(final_df, label_columns, 'like', 0.7, filter=True, temporal=False)

print('Training set length is: ', len(train_X.index))
print('Test set length is: ', len(test_X.index))

# Feature Selection
max_features = 15
selected_features, ordered_features, ordered_scores = feature_select.forward_selection(max_features,
                                                                                       train_X,
                                                                                       test_X,
                                                                                       train_y,
                                                                                       test_y,
                                                                                       gridsearch=False)

# selected_features = ['Accelerometer Y (m/s^2)_temp_min_ws_500', 
#                      'Accelerometer X (m/s^2)_temp_min_ws_500', 
#                      'Accelerometer Y (m/s^2)_freq_0.0_Hz_ws_500', 
#                      'Accelerometer Z (m/s^2)_temp_max_ws_500', 
#                      'Magnetometer Z (µT)_temp_median_ws_500', 
#                      'Accelerometer Z (m/s^2)_temp_min_ws_500', 
#                      'Accelerometer X (m/s^2)', 
#                      'Accelerometer Z (m/s^2)_temp_median_ws_500', 
#                      'Magnetometer Z (µT)_temp_min_ws_500', 
#                      'Accelerometer X (m/s^2)_temp_mean_ws_500', 
#                      'Magnetometer X (µT)_freq_48.6_Hz_ws_500', 
#                      'Accelerometer Y (m/s^2)_freq_2.6_Hz_ws_500', 
#                      'Gyroscope X (rad/s)_freq_5.8_Hz_ws_500', 
#                      'Accelerometer X (m/s^2)_freq_3.4_Hz_ws_500', 
#                      'Accelerometer X (m/s^2)_freq_24.4_Hz_ws_500']

selected_train_X = train_X[selected_features]
selected_test_X = test_X[selected_features]

cv_rep = 5
performance_tr_nn, performance_te_nn = 0, 0
performance_tr_rf, performance_te_rf = 0, 0
performance_tr_svm, performance_te_svm = 0, 0

for repeat in range(cv_rep):
    # print("Training NeuralNetwork run {} / {} ... ".format(repeat, cv_rep))
    # class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
    #     selected_train_X, train_y, selected_test_X, gridsearch=True
    # )
    # performance_tr_nn += eval.accuracy(train_y, class_train_y)
    # performance_te_nn += eval.accuracy(test_y, class_test_y)
    
    print("Training RandomForest run {} / {} ... ".format(repeat, cv_rep))
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
        selected_train_X, train_y, selected_test_X, gridsearch=True
    )
    performance_tr_rf += eval.accuracy(train_y, class_train_y)
    performance_te_rf += eval.accuracy(test_y, class_test_y)

    # print("Training SVM run {} / {} ... ".format(repeat, cv_rep))
    # class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.support_vector_machine_with_kernel(
    #     selected_train_X, train_y, selected_test_X, gridsearch=True
    # )
    # performance_tr_svm += eval.accuracy(train_y, class_train_y)
    # performance_te_svm += eval.accuracy(test_y, class_test_y)

# overall_performance_tr_nn = performance_tr_nn/cv_rep
# overall_performance_te_nn = performance_te_nn/cv_rep
overall_performance_tr_rf = performance_tr_rf/cv_rep
overall_performance_te_rf = performance_te_rf/cv_rep
# overall_performance_tr_svm = performance_tr_svm/cv_rep
# overall_performance_te_svm = performance_te_svm/cv_rep
