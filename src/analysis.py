import copy
import pandas as pd

from descriptives import describe
from outlier_detection import DistributionBasedOutlierDetection
from missing_data import impute_interpolate
from data_transformation import low_pass_filter
from feature_engineering import TemporalAbstraction, FourierTransformation
from non_temporal_modelling import ClassificationPrepareData, ClassificationFeatureSelection, ClassificationEvaluation, ClassificationAlgorithms


## ANALYSIS SECTION: Combining function from other modules ##
### Read in cleaned data ###
df = pd.read_csv("../dataset/data_cleaned.csv")
sensor_columns = [col for col in df.columns[2:5]] # [col for col in df.columns[2:15]]
label_columns = [col for col in df.columns[15:]] 

# Initialize classes
outlier_distribution = DistributionBasedOutlierDetection()
temporal_features = TemporalAbstraction()
frequency_features = FourierTransformation()
prepare = ClassificationPrepareData()
feature_select = ClassificationFeatureSelection()
eval = ClassificationEvaluation()
learner = ClassificationAlgorithms()

### Descriptive Analysis ###
_ = describe(df)


### Noise Handling ###
## Outlier Analysis
chauvenet_df = outlier_distribution.chauvenet(df, sensor_columns)
chauvenet_df.sum()
mixture_df = outlier_distribution.mixture_model(df, sensor_columns)

## Missing Data Imputation
df = impute_interpolate(df, sensor_columns)

## Data Transformation
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

# Merge final_df with label columns from original dataframe (df)
final_df = pd.merge(filter_df, df[label_columns], left_index=True, right_index=True)

# The percentage of overlap we allow
window_overlap = 0.95
skip_points = int((1-window_overlap) * window_sizes[0])
final_df = final_df.iloc[::skip_points,:].reset_index(drop=True)


### Non-temporal Predictive Modelling ###
# Prepare Classification Dataset for Training###
train_X, test_X, train_y, test_y = prepare.split_classification(final_df, label_columns, 'like', 0.7, filter=True, temporal=False)

print('Training set length is: ', len(train_X.index))
print('Test set length is: ', len(test_X.index))

# Feature Selection
max_features = 10
selected_features, ordered_features, ordered_scores = feature_select.forward_selection(max_features,
                                                                                       train_X,
                                                                                       test_X,
                                                                                       train_y,
                                                                                       test_y,
                                                                                       gridsearch=False)

# selected_features = ['Magnetometer Z (µT)_temp_max_ws_500', 
#                      'Magnetometer Y (µT)_freq_0.0_Hz_ws_500', 
#                      'Magnetometer Z (µT)_temp_min_ws_500', 
#                      'Magnetometer Z (µT)_freq_0.8_Hz_ws_500', 
#                      'Magnetometer X (µT)_temp_min_ws_500', 
#                      'Magnetometer Y (µT)_temp_min_ws_500', 
#                      'Magnetometer Z (µT)_freq_0.0_Hz_ws_500', 
#                      'Magnetometer Z (µT)_freq_1.4_Hz_ws_500', 
#                      'Magnetometer Z (µT)_freq_0.6_Hz_ws_500', 
#                      'Magnetometer X (µT)_temp_median_ws_500']

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
    
    # print("Training RandomForest run {} / {} ... ".format(repeat, cv_rep))
    # class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
    #     selected_train_X, train_y, selected_test_X, gridsearch=True
    # )
    # performance_tr_rf += eval.accuracy(train_y, class_train_y)
    # performance_te_rf += eval.accuracy(test_y, class_test_y)

    print("Training SVM run {} / {} ... ".format(repeat, cv_rep))
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.support_vector_machine_with_kernel(
        selected_train_X, train_y, selected_test_X, gridsearch=True
    )
    performance_tr_svm += eval.accuracy(train_y, class_train_y)
    performance_te_svm += eval.accuracy(test_y, class_test_y)

# overall_performance_tr_nn = performance_tr_nn/cv_rep
# overall_performance_te_nn = performance_te_nn/cv_rep
# overall_performance_tr_rf = performance_tr_rf/cv_rep
# overall_performance_te_rf = performance_te_rf/cv_rep
overall_performance_tr_svm = performance_tr_svm/cv_rep
overall_performance_te_svm = performance_te_svm/cv_rep
