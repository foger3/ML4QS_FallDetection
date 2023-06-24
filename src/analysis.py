import copy
import pandas as pd
import numpy as np

from descriptives import describe
from outlier_detection import OutlierDetectionDistribution
from data_transformation import DataTransformation
from feature_engineering import FeatureAbstraction
from miscellaneous import selected_features_outlier, selected_features_no_outlier 
from modelling import NonTemporalClassification, TemporalClassification, ClassificationEvaluation
class_eval = ClassificationEvaluation() # prepare class


                                #### ANALYSIS SECTION ####

### Read in cleaned data and defined reoccruing objects ###
df = pd.read_csv("../dataset/data_cleaned.csv")
sensor_columns = [              # Save the individual sensor columns
    col for col in df.columns[3:16] if "Linear" not in col
]
label_columns = sorted(         # Save the invdividual label columns
    [col for col in df.columns[16:]]
)
milliseconds_per_instance = (   # Compute number of milliseconds covered by an instance
    df.loc[1, "Time difference (s)"] * 1000
)

# Define classification and data split type (defaults of our analysis are selected)
matching = "like"   # "binary" for binary classification
temporal = True     # False for non-temporal train/test split



                                ### Descriptive Analysis ###

# Let's have a look at the data
_ = describe(df)

# Reduce df to relevant data (cut out time and linear acceleration)
df = df[["ID"] + sensor_columns + label_columns]



                                ### Noise Handling ###

## Outlier Analysis: Let's see if we can find outliers based on distribution-based methods
outlier = OutlierDetectionDistribution(df, sensor_columns) # prepare class

# Outlier Detection according to Chauvenet's criterion (high as we want to preserve data)
# chauvenet_df = outlier.chauvenet(C=10)
# outlier.chauvenet_visualize(chauvenet_df)
# chauvenet_df.columns = sensor_columns
# df.loc[:,sensor_columns] = df.loc[:,sensor_columns].where(~chauvenet_df)

# Outlier Detection according to Mixture Model
mixture_df = outlier.mixture_model(n_components=2)
mixture_df.columns = sensor_columns
df.loc[:,sensor_columns] = df.loc[:,sensor_columns].where(mixture_df >= 1e-60)
print(df[df.isna().any(axis=1)].sum()[label_columns])

## Missing Data: As we threw out some data, we have to impute the missing values
transform = DataTransformation(df, sensor_columns) # prepare class
df = transform.impute_interpolate(df, sensor_columns)

## Transform the data: Led to worse results so we did not use it
# granularity = 10
# df = transform.low_pass_filter(df, sensor_columns, sampling_frequency=(1000 / granularity), cutoff_frequency=1.5)
# transform.low_pass_filter_visualize(df, label_columns)



                                ### Feature Engineering ###

# Next, we create some features based on the raw data
features = FeatureAbstraction( # prepare class
    window_size=int(float(5000) / milliseconds_per_instance), #  representing 5 seconds
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

# If available, load in the full dataset (saves time)
# df = pd.read_pickle("c:\\Users\\lucat\\OneDrive\\Documents\\Uni Amsterdam\\ml4qs_full_df.pkl")

## Overlap: The percentage of overlap we allow: 90%
window_overlap = 0.9
skip_points = int((1 - window_overlap) * int(float(5000) / milliseconds_per_instance))
final_df = df.iloc[::skip_points, :].reset_index(drop=True)

# Further measure to account for overlap when randomizing the dataset (split: non-temporal)
# Idea: instead of randomizing individual instances, we randomize the chunks
chunk_size = 10
final_df["ID"]= (final_df.index / chunk_size + 1).astype(int)

## PCA: to extract more important features
final_df, pca_cols = features.abstract_features_with_pca(final_df, label_columns, n_components=370)



                                ### Feature Selection ###

# Select 30 most important features based on forward selection
class_feature = NonTemporalClassification( # prepare class
    final_df, label_columns, matching, temporal
)
selected_features, _, _ = class_feature.forward_selection(max_features=30)

# Selected features with OR without outlier & missing data handling
selected_features = selected_features_outlier if True else selected_features_no_outlier



                                ### Non-temporal Predictive Modelling ###

# Apply Random Forest and SVM as our non-temporal predictive models
class_pro = NonTemporalClassification( # prepare class
    final_df, label_columns, matching, temporal, selected_features
)

performance_tr_rf, performance_te_rf = 0, 0
performance_tr_svm, performance_te_svm = 0, 0
cm_te_rf = np.zeros((len(label_columns), len(label_columns)))
cm_te_svm = np.zeros((len(label_columns), len(label_columns)))

n_cv_rep = 5
for repeat in range(n_cv_rep):
    print("Training RandomForest run {} / {} ... ".format(repeat, n_cv_rep))
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = class_pro.random_forest(gridsearch=True)

    performance_tr_rf += class_eval.accuracy(class_pro.train_y, class_train_y)
    performance_te_rf += class_eval.accuracy(class_pro.test_y, class_test_y)
    cm = class_eval.confusion_matrix(class_pro.test_y, class_test_y, class_train_prob_y.columns)
    cm_te_rf += cm

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
    cm_te_svm += cm

print("RandomForest train accurancy ({} times average) : {}".format(n_cv_rep, performance_tr_rf/n_cv_rep))
print("RandomForest test accurancy ({} times average) : {}".format(n_cv_rep, performance_te_rf/n_cv_rep))

# Visualiue the confusion matrix (evaluation)
class_eval.confusion_matrix_visualize(
    cm_te_rf/n_cv_rep, 
    [col.split(" ")[1] for col in class_train_prob_y.columns], 
    "./cm_rf.png"
)



                                ### Temporal Predictive Modelling ###

# Apply versions of LSTM and a GRU as our temporal predictive models

## Two versions of the data were tested
# 1. The original raw data
df = pd.read_csv("../dataset/data_cleaned.csv")
df = df[sensor_columns + label_columns]

# 2. The data with features extracted (temporal & frequency domain, but not PCA)
df = pd.read_pickle("c:\\Users\\lucat\\OneDrive\\Documents\\Uni Amsterdam\\ml4qs_full_df.pkl")
del selected_features[8:10] # remove PCA components as they are not available in the big df
df = df[selected_features + label_columns]

# Fitting Procedure
class_nn = TemporalClassification(
                            df, 
                            label_columns, 
                            step=10, # controls overlap (fraction of time_interval)
                            time_intervals=int(float(1000) / milliseconds_per_instance)
                        )

# Manual Fitting Example
class_nn.conv_lstm(print_model_details=False)
model, hist = class_nn.fit(epochs=10, batch_size=128)
result = class_nn.prediction(model)

# Automatic Fitting (above can be skipped, only when individual components are investigated)
_, _, result = class_nn.fit_predict(model="conv_lstm", epochs=10, batch_size=128)

# Visualiue the confusion matrix (evaluation)
class_eval.confusion_matrix_visualize(
    result, 
    [col.split(" ")[1] for col in label_columns], 
    "./cm_rf.png"
)
