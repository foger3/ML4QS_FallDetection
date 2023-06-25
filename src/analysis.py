import copy
import pandas as pd
import numpy as np

from descriptives import describe
from outlier_detection import OutlierDetectionDistribution
from data_transformation import DataTransformation
from feature_engineering import FeatureAbstraction
from miscellaneous import selected_features_outlier, selected_features_no_outlier
from modelling import (
    NonTemporalClassification,
    TemporalClassification,
    ClassificationEvaluation,
)


def generate_final_dataset(
    matching: str, temporal_split: bool, filter_outlier: bool, for_temporal_model: bool
) -> tuple[pd.DataFrame, list[str], list[str], float]:
    ### Read in cleaned data and defined reoccruing objects ###
    df = pd.read_csv("../dataset/data_cleaned.csv")
    sensor_columns = [  # Save the individual sensor columns
        col for col in df.columns[3:16] if "Linear" not in col
    ]
    label_columns = sorted(  # Save the invdividual label columns
        [col for col in df.columns[16:]]
    )
    milliseconds_per_instance = (  # Compute number of milliseconds covered by an instance
        df.loc[1, "Time difference (s)"] * 1000
    )

    ### Descriptive Analysis ###

    ## Let's have a look at the data (descriptive statistics)
    _ = describe(df)

    ## Reduce df to relevant data (cut out time and linear acceleration)
    df = df[["ID"] + sensor_columns + label_columns]

    ### Noise Handling ###
    if filter_outlier:
        ## Outlier Analysis: Let's see if we can find outliers based on distribution-based methods
        outlier = OutlierDetectionDistribution(df, sensor_columns)  # prepare class

        ## Outlier Detection according to Chauvenet's criterion (high as we want to preserve data)
        # chauvenet_df = outlier.chauvenet(C=10)
        # chauvenet_df.columns = sensor_columns
        ## Remove outliers from the data based on Chauvenet's criterion
        # df.loc[:, sensor_columns] = df.loc[:, sensor_columns].where(~chauvenet_df)

        ## Outlier Detection according to Mixture Model
        mixture_df = outlier.mixture_model(n_components=2)
        mixture_df.columns = sensor_columns
        ## Remove outliers from the data based on Mixture models
        df.loc[:, sensor_columns] = df.loc[:, sensor_columns].where(mixture_df >= 1e-60)

        ## How many missing values do we have now?
        print(df[df.isna().any(axis=1)].sum()[label_columns])

        ## Missing Data: As we threw out some data, we have to impute the missing values
        transform = DataTransformation(df, sensor_columns)  # prepare class
        df = transform.impute_interpolate(df, sensor_columns)

    ## Transform the data: Use if applicable
    # granularity = 10
    # df = transform.low_pass_filter(
    #     df, sensor_columns, sampling_frequency=(1000 / granularity), cutoff_frequency=5
    # )
    # transform.low_pass_filter_visualize(df, label_columns)

    ### Feature Engineering ###

    ## Next, we create some features based on the raw data
    features = FeatureAbstraction(  # prepare class
        window_size=int(
            float(5000) / milliseconds_per_instance
        ),  #  representing 5 seconds
        sampling_rate=(float(1000) / milliseconds_per_instance),
    )  # important for frequency domain

    ## Temporal Domain: Create features based on the time domain
    for feature in [
        "mean",
        "std",
        "median",
        "min",
        "max",
        "sem",
    ]:  # "slope" = takes very long and does not add much value
        df = features.abstract_numerical(df, sensor_columns, feature)

    ## Frequency Domain: Create features based on Fast Fourier Transformation
    df = features.abstract_frequency(copy.deepcopy(df), sensor_columns)
    final_df = copy.deepcopy(df) if for_temporal_model else None

    ## Overlap: The percentage of overlap we allow: 90%
    window_overlap = 0.9
    skip_points = int(
        (1 - window_overlap) * int(float(5000) / milliseconds_per_instance)
    )
    df = df.iloc[::skip_points, :].reset_index(drop=True)

    ## Further measure to account for overlap when randomizing the dataset (split: non-temporal)
    ## Idea: instead of randomizing individual instances, we randomize the chunks
    chunk_size = 10
    df["ID"] = (df.index / chunk_size + 1).astype(int)

    ## PCA: to extract more important features
    df, pca_cols = features.abstract_features_with_pca(
        df, label_columns, n_components=370
    )

    ### Feature Selection ###

    ## Select 30 most important features based on forward selection
    # class_feature = NonTemporalClassification(  # prepare class
    #     df, label_columns, matching, temporal_split
    # )
    # selected_features, _, _ = class_feature.forward_selection(max_features=30)

    ## Selected features with OR without outlier/missing data handling
    selected_features = (
        selected_features_outlier.copy()
        if filter_outlier
        else selected_features_no_outlier.copy()
    )

    if for_temporal_model:
        del df
        selected_features = [x for x in selected_features if "PCA" not in x]
        final_df = final_df[selected_features + label_columns]
    else:
        final_df = df

    return final_df, selected_features, label_columns, milliseconds_per_instance


def applied_model(
    model_name: str,
    df: pd.DataFrame,
    selected_features: list[str],
    label_columns: list[str],
    matching: str,
    temporal_split: bool,
    milliseconds_per_instance: float,
) -> None:
    #### ANALYSIS SECTION ####

    class_eval = ClassificationEvaluation()  # prepare class

    ### Non-temporal Predictive Modelling ###

    if model_name in ["rf", "svm"]:
        # Apply Random Forest and SVM as our non-temporal predictive models
        class_pro = NonTemporalClassification(  # prepare class
            df,
            label_columns,
            matching,
            temporal_split,
            selected_features,
        )

        n_cv_rep = 10  # number of cross-validation repetitions

        # performance_tr_rf, performance_te_rf = 0, 0
        performance_tr, performance_te = 0, 0
        cm_te = np.zeros((len(label_columns), len(label_columns)))
        for repeat in range(n_cv_rep):
            name = None
            class_train_y = None
            class_test_y = None
            class_train_prob_y = None
            class_test_prob_y = None
            if model_name == "rf":
                name = "RandomForest"
                print(f"Training {name} run {repeat} / {n_cv_rep} ... ")
                (
                    class_train_y,
                    class_test_y,
                    class_train_prob_y,
                    class_test_prob_y,
                ) = class_pro.random_forest(gridsearch=True)
            else:
                name = "SVM"
                print(f"Training {name} run {repeat} / {n_cv_rep} ... ")
                (
                    class_train_y,
                    class_test_y,
                    class_train_prob_y,
                    class_test_prob_y,
                ) = class_pro.support_vector_machine(gridsearch=True)

            performance_tr += class_eval.accuracy(class_pro.train_y, class_train_y)
            performance_te += class_eval.accuracy(class_pro.test_y, class_test_y)
            cm = class_eval.confusion_matrix(
                class_pro.test_y, class_test_y, class_train_prob_y.columns
            )
            cm_te += cm

        print(
            f"{name} train accurancy ({n_cv_rep} times average) : {performance_tr / n_cv_rep}"
        )
        print(
            f"{name} test accurancy ({n_cv_rep} times average) : {performance_te / n_cv_rep}"
        )

        # Visualize the confusion matrix (evaluation)
        class_eval.confusion_matrix_visualize(
            cm_te / n_cv_rep,
            [col.split(" ")[1] for col in class_train_prob_y.columns],
            f"./cm_{model_name}.png",
        )

    elif model_name in ["lstm", "conv_lstm", "time_conv_lstm", "gru"]:
        ### Temporal Predictive Modelling ###

        # Apply versions of LSTM and a GRU as our temporal predictive models

        # Fitting Procedure
        class_nn = TemporalClassification(
            df,
            label_columns,
            step=10,  # controls overlap (fraction of time_interval)
            time_intervals=int(float(1000) / milliseconds_per_instance),
        )

        ## Manual Fitting Example (fitting step-by-step)
        # class_nn.conv_lstm(print_model_details=False)
        # model, hist = class_nn.fit(epochs=10, batch_size=128)
        # result = class_nn.prediction(model)

        # Automatic Fitting (fitting is done automatically)
        _, _, result = class_nn.fit_predict(model=model_name, epochs=10, batch_size=128)

        # Visualiue the confusion matrix (evaluation)
        class_eval.confusion_matrix_visualize(
            result,
            [col.split(" ")[1] for col in label_columns],
            f"./cm_{model_name}.png",
        )

    else:
        print(f"{model_name} not found!")


if __name__ == "__main__":
    # Define classification and data split type (defaults of our analysis are selected)
    matching = "like"  # "binary" for binary classification
    temporal_split = True  # False for non-temporal train/test split
    filter_outlier = False
    for_temporal_model = True

    (
        df,
        selected_features,
        label_columns,
        milliseconds_per_instance,
    ) = generate_final_dataset(
        matching, temporal_split, filter_outlier, for_temporal_model
    )

    applied_model(
        "conv_lstm",
        df,
        selected_features,
        label_columns,
        matching,
        temporal_split,
        milliseconds_per_instance,
    )
