import argparse
import os
from data_clean_label import aggregate
from analysis import generate_final_dataset, applied_model
from miscellaneous import logger


parser = argparse.ArgumentParser(
    prog="python3 main.py",
    description="""Aggregate the dataset from raw data 
        and apply feature engineering technique based on given arguments.
        Also, train and test a ML model based on given arguments.
        """,
)

parser.add_argument(
    "-a",
    "--aggregate",
    action="store_true",
    help="convert the raw data to an aggregated data format",
)
parser.add_argument(
    "-g",
    "--granularity",
    action="store",
    metavar="N",
    default=10,
    type=int,
    help="the granularity in milliseconds dataset (default: 10)",
)

parser.add_argument(
    "-s",
    "--split",
    action="store",
    choices=["temporal", "non-temporal"],
    default="temporal",
    type=str,
    help="the train test set split method (default: temporal)",
)

parser.add_argument(
    "-m",
    "--matching",
    action="store",
    choices=["like", "binary"],
    default="like",
    type=str,
    help="the 'like' means 7 classes classification (wallking, running, lying, standing, sitting, kneeling, and falling), and 'binary' means binary classification (falling and non-falling) (default: like)",
)

parser.add_argument(
    "-o",
    "--outlier",
    action="store_true",
    help="apply outlier detection on dataset",
)

parser.add_argument(
    "-l",
    "--lowpass",
    action="store_true",
    help="apply low-pass filter on dataset",
)


parser.add_argument(
    "-n",
    "--model-name",
    action="store",
    choices=["rf", "svm", "lstm", "conv_lstm", "time_conv_lstm", "gru"],
    required=True,
    type=str,
    help="the name of machine learning model",
)

if __name__ == "__main__":
    args = parser.parse_args()

    ### convert the raw data to an aggregated data format ###
    if args.aggregate:
        logger.info(
            "Start converting the raw data to an aggregated data format (granularity = %s ms) ...",
            args.granularity,
        )
        aggregate(args.granularity)
        logger.info("Data aggregation succeed.")
    else:
        if not os.path.isfile(f"../dataset/data_cleaned_{args.granularity}.csv"):
            logger.error(
                "Aggregated data with granularity %s ms not existed. Please specify the `-a` option to aggregate data with desired granularity.",
                args.granularity,
            )
            parser.print_usage()
            exit(-1)

    ### feature engineering ###
    logger.info(
        "Start feature engineering (%s outlier detection | %s low-pass filter) ...",
        "apply" if args.outlier else "no",
        "apply" if args.lowpass else "no",
    )
    temporal_split = args.split == "temporal"
    for_temporal_model = args.model_name in [
        "lstm",
        "conv_lstm",
        "time_conv_lstm",
        "gru",
    ]
    df, selected_features = generate_final_dataset(
        args.granularity,
        args.matching,
        temporal_split,
        args.outlier,
        args.lowpass,
        for_temporal_model,
    )
    logger.info("Feature engineering succeed.")
    
    ### model ###
    logger.info(
        "Start model generation (%s split | %s model) ...",
        args.split,
        args.model_name,
    )
    applied_model(
        args.granularity,
        args.model_name,
        df,
        selected_features,
        args.matching,
        temporal_split,
    )
    logger.info("Model generation succeed.")
