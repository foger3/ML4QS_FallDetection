from datetime import timedelta
from glob import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from labels import label_map, label_categories

def load_dataset_map(
    filepath: str,
    sensor_names: list = [
        "Magnetometer",
        "Accelerometer",
        "Linear Accelerometer",
        "Gyroscope",
        "Barometer",
    ],
):
    meta_names = ["time"]
    df_map = dict()
    for sensor_name in sensor_names:
        filename = f"{filepath}/{sensor_name}.csv"
        df_map[sensor_name] = pd.read_csv(filename)
        df_map[sensor_name].columns = list(df_map[sensor_name].columns[:1]) + [
            f"{sensor_name} {i}" for i in df_map[sensor_name].columns[1:]
        ]
    for meta_name in meta_names:
        filename = f"{filepath}/meta/{meta_name}.csv"
        df_map[meta_name] = pd.read_csv(filename)
    return df_map


def concat_sensor_dataset(
    df_map: pd.DataFrame,
    sensor_names: list = [
        "Magnetometer",
        "Accelerometer",
        "Linear Accelerometer",
        "Gyroscope",
        "Barometer",
    ],
):
    df_result = pd.DataFrame(columns=["Time (s)"])
    for sensor_name in sensor_names:
        df = df_map[str(sensor_name)]
        df_result = pd.merge(df_result, df, on="Time (s)", how="outer")
    df_result.sort_values(by=["Time (s)"], ignore_index=True, inplace=True)
    return df_result


def label_validation(df: pd.DataFrame, labels: list):
    dataset_row_num = df[df["event"] == "START"].shape[0]
    print("{} movements in this dataset.".format(dataset_row_num))
    print(
        "{} labels{}\n".format(
            len(labels),
            ", not match." if len(labels) != dataset_row_num else ", match.",
        )
    )


def labeling(
    df: pd.DataFrame, df_time: pd.DataFrame, labels: list, label_categories: list
):
    # label_validation(df_time, labels)
    # labeling
    for i in range(0, df_time.shape[0], 2):
        start_time = df_time["experiment time"][i]
        end_time = df_time["experiment time"][i + 1]
        target_idx = df[
            (df["Time (s)"] >= start_time) & (df["Time (s)"] <= end_time)
        ].index
        df.loc[target_idx, ["Label"]] = labels[i // 2]
    # one hot encode
    df = one_hot_encoding(df, "Label", label_categories)

    return df


def one_hot_encoding(df, column_name, label_categories: list):
    enc = OneHotEncoder(categories=[label_categories], sparse_output=False)
    arr_label = enc.fit_transform(df[column_name].to_numpy().reshape(-1, 1))
    df_label = pd.DataFrame(
        arr_label, columns=enc.get_feature_names_out([column_name])
    ).astype("int")
    df_label.columns = label_categories
    df.drop(columns=column_name, inplace=True)
    df = pd.concat([df, df_label], axis=1)

    return df


def adjust_granularity_timestamp_timediff(
    df: pd.DataFrame, timestamp_col: str, granularity: float, time: pd.DataFrame
) -> pd.DataFrame:
    new_timestamp = np.empty((0,), dtype="datetime64")
    for i in range(0, len(time), 2):
        start_text = pd.Timestamp(time["system time text"][i][:19])
        start = time["experiment time"][i]
        end = time["experiment time"][i + 1]
        period = df.loc[
            (df[timestamp_col] >= start) & (df[timestamp_col] < end), timestamp_col
        ]
        new_timestamp = np.concatenate(
            (new_timestamp, (start_text + pd.to_timedelta(period, unit="s")).values)
        )
    df[timestamp_col] = new_timestamp
    timestamps = pd.date_range(
        min(df[timestamp_col]),
        max(df[timestamp_col]),
        freq=f"{granularity}ms",
    )
    columns = [col for col in df.columns if col != timestamp_col]
    new_df = pd.DataFrame(index=timestamps, columns=columns, dtype=object)
    for i in range(len(new_df)):
        # Select the relevant measurements.
        relevant_rows = df[
            (df[timestamp_col] >= new_df.index[i])
            & (
                df[timestamp_col]
                < (new_df.index[i] + timedelta(milliseconds=granularity))
            )
        ]
        for col in columns:
            # Take the average value
            if len(relevant_rows) > 0:
                new_df.loc[new_df.index[i], col] = np.average(relevant_rows[col])
            else:
                new_df.loc[new_df.index[i], col] = np.nan
    new_df.insert(0, timestamp_col, new_df.index)
    new_df.insert(
        1,
        "Time difference (s)",
        (new_df[timestamp_col] - new_df[timestamp_col].iloc[0]).dt.total_seconds(),
    )
    first_valid_row = new_df.index[0]
    while 1:
        first_null_row_index = None
        for index, row in new_df.loc[first_valid_row:,].iterrows():
            if row.isnull().any():
                first_null_row_index = index
                break
        if first_null_row_index is None:
            break
        first_valid_row = new_df.loc[first_null_row_index:,].dropna().index[0]
        new_df.loc[first_valid_row:, "Time difference (s)"] = (
            new_df.loc[first_valid_row:, "Time difference (s)"]
            - new_df.loc[first_valid_row, "Time difference (s)"]
        )
    new_df.dropna(inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    return new_df


def missing_value(df: pd.DataFrame):
    #  propagate last valid observation forward to next valid.
    df.fillna(method="ffill", inplace=True)
    #  If any NA values are present, drop that row.
    df.dropna(how="any", inplace=True, ignore_index=True)

    return df_result

filepath = "/Users/thl/Downloads/"
round_num = 3
names = ["luca", "nicole", "sam"]
df_cleaned = pd.DataFrame()
for round in range(1, round_num + 1):
    df_result = pd.DataFrame()
    for name in names:
        # concatenate, fill missing value, and drop nan
        filenames = glob(rf"{filepath}*-round{round}-{name}")
        if len(filenames) == 1:
            print(f"round{round}-{name}")
            df_map = load_dataset_map(filenames[0])
            df_result = concat_sensor_dataset(df_map)
            # missing value
            df_result = missing_value(df_result)
            # labeling
            df_result = labeling(df_result, df_map["time"], label_map[f"round{round}-{name}"], label_categories)
            df_result.drop(df_result[df_result["Error"] == 1].index, inplace = True)
            df_result.drop(columns="Error", inplace=True)
            # adjust time stamp
            df_result = adjust_granularity_timestamp_timediff(
                df_result, "Time (s)", 10, df_map["time"]
            )
    df_cleaned = pd.concat([df_cleaned, df_result], ignore_index=True)
df_cleaned.to_csv("../dataset/data_cleaned.csv", index=False)
