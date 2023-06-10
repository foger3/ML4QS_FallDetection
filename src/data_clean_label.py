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
) -> pd.DataFrame:
    df_result = pd.DataFrame(columns=["Time (s)"])
    for sensor_name in sensor_names:
        df = df_map[str(sensor_name)]
        df_result = pd.merge(df_result, df, on="Time (s)", how="outer")
    df_result.sort_values(by=["Time (s)"], ignore_index=True, inplace=True)
    return df_result


def label_validation(df: pd.DataFrame, labels: list) -> None:
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
) -> pd.DataFrame:
    # label_validation(df_time, labels)
    # labeling
    for i in range(0, df_time.shape[0], 2):
        start_time = df_time["experiment time"][i]
        end_time = df_time["experiment time"][i + 1]
        target_idx = df[
            (df["Time (s)"] >= start_time) & (df["Time (s)"] <= end_time)
        ].index
        if "Label" not in df.columns:
            df["Label"] = np.nan
        df.iloc[target_idx, pd.Index(df.columns).get_loc("Label")] = labels[i // 2]
    # one hot encode
    df = one_hot_encoding(df, "Label", label_categories)

    return df


def one_hot_encoding(df, column_name, label_categories: list) -> pd.DataFrame:
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
    original_columns = df.columns
    sensor_columns = [
        "Magnetometer X (µT)",
        "Magnetometer Y (µT)",
        "Magnetometer Z (µT)",
        "Accelerometer X (m/s^2)",
        "Accelerometer Y (m/s^2)",
        "Accelerometer Z (m/s^2)",
        "Linear Accelerometer X (m/s^2)",
        "Linear Accelerometer Y (m/s^2)",
        "Linear Accelerometer Z (m/s^2)",
        "Gyroscope X (rad/s)",
        "Gyroscope Y (rad/s)",
        "Gyroscope Z (rad/s)",
        "Barometer X (hPa)",
    ]
    other_columns = list(set(df.columns) - set(sensor_columns))

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

    new_timestamp_name = "System time (s)"
    df[new_timestamp_name] = new_timestamp
    timestamps = pd.date_range(
        min(df[new_timestamp_name]),
        max(df[new_timestamp_name]),
        freq=f"{granularity}ms",
    )
    new_df = pd.DataFrame(index=timestamps, columns=original_columns, dtype=object)
    for i in range(len(new_df)):
        # Select the relevant measurements.
        relevant_rows = df[
            (df[new_timestamp_name] >= new_df.index[i])
            & (
                df[new_timestamp_name]
                < (new_df.index[i] + timedelta(milliseconds=granularity))
            )
        ]
        # new add
        if relevant_rows.shape[0] > 0:
            new_df.loc[[new_df.index[i]], sensor_columns] = np.average(
                relevant_rows[sensor_columns], axis=0
            )
            new_df.loc[[new_df.index[i]], other_columns] = np.asarray(
                relevant_rows[other_columns].iloc[0]
            )

    new_df.insert(0, new_timestamp_name, new_df.index)
    new_df.dropna(inplace=True)
    new_df.reset_index(drop=True, inplace=True)

    # Add Time difference
    new_df.insert(1, "Time difference (s)", new_df[new_timestamp_name])
    for i in range(0, time.shape[0], 2):
        start_time = time["experiment time"][i]
        end_time = time["experiment time"][i + 1]
        target_idx = new_df[
            (new_df["Time (s)"] >= start_time) & (new_df["Time (s)"] <= end_time)
        ].index
        if len(target_idx):
            new_df.loc[target_idx, ["Time difference (s)"]] = (
                new_df.loc[target_idx, "System time (s)"]
                - new_df.loc[target_idx, "System time (s)"].iloc[0]
            )
    new_df["Time difference (s)"] = (
        new_df["Time difference (s)"].astype("timedelta64[ms]").dt.total_seconds()
    )
    new_df.drop(columns="Time (s)", inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    return new_df


def fill_missing_value(df: pd.DataFrame) -> pd.DataFrame:
    timestamp_name = "Time (s)"
    columns = [col for col in df.columns if col != timestamp_name]
    data = df[columns].values
    timestamp = df[timestamp_name].values
    nan_indices = np.isnan(data)
    for i, _ in enumerate(columns):
        column_data = data[:, i]
        nan_index = nan_indices[:, i]
        valid_indices = np.where(~nan_index)[0]
        column_data[nan_index] = np.interp(
            np.where(nan_index)[0], valid_indices, column_data[valid_indices]
        )
        for j in range(len(valid_indices) - 1):
            head = valid_indices[j]
            tail = valid_indices[j + 1]
            time_diff_head = timestamp[head + 1 : tail] - timestamp[head]
            time_diff_tail = timestamp[tail] - timestamp[head + 1 : tail]
            mask = time_diff_head - time_diff_tail <= 0
            column_data[head + 1 : tail][mask] = column_data[head]
            column_data[head + 1 : tail][~mask] = column_data[tail]
    df[columns] = data
    df.drop_duplicates(subset=columns, inplace=True, ignore_index=True)
    return df[df[timestamp_name] >= 0.0]


def missing_value(df: pd.DataFrame) -> pd.DataFrame:
    #  propagate last valid observation forward to next valid.
    df.fillna(method="ffill", inplace=True)
    #  If any NA values are present, drop that row.
    df.dropna(how="any", inplace=True, ignore_index=True)

    return df_result


filepath = "../dataset/raw/"
round_num = 3
names = ["luca", "nicole", "sam"]
df_cleaned = pd.DataFrame()
for round in range(1, round_num + 1):
    for name in names:
        # concatenate, fill missing value, and drop nan
        filenames = glob(rf"{filepath}*-round{round}-{name}")
        if len(filenames) == 1:
            print(f"round{round}-{name}")
            df_map = load_dataset_map(filenames[0])
            df_result = concat_sensor_dataset(df_map)
            # missing value
            df_result = fill_missing_value(df_result)
            # labeling
            df_result = labeling(
                df_result,
                df_map["time"],
                label_map[f"round{round}-{name}"],
                label_categories,
            )
            df_result.drop(df_result[df_result["Error"] == 1].index, inplace=True)
            df_result.drop(columns="Error", inplace=True)
            df_cleaned.reset_index(level=0, drop=True)
            # adjust time stamp
            df_result = adjust_granularity_timestamp_timediff(
                df_result, "Time (s)", 10, df_map["time"]
            )
            df_cleaned = pd.concat([df_cleaned, df_result], ignore_index=True)

df_cleaned.to_csv("../dataset/data_cleaned.csv", index=False)
