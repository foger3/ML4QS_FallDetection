from glob import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from miscellaneous import label_map, label_categories


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
    df: pd.DataFrame, 
    df_time: pd.DataFrame, 
    labels: list, 
    label_categories: list
) -> pd.DataFrame:
    # label_validation(df_time, labels)
    # labeling
    df.reset_index(level=0, drop=True, inplace = True)
    for i in range(0, df_time.shape[0], 2):
        start_time = df_time["experiment time"][i]
        end_time = df_time["experiment time"][i + 1]
        target_idx = df[
            (df["Time (s)"] >= start_time) & (df["Time (s)"] <= end_time)
        ].index
        if "Label" not in df.columns:
            df["Label"] = np.nan
        df.loc[target_idx, "Label"] = labels[i // 2]
    # one hot encode
    df = one_hot_encoding(df, "Label", label_categories)

    return df


def one_hot_encoding(
    df: pd.DataFrame, 
    column_name, 
    label_categories: list
) -> pd.DataFrame:
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
    df: pd.DataFrame, 
    granularity: float, 
    df_time: pd.DataFrame
) -> pd.DataFrame:
    df.insert(1, "System time (s)", np.nan)
    df.insert(2, "Time difference (s)", np.nan)
    original_columns = df.columns
    sensor_names = ["Magnetometer", "Accelerometer", "Linear Accelerometer", "Gyroscope", "Barometer"]
    sensor_columns = df.columns[df.columns.str.startswith(tuple(sensor_names))].values

    # dataframe to ndarray (for speedup)
    arr = np.array(df, dtype=object)
    arr_time_idx = df.columns.get_loc("Time (s)")
    arr_systime_idx = df.columns.get_loc("System time (s)")
    arr_sensor_idx = [df.columns.get_loc(col) for col in sensor_columns]
    arr_label_idx = [df.columns.get_loc(col) for col in df.columns[df.columns.str.startswith('Label')]]

    arr_new = np.empty([0,2 + len(arr_sensor_idx) + len(arr_label_idx)], dtype=object)
    for i in range(0, df_time.shape[0], 2):
        start_systime = pd.Timestamp(df_time["system time text"][i][:23])
        end_systime = pd.Timestamp(df_time["system time text"][i+1][:23])
        start, end = df_time["experiment time"][i:i+2]
        # add system time on target array
        arr_trg = arr[((arr[:,arr_time_idx] >= start) & (arr[:,arr_time_idx] <= end)), :]
        arr_trg[:, arr_systime_idx] = np.array(pd.to_timedelta(arr_trg[:,arr_time_idx] - start, unit="s")) + start_systime.to_datetime64()
        # new timestamp base on granularity
        timestamps = np.array(pd.date_range(start_systime, end_systime + pd.to_timedelta(granularity, unit="ms"), freq=f"{granularity}ms"))
        start_new = 0
        start_flag = False
        for ts in timestamps:
            relevant_rows = arr_trg[(arr_trg[:, arr_systime_idx].astype(datetime) >= ts) & (arr_trg[:, arr_systime_idx].astype(datetime) < (ts + np.timedelta64(granularity, 'ms'))),:]
            if relevant_rows.shape[0] > 0:
                start_flag = True
                sensor_avg = np.average(relevant_rows[:,arr_sensor_idx], axis=0)
                arr_tmp = np.concatenate((np.array([ts, start_new*granularity/1000.0]), sensor_avg, relevant_rows[:,arr_label_idx][0].astype("int")), dtype=object)
                arr_new = np.concatenate((arr_new, np.expand_dims(arr_tmp, axis=0)))
                # remove useless
                arr_trg = arr_trg[relevant_rows.shape[0]:,:]
            start_new += int(start_flag)
    df = pd.DataFrame(arr_new, columns = original_columns[1:])
    df["System time (s)"] = pd.to_datetime(df["System time (s)"])
    df.reset_index(drop=True, inplace=True)

    return df


def fill_missing_value(
    df: pd.DataFrame
) -> pd.DataFrame:
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
    df = df[df[timestamp_name] >= 0.0]
    df.reset_index(level=0, drop=True, inplace = True)
    
    return df


def assign_id_per_movement(
    df: pd.DataFrame, based: int
) -> pd.DataFrame:
    movement_start_idx = list(df[df["Time difference (s)"] == 0].index) + [df.shape[0]]
    df.insert(0, "ID", 0)
    for i in range(len(movement_start_idx) - 1):
        df.loc[movement_start_idx[i]:movement_start_idx[i+1], "ID"] = i

    return df


filepath = "/Users/thl/Downloads/"
round_num = 3
names = ["luca", "nicole", "sam"]
granularity = 10
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
            # adjust time stamp
            df_result = adjust_granularity_timestamp_timediff(
                df_result, granularity, df_map["time"]
            )
            df_cleaned.reset_index(level=0, drop=True, inplace = True)
            df_cleaned = pd.concat([df_cleaned, df_result], ignore_index=True)

# assign id
df_cleaned = assign_id_per_movement(df_cleaned)
df_cleaned.to_csv(f"../dataset/data_cleaned_{granularity}.csv", index=False)
