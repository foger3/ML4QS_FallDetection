import pandas as pd
from glob import glob
from sklearn.preprocessing import OneHotEncoder

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
    meta_names = ["Time"]
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


def label_validation(df: pd.DataFrame, labels):
    dataset_row_num = df[df["event"] == "START"].shape[0]
    print("{} movements in this dataset.".format(dataset_row_num))
    print("{} labels{}\n".format(len(labels), ", not match."if len(labels) != dataset_row_num else ", match."))


def labeling(df: pd.DataFrame, df_time: pd.DataFrame, labels: list, label_categories: list):
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
    enc = OneHotEncoder(categories = [label_categories], sparse_output=False)
    arr_label = enc.fit_transform(
        df[column_name].to_numpy().reshape(-1, 1)
    )
    df_label = pd.DataFrame(arr_label, columns = enc.get_feature_names_out([column_name])).astype("int")
    df_label.columns = label_categories
    df.drop(columns=column_name, inplace=True)
    df = pd.concat([df, df_label], axis = 1)

    return df

# labeling
label_categories = [
    "Label Walking",
    "Label Running",
    "Label Kneeing",
    "Label Lying",
    "Label Sitting",
    "Label Standing",
    "Label Falling",
    "Error",
]
label_map = dict()

# round1-luca
labels = [
    label_categories[0],
    label_categories[1],
    label_categories[4],
    label_categories[5],
    label_categories[6],
    label_categories[3],
    label_categories[2],
    label_categories[2],
    label_categories[4],
    label_categories[1],
    label_categories[0],
    label_categories[6],
    label_categories[3],
    label_categories[6],
    label_categories[4],
    label_categories[6],
    label_categories[7],
    label_categories[2],
    label_categories[6],
    label_categories[7],
]
label_map["round1-luca"] = labels
# round1-nicole
labels = [
    label_categories[0],
    label_categories[1],
    label_categories[4],
    label_categories[6],
    label_categories[2],
    label_categories[4],
    label_categories[6],
    label_categories[7],
    label_categories[7],
    label_categories[5],
    label_categories[4],
    label_categories[6]
]
label_map["round1-nicole"] = labels
# round1-sam
labels = [
    label_categories[0],
    label_categories[6],
    label_categories[4],
    label_categories[6],
    label_categories[1],
    label_categories[2],
    label_categories[3],
    label_categories[6],
    label_categories[5],
    label_categories[6],
    label_categories[2],
    label_categories[0],
    label_categories[6],
    label_categories[4]
]
label_map["round1-sam"] = labels
# round2-luca
labels = [
    label_categories[5],
    label_categories[0],
    label_categories[1],
    label_categories[6],
    label_categories[4],
    label_categories[3],
    label_categories[6],
    label_categories[2],
    label_categories[6],
    label_categories[5],
    label_categories[6],
    label_categories[4],
    label_categories[6],
    label_categories[2]
]
label_map["round2-luca"] = labels
# round2-nicole
labels = [
    label_categories[4],
    label_categories[2],
    label_categories[3],
    label_categories[0],
    label_categories[1],
    label_categories[6],
    label_categories[4],
    label_categories[4],
    label_categories[6],
    label_categories[2],
    label_categories[0],
    label_categories[7],
    label_categories[3],
]
label_map["round2-nicole"] = labels
# round3-nicole
labels = [
    label_categories[0],
    label_categories[4],
    label_categories[6],
    label_categories[6],
    label_categories[3],
    label_categories[6],
    label_categories[6],
    label_categories[2],
    label_categories[6],
    label_categories[1],
    label_categories[4],
    label_categories[7],
    label_categories[6],
    label_categories[0],
    label_categories[6],
    label_categories[6],
    label_categories[5],
    label_categories[3],
    label_categories[6],
    label_categories[2],
    label_categories[6],
]
label_map["round3-nicole"] = labels
# round3-sam
labels = [
    label_categories[6],
    label_categories[2],
    label_categories[6],
    label_categories[0],
    label_categories[6],
    label_categories[7],
    label_categories[3],
    label_categories[1],
    label_categories[6],
    label_categories[4],
    label_categories[6],
    label_categories[2],
    label_categories[5],
    label_categories[5],
    label_categories[3],
    label_categories[6],
    label_categories[0],
]
label_map["round3-sam"] = labels


filepath = "/Users/thl/Downloads/"
round_num = 3
names = ["luca", "nicole", "sam"]
df_cleaned = pd.DataFrame()
for round in range(1, round_num + 1):
    for name in names:
        # concatenate, fill missing value, and drop nan
        print(f"round{round}-{name}")
        filenames = glob(rf"{filepath}*-round{round}-{name}")
        if len(filenames) == 1:
            df_map = load_dataset_map(filenames[0])
            # label_validation(df_map["Time"], label_map[f"round{round}-{name}"])
            df_result = concat_sensor_dataset(df_map)
            df_result.fillna(method="ffill", inplace=True)
            df_result.dropna(how="any", inplace=True, ignore_index=True)
            # labeling
            df_result = labeling(df_result, df_map["Time"], label_map[f"round{round}-{name}"], label_categories)
            # delete error labeling data
            df_result.drop(df_result[df_result["Error"] == 1].index, inplace = True)
            df_result.drop(columns="Error", inplace=True)

    df_cleaned = pd.concat([df_cleaned, df_result], ignore_index=True)
df_cleaned.to_csv("../dataset/data_cleaned.csv", index = False)