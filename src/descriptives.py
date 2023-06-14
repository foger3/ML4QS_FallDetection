import pandas as pd

def describe(
    df: pd.DataFrame
) -> pd.DataFrame:
    movement_num = df[df["Time difference (s)"] == 0].shape[0]
    print(f"The number of movement: {movement_num}")
    movement_start_idx = df[df["Time difference (s)"] == 0].index
    label_names = df.columns[df.columns.str.startswith('Label')]
    df_start = df.loc[movement_start_idx]
    movement_cnts = []
    for name in label_names:
        movement_cnt = df_start[df_start[name] == 1].shape[0]
        print("  - The number of {:<14}:{:>3}".format(name, movement_cnt))
        movement_cnts.append(movement_cnt)
    des_df = pd.DataFrame([movement_num] + movement_cnts, columns = ["count"], index = ["Total"] + list(label_names))
    
    return des_df