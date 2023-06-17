from math import sqrt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import special
from sklearn.mixture import GaussianMixture
from typing import List
from plotly.subplots import make_subplots

# Class for outlier detection algorithms based on some distribution of the data. They
# all consider only single points per row (i.e. one column).


class OutlierDetectionDistribution:
    def __init__(self, df: pd.DataFrame, columns: List[str]):

        self.df = df.copy()
        self.columns = columns

    # Finds outliers in the specified column of datatable and adds a binary column with
    # the same name extended with '_outlier' that expresses the result per data point.
    def chauvenet(self, C: int = 2) -> pd.DataFrame:
        # Taken partly from: https://www.astro.rug.nl/software/kapteyn/

        chauvenet_df = pd.DataFrame()
        for col in self.columns:
            # Computer the mean and standard deviation.
            mean = self.df[col].mean()
            std = self.df[col].std()
            N = len(self.df.index)
            criterion = 1.0 / (C * N)

            # Consider the deviation for the data points.
            deviation = abs(self.df[col] - mean) / std

            # Express the upper and lower bounds.
            low = -deviation / sqrt(C)
            high = deviation / sqrt(C)
            prob = []
            mask = []

            # Pass all rows in the dataset.
            for i in range(N):
                # Determine the probability of observing the point
                prob.append(1.0 - 0.5 * (special.erf(high[i]) - special.erf(low[i])))
                # And mark as an outlier when the probability is below our criterion.
                mask.append(prob[i] < criterion)
            chauvenet_df[f"{col}_outlier"] = mask

        return chauvenet_df

    # Fits a mixture model towards the data expressed in col and adds a column with the probability
    # of observing the value given the mixture model.
    def mixture_model(self, n_components: int = 3) -> pd.DataFrame:
        # Fit a mixture model to our data.

        mixture_df = pd.DataFrame()
        for col in self.columns:
            data = self.df[self.df[col].notnull()][col]
            g = GaussianMixture(n_components=n_components, max_iter=100, n_init=1)
            reshaped_data = data.values.reshape(-1, 1)
            g.fit(reshaped_data)

            # Predict the probabilities
            probs = g.score_samples(reshaped_data)

            # Create the right data frame and concatenate the two.
            data_probs = pd.DataFrame(
                np.power(10, probs), index=data.index, columns=[f"{col}_mixture"]
            )

            mixture_df = pd.concat([mixture_df, data_probs], axis=1)

        return mixture_df

    def chauvenet_visualize(
        self,
        df_chauvenet: pd.DataFrame,
        movements: List = [],
        result_filepath: str = ".",
    ):
        movements = (
            movements
            if movements
            else self.df.columns[self.df.columns.str.startswith("Label")]
        )
        for movement in movements:
            all_movement_start_idx = list(
                self.df[(self.df["Time difference (s)"] == 0)].index
            ) + [self.df.shape[0]]
            movement_start_idx = list(
                self.df[
                    (self.df["Time difference (s)"] == 0) & (self.df[movement] == 1)
                ].index
            )

            fig = make_subplots(
                rows=len(self.columns) // 3 + 1,
                cols=3,
                shared_yaxes=True,
                vertical_spacing=0.08,
                horizontal_spacing=0.02,
                subplot_titles=[i.split(" (")[0] for i in self.columns],
            )

            for sensor_cnt, sensor_name in enumerate(self.columns):
                df_normal = self.df[~df_chauvenet.iloc[:, sensor_cnt]]
                df_outlier = self.df[df_chauvenet.iloc[:, sensor_cnt]]
                for i in movement_start_idx[:1]:
                    idx = all_movement_start_idx.index(i)
                    start_idx = all_movement_start_idx[idx]
                    end_idx = all_movement_start_idx[idx + 1] - 1
                    df_plot = df_normal.loc[start_idx:end_idx]
                    df_plot_outlier = df_outlier.loc[start_idx:end_idx]
                    legend_flag = sensor_cnt == 4
                    fig.add_trace(
                        go.Scatter(
                            x=df_plot["Time difference (s)"],
                            y=df_plot[sensor_name],
                            mode="lines",
                            marker=dict(color="blue"),
                            name=f"{movement} normal",
                            showlegend=legend_flag,
                            legendgroup="normal",
                        ),
                        row=sensor_cnt // 3 + 1,
                        col=sensor_cnt % 3 + 1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df_plot_outlier["Time difference (s)"],
                            y=df_plot_outlier[sensor_name],
                            mode="markers",
                            marker=dict(size=3, color="red"),
                            name=f"{movement} outlier",
                            showlegend=legend_flag,
                            legendgroup="outlier",
                        ),
                        row=sensor_cnt // 3 + 1,
                        col=sensor_cnt % 3 + 1,
                    )
                    if sensor_cnt % 3 == 0:
                        fig.update_yaxes(
                            title_text=self.columns[sensor_cnt].split(" ")[-1][1:-1],
                            row=sensor_cnt // 3 + 1,
                            col=sensor_cnt % 3 + 1,
                        )
                    if sensor_cnt > 9:
                        fig.update_xaxes(
                            title_text="Time (s)",
                            row=sensor_cnt // 3 + 1,
                            col=sensor_cnt % 3 + 1,
                        )
            fig.update_layout(
                width=800,
                height=700,
                showlegend=True,
                font=dict(size=12),
                title_text=f"{movement.split(' ')[1]}",
            )
            fig.update_annotations(font_size=12)
            fig.write_html(
                f"{result_filepath}/{movement.split(' ')[1].lower()}_sensor_chauvenet_outlier.html"
            )
