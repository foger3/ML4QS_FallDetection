import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, lfilter, filtfilt
from plotly.subplots import make_subplots
from typing import List


class DataTransformation:
    def __init__(self, df: pd.DataFrame, columns: List[str]):
        self.df_original = df.copy()
        self.df = df.copy()
        self.columns = columns

    # Interpolate the dataset based on previous/next values..
    def impute_interpolate(self) -> pd.DataFrame:

        self.df[self.columns] = self.df[self.columns].interpolate().ffill().bfill()

        return self.df

    def low_pass_filter(
        self,
        sampling_frequency: float,
        cutoff_frequency: float = 1.5,
        order: int = 10,
        phase_shift: bool = True,
    ):
        # http://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
        # Cutoff frequencies are expressed as the fraction of the Nyquist frequency, which is half the sampling frequency

        nyq = 0.5 * sampling_frequency
        for col in self.columns:
            cut = cutoff_frequency / nyq
            b, a = butter(order, cut, btype="low", output="ba", analog=False)
            if phase_shift:
                self.df[col] = filtfilt(b, a, self.df[col])
            else:
                self.df[col] = lfilter(b, a, self.df[col])

        return self.df

    def visualize_low_pass(self, movements: List = [], result_filepath: str = "."):
        df_lowpass = self.df
        df = self.df_original
        movements = (
            movements if movements else df.columns[df.columns.str.startswith("Label")]
        )
        for movement in movements:
            all_movement_start_idx = list(
                df[(df["Time difference (s)"] == 0)].index
            ) + [df.shape[0]]
            movement_start_idx = list(
                df[(df["Time difference (s)"] == 0) & (df[movement] == 1)].index
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
                for i in movement_start_idx[:1]:
                    idx = all_movement_start_idx.index(i)
                    start_idx = all_movement_start_idx[idx]
                    end_idx = all_movement_start_idx[idx + 1] - 1
                    df_plot = df.loc[start_idx:end_idx]
                    df_plot_lowpass = df_lowpass.loc[start_idx:end_idx]
                    legend_flag = sensor_cnt == 4
                    fig.add_trace(
                        go.Scatter(
                            x=df_plot["Time difference (s)"],
                            y=df_plot[sensor_name],
                            mode="lines",
                            marker=dict(color="blue"),
                            name=f"Original",
                            showlegend=legend_flag,
                            legendgroup="normal",
                        ),
                        row=sensor_cnt // 3 + 1,
                        col=sensor_cnt % 3 + 1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df_plot_lowpass["Time difference (s)"],
                            y=df_plot_lowpass[sensor_name],
                            mode="lines",
                            marker=dict(size=3, color="red"),
                            name=f"Low-pass",
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
                f"{result_filepath}/{movement.split(' ')[1].lower()}_sensor_lowpass.html"
            )
