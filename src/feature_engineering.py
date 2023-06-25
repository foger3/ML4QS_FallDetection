import copy
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from miscellaneous import logger


class FeatureAbstraction:
    def __init__(self, window_size: int, sampling_rate: float):
        self.ws = window_size
        self.fs = sampling_rate

        self.temp_list = []
        self.freqs = None

    # We create time points, assuming discrete time steps with fixed delta t:
    @staticmethod
    def get_slope(data: pd.Series):
        times = np.array(range(0, len(data.index)))
        data = data.astype(np.float32)

        # Check for NaN's
        mask = ~np.isnan(data)

        # If we have no data but NaN we return NaN.
        if len(data[mask]) == 0:
            return np.nan
        # Otherwise we return the slope.
        else:
            slope, _, _, _, _ = stats.linregress(times[mask], data[mask])

            return slope

    # This function aggregates a list of values using the specified aggregation
    # function (which can be 'mean', 'max', 'min', 'median', 'std', 'slope')
    def aggregate_value(self, data: pd.Series, aggregation_function: str):
        window_size = self.ws
        # Compute the values and return the result.
        if aggregation_function == "mean":
            return data.rolling(window_size, min_periods=window_size).mean()
        elif aggregation_function == "median":
            return data.rolling(window_size, min_periods=window_size).median()
        elif aggregation_function == "max":
            return data.rolling(window_size, min_periods=window_size).max()
        elif aggregation_function == "min":
            return data.rolling(window_size, min_periods=window_size).min()
        elif aggregation_function == "std":
            return data.rolling(window_size, min_periods=window_size).std()
        elif aggregation_function == "sem":
            return data.rolling(window_size, min_periods=window_size).sem()
        elif aggregation_function == "slope":
            return data.rolling(window_size, min_periods=window_size).apply(
                self.get_slope
            )
        else:
            return np.nan

    def abstract_numerical(
        self, df: pd.DataFrame, columns: list[str], aggregation_function_name: str
    ) -> pd.DataFrame:
        for col in columns:
            aggregations = self.aggregate_value(df[col], aggregation_function_name)
            df[
                col + "_temp_" + aggregation_function_name + "_ws_" + str(self.ws)
            ] = aggregations

        return df

    # Find the amplitudes of the different frequencies using a fast fourier transformation.
    # Here, the sampling rate expresses the number of samples per second (i.e. Frequency is Hertz of the dataset).
    def find_fft_transformation(self, data: pd.Series):
        # Create the transformation, this includes the amplitudes of both the real and imaginary part.
        transformation = np.fft.rfft(data, len(data))
        # real
        real_ampl = transformation.real
        # max
        max_freq = self.freqs[np.argmax(real_ampl[0 : len(real_ampl)])]
        # weigthed
        freq_weigthed = float(np.sum(self.freqs * real_ampl)) / np.sum(real_ampl)
        # pse
        PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
        PSD_pdf = np.divide(PSD, np.sum(PSD))

        # Make sure there are no zeros.
        if np.count_nonzero(PSD_pdf) == PSD_pdf.size:
            pse = -np.sum(np.log(PSD_pdf) * PSD_pdf)
        else:
            pse = 0

        real_ampl = np.insert(real_ampl, 0, max_freq)
        real_ampl = np.insert(real_ampl, 0, freq_weigthed)
        row = np.insert(real_ampl, 0, pse)

        self.temp_list.append(row)

        return 0

    # Get frequencies over a certain window.
    def abstract_frequency(self, df: pd.DataFrame, columns: list[str]):
        self.freqs = (self.fs * np.fft.rfftfreq(int(self.ws))).round(3)

        for col in columns:
            collist = []
            # prepare column names
            collist.append(col + "_max_freq")
            collist.append(col + "_freq_weighted")
            collist.append(col + "_pse")

            collist = collist + [
                col + "_freq_" + str(freq) + "_Hz_ws_" + str(self.ws)
                for freq in self.freqs
            ]

            # rolling statistics to calculate frequencies, per window size.
            df[col].rolling(self.ws + 1).apply(self.find_fft_transformation)

            # Pad the missing rows with nans
            frequencies = np.pad(
                np.array(self.temp_list),
                ((self.ws, 0), (0, 0)),
                "constant",
                constant_values=np.nan,
            )

            # add new freq columns to frame
            frequencies = pd.DataFrame(frequencies, index=df.index, columns=collist)
            df = pd.concat([df, frequencies], axis=1)

            # reset temp-storage array
            del self.temp_list[:]

        return df

    def abstract_features_with_pca(
        self, df: pd.DataFrame, label_columns: list[str], n_components: int = 20
    ) -> pd.DataFrame:

        # Interpolate missing values as PCA cannot handle these
        df_copy = copy.deepcopy(df).interpolate().ffill().bfill()

        # Drop columns not of interest for PCA
        df_copy = df_copy.drop(["ID"] + label_columns, axis=1)

        # Normalize data & fit PCA based on number of components
        X_normalized = (df_copy - df_copy.mean()) / df_copy.std()
        X_normalized.dropna(axis=1, inplace=True)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_normalized)

        # How much variance can we explain this way
        logger.info(
            f"{n_components} components capture {pca.explained_variance_ratio_.sum().round(3)} (%) of the feature's variance"
        )

        # Add PCA components to dataframe
        collist = [f"PCA_Component_{i}" for i in range(n_components)]
        X_pca = pd.DataFrame(X_pca, index=df.index, columns=collist)
        df = pd.concat([df, X_pca], axis=1)

        return df, collist
