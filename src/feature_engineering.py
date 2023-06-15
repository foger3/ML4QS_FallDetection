import numpy as np
import pandas as pd
import scipy.stats as stats


# Class to abstract a history of numerical values we can use as an attribute.
class TemporalAbstraction:

    # For the slope we need a bit more work.
    # We create time points, assuming discrete time steps with fixed delta t:
    def get_slope(
        self, 
        data: pd.Series
    ):
        
        times = np.array(range(0, len(data.index)))
        data = data.astype(np.float32)

        # Check for NaN's
        mask = ~np.isnan(data)

        # If we have no data but NaN we return NaN.
        if (len(data[mask]) == 0):

            return np.nan
        # Otherwise we return the slope.
        else:
            slope, _, _, _, _ = stats.linregress(times[mask], data[mask])

            return slope

    # This function aggregates a list of values using the specified aggregation
    # function (which can be 'mean', 'max', 'min', 'median', 'std', 'slope')
    def aggregate_value(
        self, 
        data: pd.Series, 
        window_size: int, 
        aggregation_function: str
    ):
        # window = str(window_size) + 's'
        # Compute the values and return the result.
        if aggregation_function == 'mean':
            return data.rolling(window_size, min_periods=window_size).mean()
        elif aggregation_function == 'median':
            return data.rolling(window_size, min_periods=window_size).median()
        elif aggregation_function == 'max':
            return data.rolling(window_size, min_periods=window_size).max()
        elif aggregation_function == 'min':
            return data.rolling(window_size, min_periods=window_size).min()
        elif aggregation_function == 'std':
            return data.rolling(window_size, min_periods=window_size).std()
        elif aggregation_function == 'sem':
            return data.rolling(window_size, min_periods=window_size).sem()
        elif aggregation_function == 'slope':
            return data.rolling(window_size, min_periods=window_size).apply(self.get_slope)        
        else:
            return np.nan

    def abstract_numerical(
        self, 
        df: pd.DataFrame, 
        columns: list[str], 
        window_size: int, 
        aggregation_function_name: str
    ) -> pd.DataFrame:
    
        for col in columns: 
            aggregations = self.aggregate_value(df[col], window_size, aggregation_function_name)
            df[col + '_temp_' + aggregation_function_name + '_ws_' + str(window_size)] = aggregations
      
        return df


# This class performs a Fourier transformation to find frequencies that occur often and filter noise.
class FourierTransformation:
    
    def __init__(self):
        self.temp_list = []
        self.freqs = None

    # Find the amplitudes of the different frequencies using a fast fourier transformation. 
    # Here, the sampling rate expresses the number of samples per second (i.e. Frequency is Hertz of the dataset).
    
    def find_fft_transformation(
        self, 
        data: pd.Series
    ):
        # Create the transformation, this includes the amplitudes of both the real and imaginary part.
        # print(data.shape)
        transformation = np.fft.rfft(data, len(data))
        # real
        real_ampl = transformation.real
        # max
        max_freq = self.freqs[np.argmax(real_ampl[0:len(real_ampl)])]
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
    def abstract_frequency(
        self, 
        df: pd.DataFrame, 
        columns: list[str], 
        window_size: int, 
        sampling_rate: int
    ):
        self.freqs = (sampling_rate * np.fft.rfftfreq(int(window_size))).round(3)

        for col in columns:
            collist = []
            # prepare column names
            collist.append(col + '_max_freq')
            collist.append(col + '_freq_weighted')
            collist.append(col + '_pse')
            
            collist = collist + [col + '_freq_' +
                    str(freq) + '_Hz_ws_' + str(window_size) for freq in self.freqs]
           
            # rolling statistics to calculate frequencies, per window size. 
            # Pandas Rolling method can only return one aggregation value. 
            # Therefore values are not returned but stored in temp class variable 'temp_list'.

            # note to self! Rolling window_size would be nicer and more logical! In older version windowsize is actually 41. (ws + 1)
            df[col].rolling(window_size + 1).apply(self.find_fft_transformation)

            # Pad the missing rows with nans
            frequencies = np.pad(np.array(self.temp_list), ((window_size, 0), (0, 0)),
                                 'constant', constant_values=np.nan)
            # add new freq columns to frame
            
            df[collist] = pd.DataFrame(frequencies, index=df.index)

            # reset temp-storage array
            del self.temp_list[:]
            
        return df
    

