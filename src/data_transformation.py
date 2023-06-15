import pandas as pd
from scipy.signal import butter, lfilter, filtfilt

class DataTransformation:

    def __init__(
        self, 
        df: pd.DataFrame, 
        columns: list[str]
    ):

        self.df = df.copy()
        self.columns = columns

    # Interpolate the dataset based on previous/next values..
    def impute_interpolate(
        self
    ) -> pd.DataFrame:
        
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

