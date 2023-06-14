import pandas as pd
from scipy.signal import butter, lfilter, filtfilt


def low_pass_filter(
    df: pd.DataFrame,
    columns: list[str],
    sampling_frequency: float,
    cutoff_frequency: float = 1.5,
    order: int = 10,
    phase_shift: bool = True,
):
    # http://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    # Cutoff frequencies are expressed as the fraction of the Nyquist frequency, which is half the sampling frequency
    nyq = 0.5 * sampling_frequency
    for col in columns:
        cut = cutoff_frequency / nyq
        b, a = butter(order, cut, btype="low", output="ba", analog=False)
        if phase_shift:
            df[col + "_lowpass"] = filtfilt(b, a, df[col])
        else:
            df[col + "_lowpass"] = lfilter(b, a, df[col])
            
    return df

