import numpy as np
import pandas as pd
import scipy.stats as stats

# Class to abstract a history of numerical values we can use as an attribute.
class TemporalNumericalAbstraction:

    # For the slope we need a bit more work.
    # We create time points, assuming discrete time steps with fixed delta t:
    def get_slope(
        self, 
        data: pd.Series
    ) -> float:
        
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
        window = str(window_size) + 's'
        # Compute the values and return the result.
        if aggregation_function == 'mean':
            return data.rolling(window, min_periods=window_size).mean()
        elif aggregation_function == 'median':
            return data.rolling(window, min_periods=window_size).median()
        elif aggregation_function == 'max':
            return data.rolling(window, min_periods=window_size).max()
        elif aggregation_function == 'min':
            return data.rolling(window, min_periods=window_size).min()
        elif aggregation_function == 'std':
            return data.rolling(window, min_periods=window_size).std()
        elif aggregation_function == 'sem':
            return data.rolling(window, min_periods=window_size).sem()
        elif aggregation_function == 'slope':
            return data.rolling(window, min_periods=window_size).apply(self.get_slope)
        
        #TODO: add your own aggregation function here
        else:
            return np.nan

    #TODO Add your own aggregation function here:
    # def my_aggregation_function(self, data)

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