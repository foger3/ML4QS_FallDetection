##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

from scipy import special
from math import sqrt
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

# Class for outlier detection algorithms based on some distribution of the data. They
# all consider only single points per row (i.e. one column).


class DistributionBasedOutlierDetection:
    # Finds outliers in the specified column of datatable and adds a binary column with
    # the same name extended with '_outlier' that expresses the result per data point.
    def chauvenet(
        self, df: pd.DataFrame, columns: list[str], C: int = 2
    ) -> pd.DataFrame:
        # Taken partly from: https://www.astro.rug.nl/software/kapteyn/
        for col in columns:
            # Computer the mean and standard deviation.
            mean = df[col].mean()
            std = df[col].std()
            N = len(df.index)
            criterion = 1.0 / (C * N)

            # Consider the deviation for the data points.
            deviation = abs(df[col] - mean) / std

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
            df[f"{col}_outlier"] = mask
        return df

    # Fits a mixture model towards the data expressed in col and adds a column with the probability
    # of observing the value given the mixture model.
    def mixture_model(
        self, df: pd.DataFrame, columns: list[str], n_components: int = 3
    ) -> pd.DataFrame:
        # Fit a mixture model to our data.
        for col in columns:
            data = df[df[col].notnull()][col]
            g = GaussianMixture(n_components=n_components, max_iter=100, n_init=1)
            reshaped_data = data.values.reshape(-1, 1)
            g.fit(reshaped_data)

            # Predict the probabilities
            probs = g.score_samples(reshaped_data)

            # Create the right data frame and concatenate the two.
            data_probs = pd.DataFrame(
                np.power(10, probs), index=data.index, columns=[f"{col}_mixture"]
            )

            df = pd.concat([df, data_probs], axis=1)

        return df
