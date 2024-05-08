# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn import set_config

set_config(transform_output="pandas")


class MedianAbsolutDeviation(BaseEstimator, TransformerMixin):
    """
    The Median Absolute Deviation (MAD) class utilizes the median as a measure of distance and is therefore more robust to outliers compared to the Z-score.
    Other terms: Modified Z-Score

    Steps for calculating the Median Absolute Deviation:\n
        (1) Calculate the median, representing the central value.\n
        (2) Calculate the median of the deviation of each data point from the median (average distance of each data point from the median).\n
        (3) Scale this average median by determining the distance of each data point from the median.\n
            (3.1) Normalize it by 0.6745 to obtain a comparable measure to the Z-score.\n
        (4) Identify values that exceed a certain threshold.\n

    Parameters
    ----------
    threshold : float
        Threshold of the MAD at which data points should be marked as outliers.

    mad_scores_output : bool
        If True, it returns the score instead of a boolean integer value (0/1).

    """

    def __init__(self, threshold=3.5, mad_scores_output=False):
        self.threshold = threshold
        self.mad_scores_output = mad_scores_output

    def MAD(self, X):
        self.median_X_ = np.median(X)
        self.mad_X_ = np.median([np.abs(x - self.median_X_) for x in X])

        if self.mad_X_ == 0:
            return [0 for _ in X]

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.MAD(X)    
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        self.z_scores_median_ = [(0.6745 * ((x - self.median_X_) / self.mad_X_)) for x in X]

        if self.mad_scores_output == False:
            return (np.abs(np.array(self.z_scores_median_ )) > self.threshold).astype(int)
        else:
            return self.z_scores_median_ 


    def get_feature_names_out(self, input_features=None):
        return [col+"_Z_MOD" for col in input_features]



# if __name__ == "__main__":
#     preprocessing = ColumnTransformer(
#         [
#             # ("cat", prep_cat, make_column_selector(dtype_include=np.object_)),
#             # ("num", prep_num, make_column_selector(dtype_include=np.number)),
#         ("z", MedianAbsolutDeviation(threshold=3.5, mad_scores_output=True), make_column_selector(dtype_include=np.number))    ],
#         remainder='passthrough' # this will pass through any columns not specified in the transformers
#     )

#     train_data = pd.DataFrame({"Example_column": np.array([1,2,3,4,5])})
#     test_data = pd.DataFrame({"Example_column":[3,1000]})

#     preprocessing.fit(train_data)

#     preprocessed_data = preprocessing.transform(test_data)
#     print(preprocessed_data)



# %%
