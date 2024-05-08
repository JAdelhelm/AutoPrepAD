# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn import set_config

set_config(transform_output="pandas")


class ZTransformerMean(BaseEstimator, TransformerMixin):
    """
    The ZTransformerMean class marks data points which are above a certain threshold.

    Steps - Calculation of the Z-score:
        (1) Calculate the mean to obtain a central point of the data distribution.\n
        (2) Calculate the standard deviation to capture the data's spread.\n
        (3) Subtract the mean value from each value in x to obtain the distance.\n
        (4) Then, divide by the standard deviation to obtain a standardization.\n
        (5) If z_scores_output == True, the associated Z-scores are output.\n
        (6) If z_scores_output == False, values with an absolute value greater than a certain threshold, abs(z) > threshold, are marked with 1.\n

    Parameters
    ----------
    threshold : float
        Threshold of the Z-value that marks outliers above |z| > threshold.
    """

    def __init__(self, threshold=3, z_scores_output=False):
        self.threshold = threshold
        self.z_scores_output = z_scores_output

    def z_method(self, X):
        self.mean_X_ = np.mean(X)
        # n-1 for std (Assumption)
        self.stdev_X_ = np.std(X, ddof=1)

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.z_method(X)    
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if self.stdev_X_ == 0:
            return [0 for _ in X]

        if self.z_scores_output == False:
            self.z_scores_ = [(x - self.mean_X_) / self.stdev_X_ for x in X]
            return (np.abs(self.z_scores_) > self.threshold).astype(int)
        else:
            self.z_scores_ = [(x - self.mean_X_) / self.stdev_X_ for x in X]
            return self.z_scores_

    def get_feature_names_out(self, input_features=None):
        return [col+"_Z" for col in input_features]



# if __name__ == "__main__":
#     preprocessing = ColumnTransformer(
#         [
#             # ("cat", prep_cat, make_column_selector(dtype_include=np.object_)),
#             # ("num", prep_num, make_column_selector(dtype_include=np.number)),
#         ("z", ZTransformerMean(threshold=3, z_scores_output=False), make_column_selector(dtype_include=np.number))    ],
#         remainder='passthrough' # this will pass through any columns not specified in the transformers
#     )

#     train_data = pd.DataFrame({"Example_column": np.array([1,2,3,4,5])})
#     test_data = pd.DataFrame({"Example_column":[3,1000]})

#     preprocessing.fit(train_data)

#     preprocessed_data = preprocessing.transform(test_data)
#     print(preprocessed_data)


# %%
