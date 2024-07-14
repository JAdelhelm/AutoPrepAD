# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn import set_config

set_config(transform_output="pandas")


class TukeyTransformer(BaseEstimator, TransformerMixin):
    """
    The TukeyTransformer class marks data points which are above a certain threshold.

    Steps - Tukey Method:
        (1) Calculate the first and third quartiles (points in a sorted ascending data set at 25% and 75%).\n
        (2) Calculate the interquartile range: Q3 - Q1.\n
        (3) Multiply these boundary ranges (IQR) by a factor (here, 1.5).\n
        (4) Data points that fall outside these boundaries (whiskers in a box plot) are marked with 1.\n
        (5) Return the (marked) data points and the index of outliers.\n

    Parameters
    ----------
    factor : float
        Threshold of the Tukey Method at which data points should be marked as outliers.\n
        This threshold is based on the IQR.

    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def tukey_method(self, X):
        self.q1_ = np.quantile(X, 0.25)
        self.q3_ = np.quantile(X, 0.75)
        self.iqr_ = self.q3_ - self.q1_

        self.lower_ = self.q1_ - (self.factor * self.iqr_)
        self.upper_ = self.q3_ + (self.factor * self.iqr_)

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.tukey_method(X)    

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return ((X < self.lower_) | (X > self.upper_)).astype(int)

    def get_feature_names_out(self, input_features=None):
        return [col+"_TUKEY" for col in input_features]


# if __name__ == "__main__":
#     preprocessing = ColumnTransformer(
#         [
#             # ("cat", prep_cat, make_column_selector(dtype_include=np.object_)),
#             # ("num", prep_num, make_column_selector(dtype_include=np.number)),
#             ("tukey", TukeyTransformer(factor=1.5), make_column_selector(dtype_include=None))
#         ],
#         remainder='passthrough' # this will pass through any columns not specified in the transformers
#     )

#     train_data = pd.DataFrame({"Example_column": np.array([1,2,3,4,5])})
#     test_data = pd.DataFrame({"Example_column":[3,1000]})

#     preprocessing.fit(train_data)

#     preprocessed_data = preprocessing.transform(test_data)
#     print(preprocessed_data)


# %%
