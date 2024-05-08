# %%

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
import pandas as pd
import numpy as np

from sklearn import set_config

set_config(transform_output="pandas")


class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    """
    Imputer for Timeseries datatypes.

    Steps:
        (1) A copy of the DataFrame is created.\n
        (2) Missing values in the time series data are filled using the selected method.\n
        (3) Return the filled DataFrame.\n

    Parameters
    ----------
    impute_method : str
        Impute method to fill NaN-Values, e.g. (ffill(bfill))

    """

    def __init__(self, impute_method="ffill"):
        self.impute_method = impute_method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.impute_method == "ffill" or self.impute_method == "bfill":
            X_copy.fillna(method=self.impute_method, inplace=True)

        self.col_names = X.columns

        return X_copy

    def get_feature_names(self, input_features=None):
        return self.col_names
