from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np

from sklearn import set_config

set_config(transform_output="pandas")


class TimeTransformer(BaseEstimator, TransformerMixin):
    """
    The TimeTransformer class is used to extract different time units.

    Steps:\n
        (1) Iteration over each individual DateTime column.\n
        (2) Extraction of year/month/weekday/hour/minute.\n
        (3) Return of the transformed DataFrame.\n

    """

    def __init__(self):
        self.X_timeseries_transformed = pd.DataFrame()

    def extract_time_columns(self, X):
        for col in X.columns:
            col_name = col
            self.X_timeseries_transformed[col_name + "_YEAR"] = X[col].dt.year
            self.X_timeseries_transformed[col_name + "_MONTH"] = X[col].dt.month
            self.X_timeseries_transformed[col_name + "_WEEKDAY"] = X[col].dt.weekday
            self.X_timeseries_transformed[col_name + "_HOUR"] = X[col].dt.hour
            self.X_timeseries_transformed[col_name + "_MINUTE"] = X[col].dt.minute

        return self.X_timeseries_transformed

    def fit(self, X, y=None):
        # Kein fitting notwendig.
        return self

    def transform(self, X):
        return self.extract_time_columns(X)

    def get_feature_names(self, input_features=None):
        # Rückgabe der Featurenamen
        return self.X_timeseries_transformed.columns
