# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn import set_config

set_config(transform_output="pandas")


class MedianAbsolutDeviationTotal(BaseEstimator, TransformerMixin):
    """
    The MedianAbsolutDeviationTotal (MAD) class aggregates the values w.r.t. MedianAbsolutDeviation to get a single  aggregated column.
    """

    def __init__(self):
        self.z_names = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            X_copy = pd.DataFrame()

            # Erzeuge ein Total
            X_copy["MAD_Total"] = X.sum(axis=1)

            self.z_names = X_copy.columns

            return X_copy
        except Exception as e:
            print(f"Error: {e}")
            return X

    def get_feature_names(self, input_features=None):
        return self.z_names


# Funktioniert
# from sklearn.pipeline import Pipeline
# from sklearn.compose import make_column_transformer
# from MedianAbsolutDeviation import *


# numeric_pipeline = Pipeline(steps=[
#     ("z", MedianAbsolutDeviation(threshold=3.5, z_scores_output=False)),
#     # ("iterative_num", IterativeImputer()),
#     ("z_total", MedianAbsolutDeviationTotal())
#     # ("impute_num", SimpleImputer(strategy="median"))
# ])
# numeric_preprocessor = ColumnTransformer(
#     transformers=[
#         ('Preprocessing_Numerical', numeric_pipeline, make_column_selector(dtype_include=np.number))
#     ],
#     remainder='passthrough',
#     n_jobs=-1,
#     verbose=True
# )

# test_values_1 = np.array([1,2,3,4,5,6,1000])
# test_values_2 = np.array([1,2,3,4,50,700,1000])
# test_data = pd.DataFrame({"Example_column_1":test_values_1, "Example_column_2":test_values_2})

# preprocessed_data = pd.DataFrame(numeric_preprocessor.fit_transform(test_data), columns=numeric_preprocessor.get_feature_names_out())
# pd.concat([preprocessed_data, pd.DataFrame({"Example_column_1":test_values_1, "Example_column_2":test_values_2})],axis=1)
# %%
