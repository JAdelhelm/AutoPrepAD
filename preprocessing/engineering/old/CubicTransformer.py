# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import re

from sklearn import set_config
set_config(transform_output="pandas")

class CubicTransformer(BaseEstimator, TransformerMixin):
    """
    Klasse, welche dazu dient, die numerischen Werte zu kubieren.
    Vorbereitung für das K-Median Clustering, welches die Distanzen zu den Centroiden bestimmen soll.
    """
    def __init__(self):
        pass

    def num_cubic(self,X):
        return X.apply(lambda x: x**3)


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.num_cubic)

    def get_feature_names(self, input_features=None):
        # Rückgabe der Featurenamen
        return [col for col in input_features]







# test_data = pd.DataFrame({"Example_num_column":np.array([1,2,3,4,5,100000, np.nan])})

# preprocessing =  Pipeline(steps=[
#     ("StandardScaling",
#         ColumnTransformer(transformers=[
#             ("StandardScaler", StandardScaler(), make_column_selector(dtype_include=np.number))
#         ], remainder='passthrough', n_jobs=-1, verbose=True)

#     ),

#     ("CubicTransformation",
#         ColumnTransformer(transformers=[
#             ("CubicTransformer", CubicTransformer(), make_column_selector(dtype_include=np.number))
#         ], remainder='passthrough', n_jobs=-1, verbose=True)

#         )

#     ])


# preprocessed_data = pd.DataFrame(preprocessing.fit_transform(test_data), columns=preprocessing.get_feature_names_out())
# preprocessed_data
# %%
