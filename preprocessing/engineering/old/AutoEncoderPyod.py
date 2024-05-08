# %%
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin


import pandas as pd
import numpy as np
import re

from pyod.models.auto_encoder_torch import AutoEncoder


from sklearn import set_config
set_config(transform_output="pandas")

class AutoEncoderStandard(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def apply_AutoEncoder(self,X):
        clf = AutoEncoder(epochs=20)
        clf.fit(X)
        return clf.decision_scores_

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.apply_AutoEncoder(X)

    def get_feature_names(self, input_features=None):
        # Rückgabe der Featurenamen
        return self.new_feature_names
    


from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer



preprocessing = Pipeline(steps=[
        ('AutoEncoder Similarity',
            ColumnTransformer(transformers=[

            ("_similarity_autoencoder",
                Pipeline(
                    steps=[
                    ("C", SimpleImputer(strategy="most_frequent")),
                    # ("OHE", OneHotEncoder(handle_unknown="ignore",sparse=False)),
                    ("OrdinalEnc",OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                    ("AutoEncoder", AutoEncoderStandard())
                    ]), make_column_selector(dtype_include=np.object_)
            )
            ], remainder='passthrough', n_jobs=-1, verbose=True))
        ])


test_daten = pd.read_csv('./train_min_february_1_to_7.csv')

preprocessed_data = preprocessing.fit_transform(test_daten)
preprocessed_data
