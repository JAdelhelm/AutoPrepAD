from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np
from collections import Counter

from sklearn import set_config
set_config(transform_output="pandas")

class StringUniqueCharacterCreator(BaseEstimator, TransformerMixin):
    """
    Klasse, welche die Anzahl verschiedener Zeichen in der Zeichenkette zählt.
    """
    def __init__(self):
        self.column_names = []

    def len_strings_unique(self, X):
        return X.apply(lambda x: len(Counter(str(x))))
    

    def fit(self, X, y=None):
        # Gibt für jede einzelne Spalte noch zusätzlich ein Integer-Wert mit, damit zwichen ihnen differenziert werden kann
        for i in range(len(X.columns)):
            self.column_names.append(X.columns[i]+"_LEN_UNIQUE_STR_"+str(i))

        X.columns = self.column_names
        return self

    def transform(self, X):
        return X.apply(self.len_strings_unique)

    def get_feature_names_out(self, input_features=None):
        # Rückgabe der Featuerenamen
        return self.column_names
        # return [col for col in input_features]
