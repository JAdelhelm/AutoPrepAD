# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import re

from sklearn_extra.cluster import KMedoids
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn import set_config
set_config(transform_output="pandas")

class KMedianEstimator(BaseEstimator, TransformerMixin):
    """
    Klasse, welche das K-Median Clustering nutzt, um die Distanzen
    zu den einzelnen Punkten zu bestimmen.
    Resultat ist ein Ähnlichkeitsmaß, welches detektiert, ob Abweichungen in den Mustern kategorischer Daten existiert.

    Ändern der method zu "alternate" für schnellere Verarbeitung. "pam" ist dafür genauer


    ---> Noch überlegen, wie man mit dem Parameter der Anzahl an Clustern umgeht.
    """
    def __init__(self, n_clusters=3, random_state=None, metric='euclidean', method='pam', max_iter=300):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.metric = metric
        self.method = method
        self.max_iter = max_iter


    def fit(self, X, y=None):
        """
        Hier werden die entsprechenden Spalten ausgewählt, welche gefittet werden.

        Diese haben folgende Form: X.columns = [column_name+"_LEN_ALPHABET_BIG" for column_name in X.columns]
        - Spaltennamen + Merkmalsbezeichnung

        Es soll für jede Spalte mit entsprechenden Merkmalen ein einzelnes Ähnlichkeitsmaß erzeugt werden.
        Also bspw. 5(Spalten) für ein Merkmal -> Großbuchstaben, Kleinbuchstaben, Sonderzeichen, Länge, Zahlen im String --> führen zu einer Spalte,
        welche ein Ähnlichkeitsmaß repräsentiert.

        Dieses Ähnlichkeitsmaß wird berechnet, indem der Abstand der einzelnen Werte zu den Centroiden zu jedem Punkt genommen wird.
        """
        self.kmedoids_ = KMedoids(self.n_clusters, self.metric, self.method, max_iter=self.max_iter, random_state=self.random_state)
        self.kmedoids_.fit(X=X)
        return self

    def transform(self, X):
        check_is_fitted(self)
        """
        Hier werden die Distanzen zu den Medoiden berechnet.
        """

        return X

    def get_feature_names_out(self, input_features=None):
        # Rückgabe der Featurenamen
        return [f"ClusterSimilarity_{col}" for col in input_features]







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
