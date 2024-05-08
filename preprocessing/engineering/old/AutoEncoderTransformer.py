# %%
from sklearn.base import BaseEstimator, TransformerMixin


import pandas as pd
import numpy as np
import re





from sklearn.preprocessing import StandardScaler

import torch
from pyod.models.auto_encoder_torch import AutoEncoder
# import warnings


# from sklearn import set_config
# set_config(transform_output="pandas")

class AutoEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Klasse, welche jegliche Ausgaben der StringTransformatoren entgegennimmt
    und einen AutoEncoder anwendet, welcher den Rekonstruktionsfehler ausgibt.
    Diese jedoch für jede Merkmale der Original-Spalte einzeln.
    
    Schritte:
        (1) Durchführen von transform_columns. Hier findet die Steuerung der Klasse statt.
        (2) Filtern der maximalen Länge mit regex_len_original aller aggregierten Spalten, wie Länge der Strings, Großbuchstaben etc.
        (3) Iteration über jede Originalspalte. Es werden alle zugehörigen Spalten zur Originalspalte rausgefiltert.
        (4) Diese fließen in den AutoEncoder ein.
        (5) Für jede Originalspalte wird ein Rekonstruktionsfehler erzeugt mit den zugehörigen Merkmalen der Originalspalte.
    """
    def __init__(self):
        pass


    def autoencoder_pyod(self,X_filtered_columns,extract_original_col_names):
        """
        AutoEncoder aus der Pyod-Bibliothek.
        - Anomalie-Scores werden berechnet. Diese stellen den Rekonstruktionsfehler dar.
        """
        try:
            # Epochen werden auf 20 gesetzt.
            clf = AutoEncoder(epochs=5)
            # X_filtered_columns_as_numpy = X_filtered_columns.to_numpy()
            X_filtered_columns_as_tensor = torch.tensor(X_filtered_columns.values, dtype=torch.float)

            nan_count = torch.sum(torch.isnan(X_filtered_columns_as_tensor)).item()
            print(f"Anzahl der NaN-Einträge: {nan_count}")

            clf_result = clf.fit(X_filtered_columns_as_tensor)

            # # Nur auf die extrahierten Spalten der Originalspalte anwenden
            # clf_result = clf.fit(X_filtered_columns.to_numpy())

            X_temp_arr_decision_score = clf_result.decision_scores_

            suffix = f"{extract_original_col_names}_AE"

            X_AE_DF = pd.DataFrame(data=X_temp_arr_decision_score, columns=[suffix])

            return X_AE_DF
        except Exception as e:
            print(f"AutoEncoder pyod: {e}")

    def ae_on_columns(self,X_filtered_columns):
        """
        Es werden alle Merkmalsspalten zur zugehörigen Originalsaplte herausgefiltert.
        """
        X_col_to_list = list(X_filtered_columns.columns)
        # print(X_col_to_list)

        extract_original_col_names = re.search(r'.*__(.*?)_LEN_', X_col_to_list[0]).group(1)

        return self.autoencoder_pyod(X_filtered_columns=X_filtered_columns, extract_original_col_names=extract_original_col_names)



    def regex_len_original(self,X):
        """
        Hier wird der Suffix verwendet des vorherigen Prozesses.
        Die höchste Zahl gibt die Maximalanzahl der zugehörigen Merkmalsspalten zur Originalspalte an.
        Beispiel: 4*Vorverarbeitungsspalten pro Originalspalte. (Länge String, Anzahl Zeichen,etc.)
        """
        # Regex, um Ziffern am Ende zu extrahieren
        pattern = re.compile(r'_\d+$')
        # Liste, um die extrahierten Ziffern zu speichern
        numbers_at_end = []

        for col in X.columns:
            match = pattern.search(col)
            if match:
                # Extrahiere nur die Ziffern (ohne den Unterstrich)
                n_at_end = match.group(0)[1:]
                numbers_at_end.append(n_at_end)

        
        max_n = int(max(numbers_at_end))
        return max_n

    def transform_columns(self,X):
        X_length_original = self.regex_len_original(X)

        X_AE_Transformed = pd.DataFrame()

        # Anzahl der Spalten + 1 (Spalten beginnen mit dem suffix 0)
        for i in range(X_length_original + 1):
            try:
                # Extrahieren der Unterspalten der Originalspalte
                X_filtered_columns = X.filter(regex=f'_LEN_.*_{i}$')

                # Jetzt PCA durchführen und die Spalten zurückgeben
                ae_result = self.ae_on_columns(X_filtered_columns=X_filtered_columns)

                # Ergebnisse der PCA anfügen an X_pca
                X_AE_Transformed = pd.concat([X_AE_Transformed, ae_result], axis=1)

            except Exception as e:
                print(e)

        self.new_feature_names = X_AE_Transformed.columns
        return X_AE_Transformed


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.transform_columns(X)

    def get_feature_names(self, input_features=None):
        # Rückgabe der Featurenamen
        return self.new_feature_names



# Code zum testen:

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from StringAlphabetCreator import *
from StringLenCreator import *
from StringNumbersCharacterCreator import *
from StringSpecialCharacterCreator import *
from StringUniqueCharacterCreator import *
from CubicTransformer import *
from sklearn.impute import SimpleImputer


preprocessing = Pipeline(steps=[
    ("impute_columns", SimpleImputer(strategy="most_frequent")),

    ("Pattern calculation",
        ColumnTransformer(transformers=[

        ("Length_String",
        Pipeline(
                steps=[
                    ("len_s", StringLenCreator()),
                    ("z_transform_s_len", StandardScaler()),
                ]), make_column_selector(dtype_include=np.object_)),
        ("Length_String_Unique",
            Pipeline(
                steps=[
                    ("len_s_unique", StringUniqueCharacterCreator()),
                    ("z_transform_s_len_unique", StandardScaler()),
                ]), make_column_selector(dtype_include=np.object_)),
        ("Quantity_CapitalLetters",
            Pipeline(
                steps=[
                    ("alphabet_s_big", StringAlphabetCreatorBig()),
                    ("z_transform_s_big", StandardScaler()),
                ]), make_column_selector(dtype_include=np.object_)),
        ("Quantity_LowerCaseLetters",
            Pipeline(
                steps=[
                    ("alphabet_s_small", StringAlphabetCreatorSmall()),
                    ("z_transform_s_small", StandardScaler()),
                ]), make_column_selector(dtype_include=np.object_)),
        ("Quantity_SpecialCharacter",
            Pipeline(
                steps=[
                    ("special_s", StringSpecialCharacterCreator()),
                    ("z-transform_s_special", StandardScaler()),
                ]), make_column_selector(dtype_include=np.object_)),
        ("Quantity_Digits",
            Pipeline(
                steps=[
                    ("numbers_s", StringNumbersCharacterCreator()),
                    ("z_transform_s_number", StandardScaler()),
                ]),make_column_selector(dtype_include=np.object_)),
        ], remainder='passthrough', n_jobs=-1, verbose=True)
    ),
      ("AutoEncoder",
        ColumnTransformer(transformers=[
                    ("AETransformation",
            Pipeline(
                steps=[
                    ("ae_transformer", AutoEncoderTransformer())
                ]), make_column_selector(dtype_include=np.number)),
             ], remainder='passthrough', n_jobs=-1, verbose=True)
         
         ),

    ])


test_daten = pd.read_csv('./train_min_february_1_to_7.csv')

preprocessed_data = preprocessing.fit_transform(test_daten)
preprocessed_data

# # %%
