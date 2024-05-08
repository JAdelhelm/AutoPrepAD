# %%
from sklearn.base import BaseEstimator, TransformerMixin


import pandas as pd
import numpy as np
import re



# from sklearn.compose import ColumnTransformer
# from sklearn.compose import make_column_selector
# from sklearn.pipeline import Pipeline
# from StringAlphabetCreator import *
# from StringLenCreator import *
# from StringNumbersCharacterCreator import *
# from StringSpecialCharacterCreator import *
# from StringUniqueCharacterCreator import *
# from CubicTransformer import *

# from sklearn.preprocessing import StandardScaler


from sklearn import set_config
set_config(transform_output="pandas")

class PCATransformer(BaseEstimator, TransformerMixin):
    """
    Klasse, welche jegliche Ausgaben der StringTransformatoren entgegennimmt
    und eine PCA auf diese anwendet.
    Diese jedoch für jede Spalte einzeln.
    """
    def __init__(self, activate_pyod_pca = False, original_column_names=None):
        self.new_feature_names = None
        self.activate_pyod_pca = activate_pyod_pca
        self.original_column_names = original_column_names


    def pca_normal(self,X_filtered_columns,extract_original_col_names):
        """
        Normale PCA aus der Sklearn Bibliothek.
        Wenn eine Varianz von über 70% erreicht wurde, dann wird keine Hauptkomponente mehr
        hinzugefügt.
        """
        from sklearn.decomposition import PCA
        # Initiern mit einer Komponente
        init_components = 1
        clf = PCA(n_components=init_components)

        # Nur auf die extrahierten Spalten der Originalspalte anwenden
        X_pca_columns_transformed = clf.fit_transform(X_filtered_columns)

        # Der Suffix gibt die Varianz der Komponente an. Ursprüngliche Spaltennamen werden überschriben mit col_names_components
        variance_explained_component = np.array(np.round(clf.explained_variance_ratio_[init_components-1] * 100),dtype=np.int16)
        suffix = f"{extract_original_col_names}_component_{init_components}_vari{variance_explained_component}"
        col_names_components = []
        col_names_components.append(suffix)

        # Falls die Varianz unter 70%, dann Komponenten hinzufügen
        # while np.sum(clf.explained_variance_ratio_) <= 0.7:
        # Probleme, wenn die Varianz von Training- und Test !=
        while init_components < 3:
            init_components += 1
            clf = PCA(n_components=init_components)

            X_pca_columns_transformed = clf.fit_transform(X_filtered_columns)

            variance_explained_component = np.array(np.round(clf.explained_variance_ratio_[init_components-1] * 100),dtype=np.int16)
            suffix = f"{extract_original_col_names}_component_{init_components}_vari{variance_explained_component}"
            col_names_components.append(suffix)

        # Umwandeln in einen DataFrame und übergeben der neuen Spaltenbezeichnungen
        X_pca_columns_transformed.columns =  col_names_components


        return X_pca_columns_transformed

    def pca_pyod(self,X_filtered_columns,extract_original_col_names):
        from pyod.models.pca import PCA
        """
        PCA aus der Pyod-Bibliothek.
        - Anomalie-Scores werden berechnet basierend auf der gewichteten Distanz
          zwischen den Datenpunkten und den ausgewählten Hauptkomponenten.
        """

        # Initiern mit einer Komponente
        init_components = 1
        clf = PCA(n_components=init_components)
        # Nur auf die extrahierten Spalten der Originalspalte anwenden
        clf_result = clf.fit(X_filtered_columns)

        # Der Suffix gibt die Varianz der Komponente an.
        # Hier werden Pandas Dataframe direkt gemergt, da die PCA von Pyod ein numpy-array zurück gibt in Form des decision_scores
        variance_explained_component = np.array(np.round(clf.explained_variance_ratio_[init_components-1] * 100),dtype=np.int16)
        suffix = f"{extract_original_col_names}_component_{init_components}_vari{variance_explained_component}"
        X_temp_arr_decision_score = clf_result.decision_scores_

        X_PCA_DF = pd.DataFrame(data=X_temp_arr_decision_score, columns=[suffix])

        # while np.sum(clf.explained_variance_ratio_) <= 0.7:
        # Probleme, wenn die Varianz von Training- und Test !=
        while init_components < 3:
            init_components += 1
            clf = PCA(n_components=init_components)
            clf_result = clf.fit(X_filtered_columns)

            variance_explained_component = np.array(np.round(clf.explained_variance_ratio_[init_components-1] * 100),dtype=np.int16)
            suffix = f"{extract_original_col_names}_component_{init_components}_vari{variance_explained_component}"
            X_temp_arr_decision_score = clf_result.decision_scores_

            X_PCA_df_temp = pd.DataFrame(data=X_temp_arr_decision_score, columns=[suffix])

            X_PCA_DF = pd.concat([X_PCA_DF, X_PCA_df_temp], axis=1)


        return X_PCA_DF

    def pca_on_columns(self,X_filtered_columns):
        """
        Hier werden die Spalten transformiert von jeder ursprünglichen Spalte.
        Um ein Ähnlichkeitsmaß zu erhalten.
        Wenn die Summe der Hauptkomponenten > 0.7 ist, dann werden diese als Spalten zurückgegeben.
        """
        X_col_to_list = list(X_filtered_columns.columns)
        # print(X_col_to_list)

        extract_original_col_names = re.search(r'.*__(.*?)_LEN_', X_col_to_list[0]).group(1)


        if self.activate_pyod_pca == False:
            return self.pca_normal(X_filtered_columns=X_filtered_columns, extract_original_col_names=extract_original_col_names)
        else:
            return self.pca_pyod(X_filtered_columns=X_filtered_columns, extract_original_col_names=extract_original_col_names)


    def regex_len_original(self,X):
        """
        Hier wird der Suffix verwendet des vorherigen Prozesses.
        Die höchste Zahl gibt die Maximalanzahl der Spalten an.
        Also bspw. 4*Vorverarbeitungsspalten pro Originalspalte. (Länge String, Anzahl Zeichen,etc.)
        """
        # Regex, um Ziffern am Ende zu finden
        pattern = re.compile(r'_\d+$')
        # Liste, um die extrahierten Ziffern zu speichern
        numbers_at_end = []

        for col in X.columns:
            match = pattern.search(col)
            if match:
                # Extrahiere nur die Ziffern (ohne den Unterstrich)
                number_at_end = match.group(0)[1:]
                numbers_at_end.append(number_at_end)

        # +1, da mit 0 angefangen wird zu zählen
        max_n = int(max(number_at_end)) + 1
        return max_n

    def transform_columns(self,X):
        """
        Auswahl der Feature-Spalten für jede Originalspalte.
        Dazu wird erstmal ein Tensor(Array) erzeugt, welcher alle Spalten
        für die jeweilige Originalspalte enthält.
        Der Suffix ist _LEN_....._0 , wobei 0 die entsprechende Original-Spalte zuordnet.

        """
        # Ich benötige die Anzahl der ursprünglichen Spalten
        # Hierzu nehme ich die höchste Zahl des Suffix der Spalten + 1, da bei 0 beginnend
        X_length_original = self.regex_len_original(X)

        X_PCA_Transformed = pd.DataFrame()
        for i in range(X_length_original):
            try:
                # Extrahieren der Unterspalten der Originalspalte
                X_filtered_columns = X.filter(regex=f'_LEN_.*_{i}$')

                # Jetzt PCA durchführen und die Spalten zurückgeben
                pca_result = self.pca_on_columns(X_filtered_columns=X_filtered_columns)

                # Ergebnisse der PCA anfügen an X_pca
                X_PCA_Transformed = pd.concat([X_PCA_Transformed, pca_result], axis=1)

            except Exception as e:
                print("Column not found")
                print(e)

        self.new_feature_names = X_PCA_Transformed.columns
        return X_PCA_Transformed


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.transform_columns(X)

    def get_feature_names(self, input_features=None):
        # Rückgabe der Featurenamen
        return self.new_feature_names



# Code zum testen:

# preprocessing = Pipeline(steps=[
#     ("Pattern calculation",
#         ColumnTransformer(transformers=[

#         ("Length_String",
#         Pipeline(
#                 steps=[
#                     ("len_s", StringLenCreator()),
#                     ("z_transform_s_len", StandardScaler()),
#                     # ("cubic_transform_s_len", CubicTransformer())
#                 ]), make_column_selector(dtype_include=np.object_)),
#         ("Length_String_Unique",
#             Pipeline(
#                 steps=[
#                     ("len_s_unique", StringUniqueCharacterCreator()),
#                     ("z_transform_s_len_unique", StandardScaler()),
#                     # ("cubic_transform_s_len_unique", CubicTransformer())
#                 ]), make_column_selector(dtype_include=np.object_)),
#         ("Quantity_CapitalLetters",
#             Pipeline(
#                 steps=[
#                     ("alphabet_s_big", StringAlphabetCreatorBig()),
#                     ("z_transform_s_big", StandardScaler()),
#                     # ("cubic_transform_s_big", CubicTransformer())
#                 ]), make_column_selector(dtype_include=np.object_)),
#         ("Quantity_LowerCaseLetters",
#             Pipeline(
#                 steps=[
#                     ("alphabet_s_small", StringAlphabetCreatorSmall()),
#                     ("z_transform_s_small", StandardScaler()),
#                     # ("cubic_transform_s_small", CubicTransformer())
#                 ]), make_column_selector(dtype_include=np.object_)),
#         ("Quantity_SpecialCharacter",
#             Pipeline(
#                 steps=[
#                     ("special_s", StringSpecialCharacterCreator()),
#                     ("z-transform_s_special", StandardScaler()),
#                     # ("cubic_transform_s_special", CubicTransformer())
#                 ]), make_column_selector(dtype_include=np.object_)),
#         ("Quantity_Digits",
#             Pipeline(
#                 steps=[
#                     ("numbers_s", StringNumbersCharacterCreator()),
#                     ("z_transform_s_number", StandardScaler()),
#                     # ("cubic_transform_s_number", CubicTransformer())
#                 ]),make_column_selector(dtype_include=np.object_)),
#         ], remainder='passthrough', n_jobs=-1, verbose=True)
#     ),
#     ("PCA",
#         ColumnTransformer(transformers=[
#                     ("PCATransformation",
#             Pipeline(
#                 steps=[
#                     ("pca_transformer", PCATransformer(activate_pyod_pca=True))
#                 ]), make_column_selector(dtype_include=None)),
#              ], remainder='passthrough', n_jobs=-1, verbose=True)
#     ),

#     ])


# test_daten = pd.read_csv('./train_february_1_to_7.csv')
# test_daten = test_daten.sample(frac=0.1,ignore_index=True, random_state=42)

# preprocessed_data = preprocessing.fit_transform(test_daten)
# preprocessed_data
