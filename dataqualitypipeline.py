from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn import set_config
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# from graphviz import Digraph
from category_encoders import BinaryEncoder

# import tensorflow as tf
import pyod
from sklearn.utils import estimator_html_repr

from visualization.explorativ import *

from pipelines import PipelinesConfiguration
from experiment import Experiment

# import pdfkit

# Activate if you use PyTorch algorithms
# import torch


import os
from joblib import dump
import itertools
from pathlib import Path

class DQPipeline(PipelinesConfiguration, Experiment):
    """
    The DQPipeline class represents the control class/main class for managing and executing data quality pipelines.
    It inherits from the PipelinesConfiguration and Experiment classes.
        - The PipelinesConfiguration class contains various pre-configured pipelines.
        - The Experiment class is used for the evaluation of the injected anomalies.

    Methods
    -------
    run_pipeline:
        Executes the pipeline based on the provided data and configuration:
        - If only training data is provided, the pipeline runs in an unsupervised mode on the training data, assigning an anomaly score using the specified anomaly detection method.
        - If training data, test data, and anomalies are provided, the anomaly detection method is trained on the training data. Anomalies are then injected into the test data, and predictions are made on that data.
        - If no anomalies are injected and only training and test data are provided, the anomaly detection method is trained on the training data and predicts anomalies in the test data, assigning an anomaly score.

    standard_pipeline_configuration:
        Creates a generic/default pipeline that can be used if no specific parameters are set.
        - Includes standard transformation steps to preprocess the data.

    pipeline_configuration:
        Returns the configured pipeline object to transform the data. This method:
        1. Loads pre-configured pipelines from the PipelinesConfiguration class.
        2. Loads a standard pipeline from the `standard_pipeline_configuration` method.
        - If nominal or ordinal columns are defined, they are excluded from the standard pipeline (`self.exclude_columns`).
        3. Creates a separate pipeline for the defined nominal or ordinal columns.
        - This is necessary as nominal or ordinal columns can be present in numerical format.
        - The dtype_include parameter is set to None to accommodate this.

    Parameters
    ----------
    time_column_names : list
        List of column names representing time data that should be converted to timestamp data types.

    nominal_columns : list
        Columns that should be transformed to nominal data types.

    ordinal_columns : list
        Columns that should be transformed to ordinal data types.

    exclude_columns : list
        List of columns to be dropped from the dataset.

    pattern_recognition_exclude_columns : list
        List of columns to be excluded from pattern recognition.

    remove_columns_with_no_variance : bool
        If set to True, all columns with zero standard deviation/variance will be removed.

    deactivate_pattern_recognition : bool
        If set to True, the pattern recognition transformer will be deactivated.

    Attributes
    ----------
    X_train_transformed : pd.DataFrame
        Transformed training data.

    X_test_transformed : pd.DataFrame
        Transformed test data.

    anomaly_indices : list
        Indices of the injected anomalies.

    feature_importances : list
        Feature importances of the model (if applicable).

    results_experiment : dict
        Results of the experiment.
    """


    def __init__(
        self,
        time_column_names: list = None,
        nominal_columns: list = None,
        ordinal_columns: list = None,
        exclude_columns: list = None,
        pattern_recognition_exclude_columns: list = None,
        remove_columns_with_no_variance: bool = False,
        deactivate_pattern_recognition: bool = False,
        timeframe_random_state_experiment: int = 42,
    ):
        super().__init__()
        self.X_test_output_injected = None
        self.time_column_names = time_column_names
        self.nominal_columns = nominal_columns
        self.ordinal_columns = ordinal_columns
        self.exclude_columns = exclude_columns
        self.pattern_recognition_exclude_columns = pattern_recognition_exclude_columns
        self.deactivate_pattern_recognition = deactivate_pattern_recognition

        self.timeframe_random_state_experiment = timeframe_random_state_experiment

        self.model_name = ""

        # For-Loop removes time_column_names from other class if time_columns should be excluded.
        if self.exclude_columns is not None:
            for col in self.exclude_columns:
                try:
                    if col in self.time_column_names:
                        self.time_column_names.remove(col)
                except Exception as e:
                    print(e)
                    


        self.remove_columns_with_no_variance = remove_columns_with_no_variance
        self._pipeline_structure = self.pipeline_configuration()

    def run_pipeline(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame = None,
        clf: pyod.models = None,
        dump_model: bool = False,
        inject_anomalies: pd.DataFrame = None,
    ) -> Pipeline:
        """
        Executes the pipeline based on the provided data and configuration.

        Behavior
        --------
        - If only training data (X_train) is provided, the pipeline will run in unsupervised mode and returns a dataframe with anomalyscore.
        - Else if (X_train, X_test) are provided, the pipeline will be trained on train data and predict test data.
        - Else if (X_train, X_test, inject_anomalies) are provided, the pipeline will inject the anomalies in test data, trains the train data and predict the test data.


        Parameters
        ----------
        X_train : pd.DataFrame
            The training data to be used by the pipeline.

        X_test : pd.DataFrame, optional
            The test data to be used by the pipeline. Default is None.

        clf : pyod.models, optional
            The anomaly detection model to be used by the pipeline. Default is None.

        dump_model : bool, optional
            If set to True, the trained model will be saved. Default is False.

        inject_anomalies : pd.DataFrame, optional
            DataFrame containing anomalies to be injected into the test data. Default is None.

        Returns
        -------
        Pipeline
            The configured and executed pipeline object.

        """
        self.model_name = type(clf).__name__
        try:
            if X_test is None and clf is not None:
                print("Running unsupervised Pipeline on train data (X_train)...")
                return self.run_only_pipeline(X_train=X_train, clf=clf, dump_model=dump_model)
            elif X_test is not None:
                if isinstance(inject_anomalies, pd.DataFrame):
                    print("Running Pipeline with injected Anomalies, train and test data (X_train, X_test, inject_anomalies)...")
                    return self.run_pipeline_with_AD_injection(
                        X_train=X_train,
                        X_test=X_test,
                        clf=clf,
                        dump_model=dump_model,
                        inject_anomalies=inject_anomalies,
                        timeframe_random_state_experiment=self.timeframe_random_state_experiment,
                    )
                else:
                    print("Running Pipeline without Anomalies, but with test data... (X_train, X_test)")
                    return self.run_pipeline_with_AD_no_injection(
                        X_train=X_train, X_test=X_test, clf=clf, dump_model=dump_model
                    )
        except Exception as e:
                print("An error occurred during the execution of the pipeline.")
                print(e)

    def standard_pipeline_configuration(self):
        return Pipeline(
            steps=[
                (
                    "Settings - Profiling and Datatypes",
                    ColumnTransformer(
                        transformers=[
                            (
                                "X",
                                super().pre_pipeline(
                                    time_column_names=self.time_column_names,
                                    exclude_columns=self.exclude_columns,
                                ),
                                make_column_selector(dtype_include=None),
                            ),
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                    ),
                ),
                (
                    "Preprocessing - Categorical, numerical and timeseries data",
                    ColumnTransformer(
                        transformers=[
                            (
                                "Numerical",
                                super().numeric_pipeline(),
                                make_column_selector(dtype_include=np.number),
                            ),
                            (
                                "Categorical",
                                super().categorical_pipeline(),
                                make_column_selector(dtype_include=np.object_),
                            ),
                            (
                                "Datetime",
                                super().timeseries_pipeline(),
                                make_column_selector(
                                    dtype_include=(
                                        np.dtype("datetime64[ns]"),
                                        np.datetime64,
                                        "datetimetz",
                                    )
                                ),
                            ),
                        ],
                        remainder="drop",
                        n_jobs=-1,
                        verbose=True,
                    ),
                ),
            ]
        )

    def pipeline_configuration(self):
        if self.nominal_columns is None and self.ordinal_columns is None:
            standard_pipeline = self.standard_pipeline_configuration()
            return Pipeline(
                steps=[
                    (
                        "Data Quality - Pipeline",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "Preprocessing Pipeline",
                                    standard_pipeline,
                                    make_column_selector(dtype_include=None),
                                ),
                                (
                                    "NaNMarker Pipeline",
                                    super().nan_marker_pipeline(),
                                    make_column_selector(dtype_include=None),
                                ),
                                (
                                    "Categorical_PatternExtraction",
                                    super().pattern_extraction(
                                        pattern_recognition_exclude_columns=self.pattern_recognition_exclude_columns,
                                        time_column_names_pattern=self.time_column_names,
                                        deactivate_pattern_recognition=self.deactivate_pattern_recognition,
                                    ),
                                    make_column_selector(dtype_include=np.object_),
                                ),
                            ],
                            remainder="drop",
                            n_jobs=-1,
                            verbose=True,
                        ),
                    )
                ]
            )

        elif self.nominal_columns is not None and self.ordinal_columns is not None:
            if self.exclude_columns is None:
                self.exclude_columns = self.nominal_columns + self.ordinal_columns
            else:
                self.exclude_columns.extend(self.nominal_columns + self.ordinal_columns)
                self.exclude_columns = set(self.exclude_columns)

            standard_pipeline = self.standard_pipeline_configuration()
            return Pipeline(
                steps=[
                    (
                        "Data Quality - Pipeline",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "Preprocessing Pipeline",
                                    standard_pipeline,
                                    make_column_selector(dtype_include=None),
                                ),
                                (
                                    "Nominal Columns",
                                    super().nominal_pipeline(
                                        nominal_columns=self.nominal_columns
                                    ),
                                    make_column_selector(dtype_include=None),
                                ),
                                (
                                    "Ordinal Columns",
                                    super().ordinal_pipeline(
                                        ordinal_columns=self.ordinal_columns
                                    ),
                                    make_column_selector(dtype_include=None),
                                ),
                                (
                                    "NaNMarker Pipeline",
                                    super().nan_marker_pipeline(),
                                    make_column_selector(dtype_include=None),
                                ),
                                (
                                    "Categorical_PatternExtraction",
                                    super().pattern_extraction(
                                        pattern_recognition_exclude_columns=self.pattern_recognition_exclude_columns,
                                        time_column_names_pattern=self.time_column_names,
                                        deactivate_pattern_recognition=self.deactivate_pattern_recognition,
                                    ),
                                    make_column_selector(dtype_include=np.object_),
                                ),
                            ],
                            remainder="drop",
                            n_jobs=-1,
                            verbose=True,
                        ),
                    )
                ]
            )

        elif self.ordinal_columns is None and self.nominal_columns is not None:
            if self.exclude_columns is None:
                self.exclude_columns = self.nominal_columns
            else:
                self.exclude_columns.extend(self.nominal_columns)
                self.exclude_columns = set(self.exclude_columns)

            standard_pipeline = self.standard_pipeline_configuration()
            return Pipeline(
                steps=[
                    (
                        "Data Quality - Pipeline",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "Preprocessing Pipeline",
                                    standard_pipeline,
                                    make_column_selector(dtype_include=None),
                                ),
                                (
                                    "Nominal Columns",
                                    super().nominal_pipeline(
                                        nominal_columns=self.nominal_columns
                                    ),
                                    make_column_selector(dtype_include=None),
                                ),
                                # ("Ordinal Columns",super().ordinal_pipeline(ordinal_columns=self.ordinal_columns),make_column_selector(dtype_include=None)),
                                (
                                    "NaNMarker Pipeline",
                                    super().nan_marker_pipeline(),
                                    make_column_selector(dtype_include=None),
                                ),
                                (
                                    "Categorical_PatternExtraction",
                                    super().pattern_extraction(
                                        pattern_recognition_exclude_columns=self.pattern_recognition_exclude_columns,
                                        time_column_names_pattern=self.time_column_names,
                                        deactivate_pattern_recognition=self.deactivate_pattern_recognition,
                                    ),
                                    make_column_selector(dtype_include=np.object_),
                                ),
                            ],
                            remainder="drop",
                            n_jobs=-1,
                            verbose=True,
                        ),
                    )
                ]
            )

        elif self.nominal_columns is None and self.ordinal_columns is not None:
            if self.exclude_columns is None:
                self.exclude_columns = self.ordinal_columns
            else:
                self.exclude_columns.extend(self.ordinal_columns)
                self.exclude_columns = set(self.exclude_columns)

            standard_pipeline = self.standard_pipeline_configuration()
            return Pipeline(
                steps=[
                    (
                        "Data Quality - Pipeline",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "Preprocessing Pipeline",
                                    standard_pipeline,
                                    make_column_selector(dtype_include=None),
                                ),
                                # ("Nominal Columns",super().nominal_pipeline(nominal_columns=self.nominal_columns),make_column_selector(dtype_include=None)),
                                (
                                    "Ordinal Columns",
                                    super().ordinal_pipeline(
                                        ordinal_columns=self.ordinal_columns
                                    ),
                                    make_column_selector(dtype_include=None),
                                ),
                                (
                                    "NaNMarker Pipeline",
                                    super().nan_marker_pipeline(),
                                    make_column_selector(dtype_include=None),
                                ),
                                (
                                    "Categorical_PatternExtraction",
                                    super().pattern_extraction(
                                        pattern_recognition_exclude_columns=self.pattern_recognition_exclude_columns,
                                        time_column_names_pattern=self.time_column_names,
                                        deactivate_pattern_recognition=self.deactivate_pattern_recognition,
                                    ),
                                    make_column_selector(dtype_include=np.object_),
                                ),
                            ],
                            remainder="drop",
                            n_jobs=-1,
                            verbose=True,
                        ),
                    )
                ]
            )

    @property
    def X_train_transformed(self):
        return super().X_train_transformed

    @property
    def X_test_anomalies_transformed(self):
        return super().X_test_anomalies_transformed

    @property
    def index_of_anomalies(self):
        return super().index_of_anomalies

    @property
    def feature_importances(self):
        return super().feature_importances

    @property
    def results_experiment(self):
        return super().results_experiment



    def run_only_pipeline(self, X_train: pd.DataFrame, clf, dump_model) -> Pipeline:
        """
        Only the data will be transformed by the pipeline.
        """
        print("Running only Transformation-Pipeline...")
        try:
            X_train_transformed = self._pipeline_structure.fit_transform(X_train)
        except Exception as e:
            print(X_train.isna().sum(),"\n",e,"\n")
            raise


        X_train_transformed = self.check_variance_train_only(
            X_train=X_train_transformed,
            remove_cols=self.remove_columns_with_no_variance,
        )

        self._X_train_transformed = X_train_transformed

        clf_no_injection = clf
        clf_no_injection.fit(X_train_transformed)

        if dump_model == True:
            try:
                dump(clf, f"clf_{type(clf).__name__}.joblib")
            except:
                print("Could not dump the model.")

        y_pred_decision_score = clf_no_injection.decision_function(X_train_transformed)
        X_train["AnomalyScore"] = y_pred_decision_score
        scaler = MinMaxScaler()
        X_train[["AnomalyScore"]] = scaler.fit_transform(X_train[["AnomalyScore"]])

        try:
            column_name_mad_total = [
                col for col in X_train_transformed.columns if col.endswith("MAD_Total")
            ][0]
            X_train["MAD_Total"] = X_train_transformed[column_name_mad_total]
            column_name_tukey_total = [
                col for col in X_train_transformed.columns if col.endswith("Tukey_Total")
            ][0]
            X_train["Tukey_Total"] = X_train_transformed[column_name_tukey_total]

            return X_train.sort_values(
                ["AnomalyScore", "MAD_Total", "Tukey_Total"], ascending=False
            )
        except Exception as e:
            return X_train.sort_values("AnomalyScore", ascending=False)



    def run_pipeline_with_AD_no_injection(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        clf: pyod.models,
        dump_model: bool,
    ):
        print("Running Pipeline without injected Anomalies...")
        """
        Processes the data through the pipeline and returns the test dataset with corresponding anomaly values.
        
        Purpose:
        - Used for monitoring purposes.
        
        Optional:
        - The model can be temporarily stored using the `dump_model` parameter.
        """
        self._pipeline_structure.fit(X_train)

        X_train_transformed = self._pipeline_structure.transform(X_train)

        X_test_transformed = self._pipeline_structure.transform(X_test)

        X_train_transformed, X_test_transformed = self.check_variance_monitoring(
            X_train=X_train_transformed,
            X_test=X_test_transformed,
            remove_cols=self.remove_columns_with_no_variance,
        )

        self._X_train_transformed = X_train_transformed
        self._X_test_transformed = X_test_transformed

        clf_no_injection = clf
        clf_no_injection.fit(X_train_transformed)

        if dump_model == True:
            try:
                dump(clf, f"clf_{type(clf).__name__}.joblib")
            except:
                print("Could not dump the model.")

        y_pred_decision_score = clf_no_injection.decision_function(X_test_transformed)
        X_test["AnomalyScore"] = y_pred_decision_score
        scaler = MinMaxScaler()
        X_test[["AnomalyScore"]] = scaler.fit_transform(X_test[["AnomalyScore"]])

        try:
            column_name_mad_total = [
                col for col in X_test_transformed.columns if col.endswith("MAD_Total")
            ][0]
            X_test["MAD_Total"] = X_test_transformed[column_name_mad_total]
            column_name_tukey_total = [
                col for col in X_test_transformed.columns if col.endswith("Tukey_Total")
            ][0]
            X_test["Tukey_Total"] = X_test_transformed[column_name_tukey_total]

            return X_test.sort_values(
                ["AnomalyScore", "MAD_Total", "Tukey_Total"], ascending=False
            )
        except Exception as e:
            return X_test.sort_values("AnomalyScore", ascending=False)

    def run_pipeline_with_AD_injection(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        clf: pyod.models,
        dump_model: bool,
        inject_anomalies: pd.DataFrame,
        timeframe_random_state_experiment: str,
    ) -> pd.DataFrame:
        """
        Method to conduct the experiment.

        Steps:
            1. Reassignment of the index for injected anomaly values (end of the test data).
            2. Merging of the test data with the injected anomalies (without their y_true label).
            3. Resetting the index for the test and training data to avoid having two index columns.
            4. Data is processed through the pipeline (structure provided in the method).
            5. Evaluation of the injected anomalies and extraction of quality metrics.
        """
        print("Running Pipeline with injected Anomalies...")
        if isinstance(inject_anomalies, pd.DataFrame):
            self.X_test_output_injected = super().start_experiment(
                X_train=X_train,
                X_test=X_test,
                injected_anomalies=inject_anomalies,
                pipeline_structure=self._pipeline_structure,
                clf=clf,
                remove_columns_with_no_variance=self.remove_columns_with_no_variance,
                timeframe_random_state_experiment=timeframe_random_state_experiment,
            )
            # self.feature_importances = super().get_feature_importances()
            return self.X_test_output_injected
        else:
            print("Invalid Injection of Anomalies.")

    def check_variance_train_only(
        self, X_train, remove_cols=False
    ) -> (pd.DataFrame, pd.DataFrame):
        X_train_cols_no_variance = X_train.loc[:, X_train.std() == 0].columns
        print("No Variance in follow Train Columns: ", X_train_cols_no_variance)

        X_train_cols_only_nans = X_train.columns[X_train.isna().any()]
        print("Only NaNs in Train Columns: ", X_train_cols_only_nans)

        drop_columns_no_variance =  X_train_cols_no_variance.tolist() +  X_train_cols_only_nans.tolist()
        
        # print("\n\n\n\n")
        # print(drop_columns_no_variance)

        if remove_cols == True:
            X_train_dropped = X_train.drop(drop_columns_no_variance, axis=1)

            print(
                f"Shape Train after drop: {X_train_dropped.shape} "
            )
            print(
                f"Check NaN Train: {X_train_dropped.columns[X_train_dropped.isna().any()].tolist()} "
            )
            if len(X_train_dropped.columns[X_train_dropped.isna().any()].tolist()) > 0:
                Path("./debugging").mkdir(parents=True, exist_ok=True)
                X_train_dropped[X_train_dropped.isna().any(axis=1)].to_csv("./debugging/NaN_Train_Cols.csv", index=False)

            print(
                f"Check inf Train: {X_train_dropped.columns[np.isinf(X_train_dropped).any()].tolist()} "
            )
            if len(X_train_dropped.columns[X_train_dropped.isna().any()].tolist()) > 0:
                Path("./debugging").mkdir(parents=True, exist_ok=True)
                X_train_dropped.columns[X_train_dropped.isna().any()].tolist().to_csv("./debugging/Inf_Train_Cols.csv", index=False)

            return X_train_dropped
        else:
            return X_train




    def check_variance_monitoring(
        self, X_train, X_test, remove_cols=False
    ) -> (pd.DataFrame, pd.DataFrame):
        X_train_cols_no_variance = X_train.loc[:, X_train.std() == 0].columns
        X_test_cols_no_variance = X_test.loc[:, X_test.std() == 0].columns

        print("No Variance in follow Train Columns: ", X_train_cols_no_variance)
        print("No Variance in follow Test Columns: ", X_test_cols_no_variance)

        X_train_cols_only_nans = X_train.columns[X_train.isna().any()]
        X_test_cols_only_nans = X_test.columns[X_test.isna().any()]

        print("Only NaNs in follow Train Columns: ", X_train_cols_only_nans)
        print("Only NaNs in follow Test Columns: ", X_test_cols_only_nans)

        drop_columns_no_variance = (
            X_train_cols_no_variance.tolist() + X_test_cols_no_variance.tolist() + \
            X_train_cols_only_nans + X_test_cols_only_nans
        )
        # print(drop_columns_no_variance)

        print(f"Remove columns with zero variance: {remove_cols}")

        print(
            f"Shape Train before drop:{X_train.shape} / Shape Test before drop: {X_test.shape} "
        )

        if remove_cols == True:
            X_train_dropped = X_train.drop(drop_columns_no_variance, axis=1)
            X_test_dropped = X_test.drop(drop_columns_no_variance, axis=1)
            print(
                f"Shape Train after drop: {X_train_dropped.shape} / Shape Test after drop: {X_test_dropped.shape}\n"
            )
            print(
                f"Check NaN Train: {X_train_dropped.columns[X_train_dropped.isna().any()].tolist()} "
            )
            print(
                f"Check inf Train: {X_train_dropped.columns[np.isinf(X_train_dropped).any()].tolist()} "
            )
            return X_train_dropped, X_test_dropped
        else:
            return X_train, X_test

    def visualize_pipeline_structure_html(self, filename="./visualization/PipelineDQ"):
        """
        Speichert die Pipeline als html-Datei.
        """
        Path("./visualization").mkdir(parents=True, exist_ok=True)
        Path("./visualization/PipelineDQ").mkdir(parents=True, exist_ok=True)
        with open(file=f"{filename}.html", mode="w", encoding="utf-8") as f:
            f.write(estimator_html_repr(self._pipeline_structure))
            f.close()

    def get_profiling(self, X: pd.DataFrame, deeper_profiling=False):
        """
        Gibt ein Profiling der Daten aus.\n
        """
        return super().get_profiling(X, deeper_profiling)

    def show_injected_anomalies(self):
        """
        Ausgabe der injizierten Anomalien.\n
        """
        from pathlib import Path
        Path("./ERGEBNISSE/output").mkdir(parents=True, exist_ok=True)
        if isinstance(self.X_test_output_injected, pd.DataFrame):
            index_anomalies = self.index_of_anomalies
            self.X_test_output_injected.loc[index_anomalies].to_csv(f"./ERGEBNISSE/output/{self.model_name}_{self.timeframe_random_state_experiment}_injected.csv", index=False, mode="w", header=True)


            return self.X_test_output_injected.loc[index_anomalies]
        else:
            print("No anomalies were injected")


# from pyod.models.auto_encoder_torch import AutoEncoder




def initialize_autoencoder():

    from pyod_modified.CustomAutoEncoder.torchCustomAE import AutoEncoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # batch_size_user = 2048
    batch_size_user = 2048 * 4

    print("Batch size: %i" % (batch_size_user))

    try:
        num_cores = os.cpu_count() - 1
        torch.set_num_threads(num_cores)
        print(f"{num_cores} cores will be used...")

        clf_ae = AutoEncoder(
            epochs=50,
            batch_size=batch_size_user,
            learning_rate=0.001,
            contamination=0.1,
        )

        return clf_ae

    except:
        print("Could not use all CPU Threads...")
        clf_ae = AutoEncoder(
            epochs=50,
            batch_size=batch_size_user,
            learning_rate=0.001,
            contamination=0.1,
        )

        return clf_ae


def initialize_autoencoder_modified(epochs=20):
    import torch
    from pyod_modified.CustomAutoEncoder.torchCustomAE import AutoEncoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # batch_size_user = 2048
    batch_size_user = 2048 * 4

    print("Batch size: %i" % (batch_size_user))

    try:
        num_cores = os.cpu_count() - 1
        torch.set_num_threads(num_cores)
        print(f"{num_cores} cores will be used...")

        clf_ae = AutoEncoder(
            epochs=epochs,
            batch_size=batch_size_user,
            learning_rate=0.001,
            contamination=0.1,
            preprocessing=True
        )

        return clf_ae

    except:
        print("Could not use all CPU Threads...")
        clf_ae = AutoEncoder(
            epochs=20,
            batch_size=batch_size_user,
            learning_rate=0.001,
            contamination=0.1,
            preprocessing=True
        )

        return clf_ae


def dummy_data():
    train_data = pd.DataFrame(
    {
        "COLTestCAT1": np.array(
            [
                "Cat",
                "Cat",
                "Dog",
            ]
        ),
        "COLTestCAT2": np.array(["Blue", "Blue", "Red"]),
        "COLNUM": np.array([6, 8, 20]),
        "COLNUM2": np.array([2000, 3500, 2500]),
        "timestamp": np.array(
            [
                "2023-02-08 09:56:14.086000+00:00",
                "2023-02-08 17:12:16.347000+00:00",
                "2023-02-09 17:12:16.347000+00:00",
            ]
        ),
    }
)

    test_data = pd.DataFrame(
        {
            "COLTestCAT1": np.array(["Lion", "Dog", "Dog"]),
            "COLTestCAT2": np.array(["Yellow", "Blue", "Red"]),
            "COLNUM": np.array([6.5, 8, 15]),
            "COLNUM2": np.array([5, 2000, 3000]),
            "timestamp": np.array(
                [
                    "2005-02-08 06:58:14.017000+00:00",
                    "2023-02-08 15:54:13.693000+00:00",
                    "2023-02-09 17:12:16.347000+00:00",
                ]
            ),
        }
    )


    anomaly_data = pd.DataFrame(
        {
            "COLTestCAT1": np.array(["Cat", np.nan, "Giraffe"]),
            "COLTestCAT2": np.array(["Blue", np.nan, "Panda"]),
            "COLNUM": np.array([7, 8, 70]),
            "COLNUM2": np.array([2000, 8, 8000]),
            "timestamp": np.array(
                [
                    "2023-02-16 06:58:14.017000+00:00",
                    "2002-12-08 15:54:13.693000+00:00",
                    np.nan,
                ]
            ),
            "y_true": np.array([0, 1, 1]),
        }
    )

    return train_data, test_data, anomaly_data