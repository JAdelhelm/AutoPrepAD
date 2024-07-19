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




# import pdfkit

## Activate if you use PyTorch algorithms
# import torch


import os
from joblib import dump
import itertools
from pathlib import Path

from pipelines.runs import PipelineRuns

try:  from pipelines.configuration_control import ConfigurationControl
except: from configuration_control import ConfigurationControl

set_config(transform_output="pandas")

class AADP():
    """
    The AADP (Automated Anomaly Detection Pipeline) class represents the control class/main class for managing and executing configurated pipelines.

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
        exclude_columns_no_variance: bool = False,
        deactivate_pattern_recognition: bool = False,
        mark_anomalies_pct_data: float = 0.1):
        #super().__init__()
        self.X_test_output_injected = None
        self.time_column_names = time_column_names
        self.nominal_columns = nominal_columns
        self.ordinal_columns = ordinal_columns
        self.exclude_columns = exclude_columns
        self.pattern_recognition_exclude_columns = pattern_recognition_exclude_columns
        self.deactivate_pattern_recognition = deactivate_pattern_recognition
        self.mark_anomalies_pct_data = mark_anomalies_pct_data

        self.exclude_columns_no_variance = exclude_columns_no_variance

        self.model_name = ""

        if self.time_column_names is not None:
            if self.exclude_columns is None:
                self.exclude_columns = []
            self.exclude_columns = self.exclude_columns + self.time_column_names




        self.X_train_transformed = None
        self.X_test_transformed = None

        config_control = ConfigurationControl(
            time_column_names = self.time_column_names,
            nominal_columns = self.nominal_columns,
            ordinal_columns = self.ordinal_columns,
            pattern_recognition_exclude_columns = self.pattern_recognition_exclude_columns,
            deactivate_pattern_recognition =self.deactivate_pattern_recognition
        )
        self.pipeline_structure = config_control.pipeline_configuration()

        self.runs = PipelineRuns(self.pipeline_structure,
                                 remove_columns_with_no_variance = self.exclude_columns_no_variance,
                                 exclude_columns = self.exclude_columns,
                                 mark_anomalies_pct_data=self.mark_anomalies_pct_data)





    def unsupervised_pipeline(
            self, 
            X_train: pd.DataFrame,
            clf: pyod.models = None,
            dump_model: bool = False,
    ) -> Pipeline:
        """
        Runs the pipeline unsuperivsed on input data.

        Behavior
        --------
        1. Transforms input data, based on the pipeline configuration.
        2. Anomaly detection will be executed to find anomalies.
        3. Insertion of extra column (Anomaly score).
        4. Attach column to original dataset.
        """
        self.model_name = type(clf).__name__


        print("Running unsupervised Pipeline on input data (X_train)...")
        return self.runs.unsupervised_run(
            X_train = X_train, 
            clf = clf,
            dump_model = dump_model
        )



    def unsupervised_pipeline_train_test(
        self, 
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        clf: pyod.models = None,
        dump_model: bool = False,
    ) -> Pipeline:
        """
        Runs the pipeline with the usage of train and test-data.

        Behavior
        --------
        1. Transforms train and test data, based on the pipeline configuration.
        2. Anomaly detection will learn the structure of the train data.
        3. Anomaly detection will predict anomalies in test data.
        3. Insertion of extra column (Anomaly score).
        4.  Attach column to original dataset. 
        """
        self.model_name = type(clf).__name__



        print("Running Pipeline on train and test data... (X_train, X_test)")
        return self.runs.unsupervised_train_test_run(
                        X_train=X_train, X_test=X_test, clf=clf, dump_model=dump_model
                    )




    def unsupervised_pipeline_train_test_anomalies(
        self, 
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        clf: pyod.models = None,
        inject_anomalies: pd.DataFrame = None,
        timeframe_random_state_experiment = "42"
    ) -> Pipeline:
        """
        Runs the pipeline with the usage of train, test-data and injected anomalies.

        Behavior
        --------
        1. Injects anomalies in test data + marks them.
        2. Transforms train and test data, based on the pipeline configuration.
        3. Anomaly detection will learn the structure of the train data.
        4. Anomaly detection will predict anomalies in test data.
        5. Insertion of extra column (Anomaly score).
        6.  Attach column to original dataset. 
        """
        self.model_name = type(clf).__name__

        print("Running Pipeline with injected Anomalies, train and test data (X_train, X_test, inject_anomalies)...")
        return self.runs.unsupervised_train_test_anomalies_run(
            X_train=X_train,
            X_test=X_test,
            clf=clf,
            inject_anomalies=inject_anomalies,
            timeframe_random_state_experiment=timeframe_random_state_experiment
        )



    def visualize_pipeline_structure_html(self, filename="./visualization/PipelineDQ"):
        """
        Speichert die Pipeline als html-Datei.
        """
        Path("./visualization").mkdir(parents=True, exist_ok=True)
        Path("./visualization/PipelineDQ").mkdir(parents=True, exist_ok=True)
        with open(file=f"{filename}.html", mode="w", encoding="utf-8") as f:
            f.write(estimator_html_repr(self.pipeline_structure))
            f.close()





    

