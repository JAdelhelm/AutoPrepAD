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

class AutoPrepAD():
    """
    The AADP (Automated Anomaly Detection Pipeline) class represents the control class/main class for managing and executing configurated pipelines.

    Parameters
    ----------
    datetime_columns : list
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
        datetime_columns: list = None,
        nominal_columns: list = None,
        ordinal_columns: list = None,
        exclude_columns: list = None,
        pattern_recognition_exclude_columns: list = None,
        exclude_columns_no_variance: bool = True,
        deactivate_pattern_recognition: bool = True,
        mark_anomalies_pct_data: float = 0.1):
        #super().__init__()
        self.X_test_output_injected = None
        self.datetime_columns = datetime_columns
        self.nominal_columns = nominal_columns
        self.ordinal_columns = ordinal_columns
        self.exclude_columns = exclude_columns
        self.pattern_recognition_exclude_columns = pattern_recognition_exclude_columns
        self.deactivate_pattern_recognition = deactivate_pattern_recognition
        self.mark_anomalies_pct_data = mark_anomalies_pct_data

        self.exclude_columns_no_variance = exclude_columns_no_variance

        self.model_name = ""

        if self.datetime_columns is not None:
            if self.exclude_columns is None:
                self.exclude_columns = []
            self.exclude_columns = self.exclude_columns + self.datetime_columns




        self.X_train_transformed = None
        self.X_test_transformed = None

        config_control = ConfigurationControl(
            datetime_columns = self.datetime_columns,
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


    def fit(
            self, 
            X_train: pd.DataFrame,
            clf: pyod.models = None,
            dump_model: bool = False,
    ) -> Pipeline:
        
        return self.runs.fit_pipeline(
            X_train=X_train,
            clf=clf,
            dump_model=dump_model
        )
    

    def predict(
            self,
            X_test: pd.DataFrame
    ) -> pd.DataFrame:
        
        return self.runs.predict_pipeline(
            X_test=X_test
        )
        


    def visualize_pipeline_structure_html(self, filename="./visualization/PipelineDQ"):
        """
        Saves pipeline structure as html.
        """
        Path("./visualization").mkdir(parents=True, exist_ok=True)
        Path("./visualization/PipelineDQ").mkdir(parents=True, exist_ok=True)
        with open(file=f"{filename}.html", mode="w", encoding="utf-8") as f:
            f.write(estimator_html_repr(self.pipeline_structure))
            f.close()





    
