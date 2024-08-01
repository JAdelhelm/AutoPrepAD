from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# from graphviz import Digraph
from category_encoders import BinaryEncoder

# import tensorflow as tf
import pyod
from sklearn.utils import estimator_html_repr



# import pdfkit

# Activate if you use PyTorch algorithms
# import torch

from pipelines.configuration import PipelinesConfiguration



class ConfigurationControl(PipelinesConfiguration):
    """
    The ConfigurationControl class extends PipelinesConfiguration and manages the configuration 
    of preprocessing pipelines for anomaly detection.

    Parameters
    ----------
    datetime_columns : list, optional
        List of column names representing time data that should be converted to timestamp data types. Default is None.

    nominal_columns : list, optional
        Columns that should be transformed to nominal data types. Default is None.

    ordinal_columns : list, optional
        Columns that should be transformed to ordinal data types. Default is None.

    pattern_recognition_exclude_columns : list, optional
        List of columns to be excluded from pattern recognition. Default is None.

    exclude_columns : list, optional
        List of columns to be dropped from the dataset. Default is None.

    deactivate_pattern_recognition : bool, optional
        If set to True, the pattern recognition transformer will be deactivated. Default is False.

    Methods
    -------
    standard_pipeline_configuration()
        Returns the standard pipeline configuration with profiling, datatypes, and preprocessing steps.

    pipeline_configuration()
        Returns the complete pipeline configuration based on provided columns and settings.

    """
    def __init__(self,
        datetime_columns: list = None,
        nominal_columns: list = None,
        ordinal_columns: list = None,
        pattern_recognition_exclude_columns: list = None,
        exclude_columns: list = None,
        deactivate_pattern_recognition: bool = False
                 ) -> None:
        super().__init__()
        self.datetime_columns = datetime_columns
        self.nominal_columns = nominal_columns
        self.ordinal_columns = ordinal_columns
        self.exclude_columns = exclude_columns
        self.pattern_recognition_exclude_columns = pattern_recognition_exclude_columns
        self.deactivate_pattern_recognition = deactivate_pattern_recognition

    def standard_pipeline_configuration(self):
        """
        Create and return the standard pipeline configuration.

        The pipeline includes profiling, datatype transformation, and preprocessing steps
        for categorical, numerical, and timeseries data.

        Returns
        -------
        Pipeline
            A configured sklearn Pipeline object.
        """
        return Pipeline(
            steps=[
                (
                    "Settings - Profiling and Datatypes",
                    ColumnTransformer(
                        transformers=[
                            (
                                "X",
                                super().pre_pipeline(
                                    datetime_columns=self.datetime_columns,
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
        """
        Create and return the complete pipeline configuration.

        The pipeline includes the standard pipeline configuration along with
        additional steps for handling nominal and ordinal columns, NaN marker,
        and pattern extraction based on the provided settings.

        Returns
        -------
        Pipeline
            A configured sklearn Pipeline object.
        """
        if self.nominal_columns is None and self.ordinal_columns is None:
            standard_pipeline = self.standard_pipeline_configuration()
            return Pipeline(
                steps=[
                    (
                        "Automated Anomaly Detection Pipeline",
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
                                        datetime_columns_pattern=self.datetime_columns,
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
                        "Automated Anomaly Detection Pipeline",
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
                                        datetime_columns_pattern=self.datetime_columns,
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
                        "Automated Anomaly Detection Pipeline",
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
                                # (
                                #     "NaNMarker Pipeline",
                                #     super().nan_marker_pipeline(),
                                #     make_column_selector(dtype_include=None),
                                # ),
                                (
                                    "Categorical_PatternExtraction",
                                    super().pattern_extraction(
                                        pattern_recognition_exclude_columns=self.pattern_recognition_exclude_columns,
                                        datetime_columns_pattern=self.datetime_columns,
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
                        "Automated Anomaly Detection Pipeline",
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
                                        datetime_columns_pattern=self.datetime_columns,
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