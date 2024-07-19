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

# Activate if you use PyTorch algorithms
# import torch

from pipelines.configuration import PipelinesConfiguration



class ConfigurationControl(PipelinesConfiguration):
    def __init__(self,
        time_column_names: list = None,
        nominal_columns: list = None,
        ordinal_columns: list = None,
        pattern_recognition_exclude_columns: list = None,
        exclude_columns: list = None,
        deactivate_pattern_recognition: bool = False
                 ) -> None:
        super().__init__()
        self.time_column_names = time_column_names
        self.nominal_columns = nominal_columns
        self.ordinal_columns = ordinal_columns
        self.exclude_columns = exclude_columns
        self.pattern_recognition_exclude_columns = pattern_recognition_exclude_columns
        self.deactivate_pattern_recognition = deactivate_pattern_recognition

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