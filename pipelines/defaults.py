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


import os
from joblib import dump
import itertools
from pathlib import Path

# from pyod.models.auto_encoder_torch import AutoEncoder


def initialize_autoencoder():
    import torch
    from pipelines.pyod_modified.CustomAutoEncoder.torchCustomAE import AutoEncoder

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
    # print(os.getcwd())
    import torch
    from pipelines.pyod_modified.CustomAutoEncoder.torchCustomAE import AutoEncoder
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



