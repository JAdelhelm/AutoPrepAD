# %%
import numpy as np
import pandas as pd

from pipelines.control import AADP
from pipelines.defaults import initialize_autoencoder, initialize_autoencoder_modified
from pipelines.defaults import dummy_data
pd.set_option("display.max_columns", None)
from pyod.models.iforest import IForest
from pyod.models.lof import LOF



if __name__ == "__main__":
    df_data = pd.read_csv("./players_20.csv")

    clf_lof = LOF(n_jobs=-1)

    anomaly_detection_pipeline = AADP(
        deactivate_pattern_recognition=True,
        exclude_columns_no_variance=True,
    )

    X_output = anomaly_detection_pipeline.unsupervised_pipeline(
        X_train=df_data.iloc[:,0:37],
        clf=clf_lof,
        dump_model=False,
    )

    X_output.to_csv("fifa_anomalies.csv", index=False)

    anomaly_detection_pipeline.visualize_pipeline_structure_html()




    







# %%


