# %%
import numpy as np
import pandas as pd

from pipelines.control import AADP
from pipelines.defaults import initialize_autoencoder, initialize_autoencoder_modified
from pipelines.defaults import dummy_data
pd.set_option("display.max_columns", None)
# from pyod.models.iforest import IForest
# from pyod.models.lof import LOF
from pyod.models.pca import PCA



if __name__ == "__main__":
    df_data = pd.read_csv("./temperature_USA.csv")

    # clf_if = IForest(n_jobs=-1)
    clf_pca = PCA()

    anomaly_detection_pipeline = AADP(
        exclude_columns=[],
        deactivate_pattern_recognition=True,
        exclude_columns_no_variance=True,
        mark_anomalies_pct_data=0.01
    )

    anomaly_detection_pipeline.fit(
        X_train=df_data,
        clf=clf_pca,
        dump_model=False,
    )
    
    X_output = anomaly_detection_pipeline.predict(X_test=df_data)


    X_output.to_csv("temperatures_anomalies.csv", index=False)

    # anomaly_detection_pipeline.visualize_pipeline_structure_html()




    


# %%


