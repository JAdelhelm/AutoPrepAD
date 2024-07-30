# %%
import numpy as np
import pandas as pd


from pipelines.defaults import initialize_autoencoder, initialize_autoencoder_modified
from pipelines.defaults import dummy_data
pd.set_option("display.max_columns", None)
# from pyod.models.iforest import IForest
# from pyod.models.lof import LOF
from pyod.models.pca import PCA

from pipelines.control import AutoPrepAD

if __name__ == "__main__":
    df_data = pd.read_csv("./temperature_USA.csv")

    # clf_if = IForest(n_jobs=-1)
    clf_pca = PCA()
    # clf_ae = initialize_autoencoder_modified()

    pipeline = AutoPrepAD()

    pipeline.fit(
        X_train=df_data,
        clf=clf_pca,
        dump_model=False,
    )
    
    X_output = pipeline.predict(X_test=df_data)


    X_output.to_csv("temperatures_anomalies.csv", index=False)

    # anomaly_detection_pipeline.visualize_pipeline_structure_html()




    


# %%


