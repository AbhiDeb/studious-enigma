import traceback
from typing import Optional
from fastapi import FastAPI
import mlflow
from mlflow.tracking import MlflowClient

import os
import pickle
import pandas as pd

relative_model_dev_path = '.s'

model_name = "Bias Mitigation Malware Detection"
mlflow.set_tracking_uri(f"sqlite:///{relative_model_dev_path}/Malware_Detection_MLFlow.db")
runs = mlflow.search_runs(experiment_ids=['1'])
client = MlflowClient()

# sys.path.insert(1, relative_model_dev_path)
from auto_feat import MakeDataSet

def load_test_dataframe():
    """Load"""
    try:
        # print(settings.DATASET_PATH)
        data_frame = pd.read_csv("../datasource/train2.csv")
        shuffled_df = data_frame.sample(
            frac=1, random_state=107).reset_index(drop=True)

        # for column in data_frame.columns:
        data_frame.columns = [column.strip() for column in data_frame.columns] 
        
        # my_report = sv.compare_intra(shuffled_df,shuffled_df[settings.Y_COLUMN[0]] == 1,["Malware Detected", "No Malware Detected"])
        # my_report.show_html(filepath='./reports/eda/eda_report.html',open_browser=False)
        # print(shuffled_df.shape)
        return shuffled_df
    except Exception as e:
        print(traceback.format_exc())

md = MakeDataSet()

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predictions")

def predict(wdft_RegionIdentifier: float, Census_IsTouchEnabled: float,	Census_IsSecureBootEnabled: float,	IsSxsPassiveMode: float, Census_MDC2FormFactor: float,	Census_OSWUAutoUpdateOptionsName: float, OsSuite: float, Census_IsPortableOperatingSystem: float, Census_ActivationChannel: float, Census_OSArchitecture: float, AutoSampleOptIn: float, Census_GenuineStateName: float, Census_IsPenCapable: float, Processor: float,	ProductName: float,	Census_PrimaryDiskTypeName: float, Platform: float, HasTpm: float, Census_HasOpticalDiskDrive: float, SkuEdition: float, IsBeta: float,	OsPlatformSubRelease: float, Census_FlightRing: float, Census_OSInstallTypeName: float, Census_DeviceFamily: float, SmartScreen: float):
    
    data = None
    try:
        data = log_production_model()
    except Exception as e:
        print(e)
    
    if data == None:
        return {"predicted_class":"","error" : "No model in production in registry"}
    
    with open(f'{relative_model_dev_path}/{data[2:]}/model.pkl', "rb") as f:
        loaded_model = pickle.load(f)
    
    row_list = [wdft_RegionIdentifier , Census_IsTouchEnabled ,	Census_IsSecureBootEnabled,	IsSxsPassiveMode, Census_MDC2FormFactor, Census_OSWUAutoUpdateOptionsName, OsSuite, Census_IsPortableOperatingSystem, Census_ActivationChannel, Census_OSArchitecture, AutoSampleOptIn, Census_GenuineStateName, Census_IsPenCapable, Processor, ProductName, Census_PrimaryDiskTypeName, Platform, HasTpm, Census_HasOpticalDiskDrive, SkuEdition, IsBeta,	OsPlatformSubRelease, Census_FlightRing, Census_OSInstallTypeName, Census_DeviceFamily, SmartScreen ,999]
    col_list = ['wdft_RegionIdentifier' , 'Census_IsTouchEnabled' ,	'Census_IsSecureBootEnabled',	'IsSxsPassiveMode', 'Census_MDC2FormFactor', 'Census_OSWUAutoUpdateOptionsName', 'OsSuite', 'Census_IsPortableOperatingSystem', 'Census_ActivationChannel', 'Census_OSArchitecture', 'AutoSampleOptIn', 'Census_GenuineStateName', 'Census_IsPenCapable', 'Processor', 'ProductName', 'Census_PrimaryDiskTypeName', 'Platform', 'HasTpm', 'Census_HasOpticalDiskDrive', 'SkuEdition', 'IsBeta',	'OsPlatformSubRelease', 'Census_FlightRing', 'Census_OSInstallTypeName', 'Census_DeviceFamily', 'SmartScreen','HasDetections']
    row_df   = pd.DataFrame([row_list],columns = col_list)
   
    row_aif360 = md.decode_dataset(data_frame=row_df)
    
    prediction = loaded_model.predict(row_aif360.convert_to_dataframe()[0].drop('HasDetections',axis=1))[0]
    
    return {"artifact_path": f'{data[2:]}/model.pkl', "predicted_class":int(prediction)}


def log_production_model():
    
    
    logged_model = None
    for reg_mod in client.list_registered_models():
        for versions in reg_mod.latest_versions:
            if versions.current_stage == 'Production':
                logged_model = versions.source
                break
    return logged_model