import traceback
import pandas as pd
import numpy as np
import os
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from shapash.explainer.smart_explainer import SmartExplainer

relative_model_dev_path = '.'

# import sys
# sys.path.insert(1, relative_model_dev_path)
from auto_feat import MakeDataSet
import settings
md = MakeDataSet()

def find_registered_model(name, uri):
    model_name = name
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_ids=['1'])

    client = MlflowClient()
    logged_model = None
    for reg_mod in client.list_registered_models():
        for versions in reg_mod.latest_versions:
            if versions.current_stage == 'Production':
                logged_model = versions.source
                return logged_model

def get_scaled_data(dataset_orig_train):
    try:
        scalar = settings.SCALAR()
        dataset_copy_train = dataset_orig_train.copy()

        dataset_copy_train.features = scalar.fit_transform(
            dataset_copy_train.features)

        return dataset_copy_train

    except Exception as e:
        print(traceback.format_exc())

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

        
if __name__ == '__main__':
    #     Load model
    loaded_model = None
    try:
        model_path = find_registered_model(name = "Bias Mitigation Malware Detection", uri = f"sqlite:///{relative_model_dev_path}/Malware_Detection_MLFlow.db")
        print(f'{relative_model_dev_path}/{model_path[2:]}/model.pkl')
        with open(f'{relative_model_dev_path}/{model_path[2:]}/model.pkl', "rb") as f:
            loaded_model = pickle.load(f)
    except Exception as e:
        print(traceback.format_exc())
        
#     Get data
    data_aif360 = None
    try:
        df = load_test_dataframe()
        df_ai360 = md.decode_dataset(data_frame=df)
        # train_data_scaled = get_scaled_data(df_ai360)
        train_data_scaled_df = df_ai360.convert_to_dataframe()[0]
        X_train = train_data_scaled_df.drop(settings.Y_COLUMN[0], axis = 1)
        y_train = train_data_scaled_df[settings.Y_COLUMN[0]]
        pass
    except Exception as e:
        print(traceback.format_exc())
        
#         Shapash
    try:
        response_dict = {0: 'No Malware Detected', 1:'Malware Detected'}
        dict1 = {}
        for col in X_train.columns:
            dict1[col] = col
        xpl = SmartExplainer(features_dict=dict1,label_dict=response_dict)
        xpl.compile(x=X_train, model = loaded_model)
        app = xpl.run_app(title_story='Malware Detection Model - XAI')
        pass
    except Exception as e:
        print(traceback.format_exc())