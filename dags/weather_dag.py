from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import json
from json import loads, dumps
import os
import time
import pandas as pd
#os.system('pip install scikit-learn')
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
from airflow.utils.task_group import TaskGroup
from airflow.operators.dummy import DummyOperator
import warnings
warnings.filterwarnings('ignore')


# Define the function to fetch API data and save it
def fetch_and_save_api_data(cities, file_path, seconds, nb_times):
    for i in range(0, nb_times):
        current_dateTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #output_path = file_path+"/{}.json".format(current_dateTime).replace(":", "_")
        output_path = file_path+"/{}.json".format(current_dateTime).replace(":", "_")
        copil_weather = []
        for c in cities:
            print(c)
            url = "https://api.openweathermap.org/data/2.5/weather?q={}&appid=KEY_ID".format(c)
            # Replace with your API endpoint
            response = requests.get(url)
            if response.status_code == 200:
                city_weather = json.loads(response.text)
                copil_weather.append(city_weather)
            else:
                raise Exception(f"Failed to fetch data: {response.status_code}, {response.text}")
        #print(copil_weather)
        #os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists
        #with open(f"/opt/airflow/raw_data/terence_apaga_{current_dateTime}.json", "w") as json_file:
        with open(output_path, "w") as json_file:
            json.dump(copil_weather, json_file, indent=4)
        print(f"Data saved to {output_path}")
        time.sleep(seconds)
        
#### Transformer les données
def transform_data_into_csv(n_files=None, filename='data.csv'):
    parent_folder = "/opt/airflow/raw_data"
    files = sorted(os.listdir(parent_folder), reverse=True)
    files = [f for f in files if f[-5:] == '.json']
    files = [f for f in files if f[:4]=='2025']    
    if n_files:
        files = files[:n_files]
    dfs = []
    for f in files:
        print(f)
        with open(os.path.join(parent_folder, f), 'r') as file:
            data_temp = json.load(file)
        for data_city in data_temp:
            dfs.append(
                {
                    'temperature': data_city['main']['temp'],
                    'city': data_city['name'],
                    'pression': data_city['main']['pressure'],
                    'date': f.split('.')[0]
                }
            )
    df = pd.DataFrame(dfs)
    print('\n', df.head(10))
    df.to_csv(os.path.join('/opt/airflow/clean_data', filename), index=False)
    
def compute_model_score(model, task_instance):
    # Pull data from XCom
    X = pd.read_json(task_instance.xcom_pull(task_ids="prepare_api_data_2", key="X_train"))
    y = pd.read_json(task_instance.xcom_pull(task_ids="prepare_api_data_2", key="Y_train"))
    # computing cross val
    cross_validation = cross_val_score(
        model,
        X,
        y,
        cv=3,
        scoring='neg_mean_squared_error')
    model_score = cross_validation.mean()
    task_instance.xcom_push(key=str(model), value=model_score)
    
def compare_perf(task_instance, path_to_model='/opt/airflow/model.pckl'):
    #  Importing models performances
    lreg_perf = task_instance.xcom_pull(task_ids="training_model.train_linear_regression", key='LinearRegression()')
    dreg_perf = task_instance.xcom_pull(task_ids="training_model.train_DecTree_regression", key='DecisionTreeRegressor()')
    rreg_perf = task_instance.xcom_pull(task_ids="training_model.train_RandFores_regression", key='RandomForestRegressor()')
    # Pull data from XCom
    X = pd.read_json(task_instance.xcom_pull(task_ids="prepare_api_data_2", key="X_train"))
    y = pd.read_json(task_instance.xcom_pull(task_ids="prepare_api_data_2", key="Y_train"))
    dict_model = {LinearRegression(): abs(lreg_perf), DecisionTreeRegressor(): abs(dreg_perf), RandomForestRegressor(): abs(rreg_perf)}
    best_model = [key for key, val in dict_model.items() if val == min([abs(v) for k, v in dict_model.items()])]
    # training the model
    best_model[0].fit(X, y)
    # saving model
    dump(best_model[0], path_to_model)
    
def prepare_data(task_instance, path_to_data='/opt/airflow/clean_data/fulldata.csv'): #, **kwargs): #  kwargs to take any parameter as input
    # reading data
    df = pd.read_csv(path_to_data)
    # ordering data according to city and date
    df = df.sort_values(['city', 'date'], ascending=True)
    dfs = []
    for c in df['city'].unique():
        df_temp = df[df['city'] == c]
        # creating target
        df_temp.loc[:, 'target'] = df_temp['temperature'].shift(1)
        # creating features
        for i in range(1, 10):
            df_temp.loc[:, 'temp_m-{}'.format(i)
                        ] = df_temp['temperature'].shift(-i)
        # deleting null values
        df_temp = df_temp.dropna()
        dfs.append(df_temp)
    # concatenating datasets
    df_final = pd.concat(
        dfs,
        axis=0,
        ignore_index=False
    )
    # deleting date variable
    df_final = df_final.drop(['date'], axis=1)
    # creating dummies for city variable
    df_final = pd.get_dummies(df_final)
    features = df_final.drop(['target'], axis=1)
    target = df_final['target']    
    features_json = features.to_json(orient="records")
    target_json = target.to_json(orient="records")
    task_instance.xcom_push(key="X_train", value=features_json)
    task_instance.xcom_push(key="Y_train", value=target_json)

#===================== DAGS ==========================

# Define default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 2, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=0.2),
}
# Define DAG
with DAG(
    "weather_forecast",
    default_args=default_args,
    schedule_interval="@daily",  # Set your desired schedule
    catchup=False,
) as dag:
    #output_file = "/Users/terence/A_NOTEBOOKS/Datasciencetest/AIRFLOW/evaluation_meteo/app/raw_files"
    output_file = "/opt/airflow/raw_data"
    cities = ['paris', 'london', 'washington']
    # Define the task
    fetch_api_data_task_1 = PythonOperator(
        task_id="fetch_api_data",
        python_callable=fetch_and_save_api_data,
        op_kwargs={'cities': cities, 'file_path': output_file, 'seconds': 2, 'nb_times':20},
        dag=dag,
    )
    prepare_modeldata_task_4 = PythonOperator(
        task_id="prepare_api_data_2",
        python_callable=prepare_data,
        provide_context=True,
        op_kwargs={'path_to_data': '/opt/airflow/clean_data/fulldata.csv'},
        dag=dag,
        )   
    compare_model_task = PythonOperator(
        task_id="comparing_regression",
        python_callable=compare_perf,
        provide_context=True,
        dag=dag,
        )
    with TaskGroup('transforming_data') as group_A:
        transform_data_task_2 = PythonOperator(
            task_id="transform_api_data_1",
            python_callable=transform_data_into_csv,
            op_kwargs={'n_files': 20, 'filename': 'data.csv'},
            dag=dag,
            )
        transform_data_task_3 = PythonOperator(
            task_id="transform_api_data_2",
            python_callable=transform_data_into_csv,
            op_kwargs={'n_files': None, 'filename': 'fulldata.csv'},
            dag=dag,
            )
    with TaskGroup('training_model') as group_B:
        train_model_task_41 = PythonOperator(
            task_id="train_linear_regression",
            python_callable=compute_model_score,
            op_kwargs={'model': LinearRegression()},
            provide_context=True,
            dag=dag,
            )
        train_model_task_42 = PythonOperator(
            task_id="train_DecTree_regression",
            python_callable=compute_model_score,
            op_kwargs={'model': DecisionTreeRegressor()},
            provide_context=True,
            dag=dag,
            )
        train_model_task_43 = PythonOperator(
            task_id="train_RandFores_regression",
            python_callable=compute_model_score,
            op_kwargs={'model': RandomForestRegressor()},
            provide_context=True,
            dag=dag,
            )
    # Set task dependencies (if needed)
    fetch_api_data_task_1 >> transform_data_task_2
    fetch_api_data_task_1 >> transform_data_task_3
    transform_data_task_3 >> prepare_modeldata_task_4
    prepare_modeldata_task_4 >> train_model_task_41 >> compare_model_task
    prepare_modeldata_task_4 >> train_model_task_42 >> compare_model_task
    prepare_modeldata_task_4 >> train_model_task_43 >> compare_model_task
    
