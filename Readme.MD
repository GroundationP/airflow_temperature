## Airflow Temperature Forecast
This pipeline is an Airflow application designed to collect, transform, and predict city temperature data via an API. Its architecture is based on a Docker image of Airflow, integrated with Scikit-learn, and follows the DAG outlined.

The process begins by retrieving real-time weather data from OpenWeatherMap for a wide selection of cities worldwide. In this solution, we use GET requests to collect temperature data for Paris, London, and Washington, storing it in a Docker container in JSON format.

The next step is to extract these JSON files and consolidate them into a single CSV file, transforming them into a dataset for training and testing. For the modeling stage, we illustrate the predictive capabilities of both linear and non-linear models without using hyperparameter tuning. Based on the lowest Mean Squared Error (MSE), the pipeline selects the best-performing model, generates the desired predictions, and exports the most effective model.

#### Setup from command line
mkdir airflow_tempw

cd airflow_tempw

mkdir ./dags ./logs ./plugins

sudo chmod -R 777 logs/

sudo chmod -R 777 dags/

sudo chmod -R 777 plugins/

echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env

#### Docker setup
docker-compose up airflow-init

docker-compose up -d

http://localhost:8080/home

#### To identify docker worker (find Container ID)
docker ps

#### to open terminal in docker worker
docker exec -it Container_ID /bin/bash (example)

docker exec -it 4255b3ce0595 /bin/bash

#### create folders to store files
mkdir raw_data

mkdir clean_data
