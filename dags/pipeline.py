from datetime import datetime

from airflow.decorators import dag, task

from deploy import deploy
from ml import Learner
from etl import preprocess


# DAG
@dag(
    schedule_interval=None,
    start_date=datetime(2022, 1, 1),
    catchup=False,
    tags=['pipeline'],
)
def pipeline():
    # Task: Data preprocess
    @task
    def task_data_preprocess():
        path_preprocessed_data = preprocess()
        return path_preprocessed_data

    # Task: Hyper-parameter tuning
    @task()
    def task_hyper_parameter_tuning(path_data: dict):
        ml = Learner()
        optimal_params = ml.hyper_parameter_tuning(path_data)
        return optimal_params

    # Task: Training
    @task()
    def task_train(optimal_params: dict, path_data: dict):
        ml = Learner()
        path_trained_model = ml.train(optimal_params, path_data)
        return path_trained_model

    # Task: Testing
    @task()
    def task_test(path_trained_model: str, path_data: dict, optimal_params: dict):
        ml = Learner()
        path_saved_model = ml.inference(path_trained_model, path_data, optimal_params)
        return path_saved_model

    # Task: Deploy
    @task()
    def task_deploy(path_saved_model: str):
        path_deployment_files = deploy(path_saved_model)
        return path_deployment_files

    path_preprocessed_data = task_data_preprocess()
    optimal_params = task_hyper_parameter_tuning(path_preprocessed_data)
    path_trained_model = task_train(optimal_params, path_preprocessed_data)
    path_saved_model = task_test(
        path_trained_model,
        path_preprocessed_data,
        optimal_params,
    )
    path_deployment_files = task_deploy(path_saved_model)


airflow_dag = pipeline()
