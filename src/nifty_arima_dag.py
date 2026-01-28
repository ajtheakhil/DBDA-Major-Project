from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ds_ubuntu',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 27),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id="nifty_arima_pipeline",
    default_args=default_args,
    schedule_interval="@once",
    catchup=False
) as dag:

    # TASK 1: The Dataset retrieval Script
    run_nifty_retrieval = BashOperator(
        task_id="generate_nifty_dataset",
        # Point this to your second script
        bash_command="/home/ds_ubuntu/ds_ma_proj/bin/python /media/sf_ubuntu_project_folder/pyspark_HIVE_new.py"

    )
        # Task 2: to run the ARIMA model with backtesting
    run_arima = BashOperator(
        task_id="run_arima_with_accuracy",
        # We use the full path to the project-specific python interpreter
        bash_command="/home/ds_ubuntu/ds_ma_proj/bin/python /media/sf_ubuntu_project_folder/arima_with_accuracy.py"
    )

    # SET THE ORDER (Dependencies)
    # This tells Airflow: Run ARIMA first, then run Summary.
    run_nifty_retrieval >> run_arima

