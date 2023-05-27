import pandas as pd
import argparse
from src.utils.common_utils import (
    read_params, 
    save_reports,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def eval_metrics(actual, predicted):
    rmse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2

def evaluate(config_path):
    config = read_params(config_path)

    artifacts = config["artifacts"]
    test_data_path = artifacts["split_data"]["test_path"]
    model_path = artifacts["model_path"]
    target = config["base"]["target_col"]
    scores_file = artifacts["reports"]["scores"]

    test = pd.read_csv(test_data_path, sep=',')

    test_x = test.drop(target, axis = 1)
    test_y = test[target]

    lr = joblib.load(model_path)

    predicted_values = lr.predict(test_x)

    rmse, mae, r2 = eval_metrics(test_y, predicted_values)

    scores = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

    save_reports(scores_file, scores)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        data = evaluate(config_path=parsed_args.config)
    except Exception as e:
        raise e