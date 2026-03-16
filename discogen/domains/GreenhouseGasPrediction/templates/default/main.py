"""Evaluation script for the GreenhouseGasPrediction task."""

import pathlib
import json

import numpy as np
from data_processing import process_data
from model import make_model

if __name__ == "__main__":
    # Load train data
    script_dir = pathlib.Path(__file__).parent.resolve()

    train_data_file = script_dir / "data" / "train_data.npy"
    if not train_data_file.exists():
        raise FileNotFoundError(f"Train data not found at {train_data_file}")
    train_data = np.load(train_data_file)

    train_data_p = process_data(train_data)

    model = make_model(train_data_p)

    model = model.fit(train_data_p[:, :-2], train_data_p[:, -1])

    predictions = model.predict(train_data_p[:, :-2])

    assert predictions.shape == train_data_p[:, -1].shape

    mse = np.mean((predictions - train_data_p[:, -1]) ** 2)
    print(f"Train MSE: {train_mse}")

    test_data_file = script_dir / "data" / "test_data.npy"

    if not test_data_file.exists():
        raise FileNotFoundError(f"Train data not found at {train_data_file}")
    test_data = np.load(test_data_file)

    test_data_p = process_data(test_data)

    predictions = model.predict(test_data_p[:, :-2])

    assert predictions.shape == test_data_p[:, -1].shape

    test_mse = np.mean((predictions - test_data_p[:, -1]) ** 2)
    print(json.dumps({"Test MSE": test_mse, "Train MSE": train_mse}))
