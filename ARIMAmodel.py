from pandas import isnull
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


class ARIMAForecaster:
    """
    A class for forecasting a variable in the patients data using AutoARIMA on patient time series data.
    """

    def __init__(self, variable_to_forecast, train_data, test_data, sp=12, seasonal=False, max_order=5, max_p=3,
                 max_q=3, max_d=1):
        """
        Initializes the forecaster with AutoARIMA model.

        Args:
            variable_to_forecast: The variable to forecast.
            train_data (pd.DataFrame): Training data in sktime-compatible format.
            test_data (pd.DataFrame): Test data in sktime-compatible format.
            sp (int): Seasonal period.
            seasonal (bool): Whether to consider seasonality.
            max_order (int): Maximum order of ARIMA model.
            max_p (int): Maximum autoregressive order.
            max_q (int): Maximum moving average order.
            max_d (int): Maximum differencing order.
        """
        self.train_data = train_data
        self.test_data = test_data
        self.forecaster = AutoARIMA(sp=sp, seasonal=seasonal, suppress_warnings=True, stepwise=True,
                                    max_order=max_order, max_p=max_p, max_q=max_q, max_d=max_d)
        self.variable_to_forecast = variable_to_forecast

    def fit(self):
        """
        Fits the AutoARIMA model to the variable to forecasts data in the training dataset.
        """
        sktime_var = self.train_data[[self.variable_to_forecast]]
        print(sktime_var)
        self.forecaster.fit(sktime_var)

    def predict(self, steps=6):
        """
        Predicts var for the next specified steps.

        Args:
            steps (int): Number of time steps to forecast.

        Returns:
            pd.Series: Forecasted values.
        """
        forecast_horizon = ForecastingHorizon(range(1, steps + 1), is_relative=True)
        return self.forecaster.predict(forecast_horizon)

    def plot_forecast(self, forecasts, patient_id, steps=6):
        """
        Plots the forecasted data alongside the actual data for a specific patient.

        Args:
            patient_id (int): Patient ID for visualisation.
            steps (int): Number of forecast steps.
            :param forecasts: forcast data
        """

        # patient to visualise
        patient_predictions = forecasts.loc[patient_id]
        patient_data_combined = pd.concat([self.train_data, self.test_data], axis=0)
        patient_data = patient_data_combined.loc[patient_id]
        patient_data_var = patient_data[self.variable_to_forecast]

        plt.figure(figsize=(10, 6))
        plt.plot(patient_data.index, patient_data_var, label=f"Actual {self.variable_to_forecast}", marker="o")
        plt.plot(patient_predictions.index, patient_predictions, label=f"Forecasted {self.variable_to_forecast}",
                 marker="x", linestyle="--",
                 color="orange")
        plt.xlabel("ICULOS (Time Index)")
        plt.ylabel(f"{self.variable_to_forecast}")
        plt.title(f"Forecast for Last {steps} Hours of Patient {patient_id}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_model(self, predicted_values):
        """
        Evaluates forecast performance using MAE, MSE, RMSE, MAPE, and Accuracy.
        Assumes predicted_values and self.test_data have the same MultiIndex.
        """
        # Extract actual values for the variable to forecast
        try:
            actual_values = self.test_data[self.variable_to_forecast]
        except KeyError:
            raise KeyError(
                f"'{self.variable_to_forecast}' not found in test data columns: {self.test_data.columns.tolist()}")

        # Align indices (should be the same, but just in case)
        actual_values = actual_values.sort_index()
        predicted_values = predicted_values.sort_index()

        # Filter to common index and drop NaNs
        common_index = actual_values.index.intersection(predicted_values.index)
        actual_values = actual_values.loc[common_index]
        predicted_values = predicted_values.loc[common_index]

        # Remove any remaining NaNs
        mask = actual_values.notna() & predicted_values.notna()
        actual_values = actual_values[mask]
        predicted_values = predicted_values[mask]

        if actual_values.empty:
            print("⚠️ No valid data points to evaluate.")
            return None

        # Evaluation metrics
        mae = mean_absolute_error(actual_values, predicted_values)
        mse = mean_squared_error(actual_values, predicted_values)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
        accuracy = 100 - mape

        # Output
        print(f"✅ Model Evaluation Results:")
        print(f"  - MAE:     {mae:.4f}")
        print(f"  - MSE:     {mse:.4f}")
        print(f"  - RMSE:    {rmse:.4f}")
        print(f"  - MAPE:    {mape:.2f}%")
        print(f"  - Accuracy:{accuracy:.2f}%")

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "Accuracy": accuracy
        }

