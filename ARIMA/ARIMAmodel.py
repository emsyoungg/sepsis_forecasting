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

    def __init__(self, variable_to_forecast, train_data, test_data, seasonal=False, max_order=8, max_p=5,
                 max_q=5, max_d=2):
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
        self.forecaster = AutoARIMA(seasonal=seasonal, suppress_warnings=True, stepwise=True,
                                    max_order=max_order, max_p=max_p, max_q=max_q, max_d=max_d)
        self.variable_to_forecast = variable_to_forecast

    def fit(self):
        """
        Fits the AutoARIMA model to the variable to forecasts data in the training dataset.
        """
        #sktime_var = self.train_data[[self.variable_to_forecast]]
        #print(sktime_var)
        self.forecaster.fit(self.train_data)

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
        #patient_data_var = patient_data[self.variable_to_forecast]

        plt.figure(figsize=(10, 6))
        plt.plot(patient_data.index, patient_data, label=f"Actual", marker="o")
        plt.plot(patient_predictions.index, patient_predictions, label=f"Forecasted",
                 marker="x", linestyle="--",
                 color="orange")
        plt.xlabel("ICULOS (Time Index)")
        plt.ylabel(f"{self.variable_to_forecast}")
        plt.title(f"Forecast for Last {steps} Hours of Patient {patient_id}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_model(self, forecast):

        ignore_nan = self.test_data[(self.test_data != -1).all(axis=1)]
        filtered_forecast = forecast.loc[ignore_nan.index]


        # Calculate error metrics
        mae = mean_absolute_error(ignore_nan, filtered_forecast)
        mse = mean_squared_error(ignore_nan, filtered_forecast)
        rmse = np.sqrt(mse)

        # Calculate MAPE (Mean Absolute Percentage Error) for accuracy
        mape = np.mean(np.abs((ignore_nan - filtered_forecast) / ignore_nan)) * 100
        accuracy = 100 - mape  # Accuracy derived from MAPE

        # Print results
        print(f"Model Evaluation Results:\n"
              f"  - Mean Absolute Error (MAE): {mae:.4f}\n"
              f"  - Mean Squared Error (MSE): {mse:.4f}\n"
              f"  - Root Mean Squared Error (RMSE): {rmse:.4f}\n"
              f"  - Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n"
              f"  - Forecasting Accuracy: {accuracy:.2f}%")

        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "Accuracy": accuracy}

