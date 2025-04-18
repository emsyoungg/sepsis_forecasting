import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sktime.forecasting.var import VAR
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error
import numpy as np
from sktime.distances import ddtw_distance

class VARForecaster:

    def __init__(self, train_data, test_data, features):
        self.train_data = train_data
        self.test_data = test_data
        self.features = features if isinstance(features, list) else [features]
        self.forecaster = VAR()

    def fit(self):
        self.forecaster.fit(self.train_data)

    def predict(self, steps=4):
        fh = ForecastingHorizon([1, 2, 3, 4], is_relative=True)
        return self.forecaster.predict(fh)

    def plot_forecast(self, forecasts, pid):
        patient_data_combined = pd.concat([self.train_data, self.test_data], axis=0)

        plt.figure(figsize=(12, 6))
        for feature in self.features:
            actual_series = patient_data_combined.loc[pid, feature]
            forecast_series = forecasts.loc[pid, feature]

            plt.plot(actual_series.index, actual_series, label=f"Actual {feature}", linestyle="-", marker="o")
            plt.plot(forecast_series.index, forecast_series, label=f"Predicted {feature}", linestyle="--", marker="x")

        plt.title(f"Patient {pid} — Forecasts (VAR Model)")
        plt.xlabel("ICULOS")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, forecast, y_train):

        ignore_nan = y_train[(y_train != -1).all(axis=1)]
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

    def dynamic_time_warping(self, forecast):
        return ddtw_distance(self.test_data, forecast)
