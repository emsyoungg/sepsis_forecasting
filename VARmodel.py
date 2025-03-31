import pandas as pd
import matplotlib.pyplot as plt
from sktime.forecasting.var import VAR
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import mean_absolute_error


class VARForecaster:

    def __init__(self, train_data, test_data, features):
        self.train_data = train_data
        self.test_data = test_data
        self.features = features if isinstance(features, list) else [features]
        self.forecaster = VAR()

    def fit(self):
        self.forecaster.fit(self.train_data)

    def predict(self, steps=6):
        fh = ForecastingHorizon(range(1, steps + 1), is_relative=True)
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

    def evaluate_model(self, predicted_values):
        mae = mean_absolute_error(self.test_data, predicted_values)
        print(f"Mean Absolute Error: {mae}")
