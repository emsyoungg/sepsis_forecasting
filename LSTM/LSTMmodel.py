import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from sktime.datatypes import check_is_mtype, convert_to
from torchmetrics.functional import mean_absolute_error, mean_squared_error


class LSTMforecaster:
    def __init__(self, train_data, test_data, features):
        self.train_data = train_data
        self.test_data = test_data
        self.features = features if isinstance(features, list) else [features]
        self.forecaster = NeuralForecastLSTM(max_steps=10)

    def fit(self):
        fh = ForecastingHorizon([44, 45, 46], is_relative=True)

        return self.forecaster.fit(self.train_data, fh=fh)

    def predict(self):
        # print(self.forecaster.cutoff())
        # print(self.forecaster.check_is_fitted())
        # print(self.forecaster.get_fitted_params())
        if self.forecaster.is_fitted is False:

            raise ValueError("Model is not fitted. Please call the 'fit' method first.")
        else:
            print("Model is fitted. Generating forecasts...")
        # fh = ForecastingHorizon([44, 45, 46, 47, 48, 49], is_relative=False)
        # print("Internal fh:", self.forecaster._fh)
        # print("Forecaster cutoff:", self.forecaster.cutoff)
        # print("ForecastingHorizon:", fh)
        return self.forecaster.predict()

    def fit_predict(self, x, y, fh):
        """
        Fit the model and then predict.
        """
        return self.forecaster.fit_predict(x, y, fh)

    def plot_forecast(self, forecasts, pid):
        patient_data_combined = pd.concat([self.train_data, self.test_data], axis=0)

        plt.figure(figsize=(12, 6))
        for feature in self.features:
            actual_series = patient_data_combined.loc[pid, feature]
            forecast_series = forecasts.loc[pid, feature]

            plt.plot(actual_series.index, actual_series, label=f"Actual {feature}", linestyle="-", marker="o")
            plt.plot(forecast_series.index, forecast_series, label=f"Predicted {feature}", linestyle="--", marker="x")

        plt.title(f"Patient {pid} — Forecasts (LSTM Model)")
        plt.xlabel("ICULOS")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
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
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Mean Absolute Percentage Error: {mape}")
