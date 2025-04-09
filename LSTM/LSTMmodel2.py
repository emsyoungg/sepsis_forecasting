import pandas as pd
from matplotlib import pyplot as plt
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.neuralforecast import NeuralForecastLSTM


class LSTMforecaster2:
    def __init__(self, target, y_train,  exogenous=None, y_test=None, X_train=None, X_test=None):
        self.target = target
        self.exogenous = exogenous
        self.y_train = y_train
        self.y_test = y_test
        self.X_train = X_train
        self.X_test = X_test
        self.forecaster = NeuralForecastLSTM(max_steps=10, futr_exog_list=exogenous)  # exogenous predictors

    def fit(self):
        fh = [1, 2, 3, 4, 5, 6]  # Forecasting horizon
        self.forecaster.fit(y=self.y_train, X=self.X_train, fh=fh)

    def predict(self):
        if self.forecaster.is_fitted is False:
            raise ValueError("Model is not fitted. Please call the 'fit' method first.")
        else:
            print("Model is fitted. Generating forecasts...")
        return self.forecaster.predict(X=self.X_test)

    def fit_predict(self):
        #fh = ForecastingHorizon([1, 2, 3, 4, 5, 6])
        return self.forecaster.fit_predict(y=self.y_train, X=self.X_train, X_pred=self.X_test)

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
        patient_data_combined = pd.concat([self.y_train, self.y_test], axis=0)
        patient_data = patient_data_combined.loc[patient_id]

        plt.figure(figsize=(10, 6))
        plt.plot(patient_data.index, patient_data, label=f"Actual", marker="o")
        plt.plot(patient_predictions.index, patient_predictions, label=f"Forecasted",
                 marker="x", linestyle="--",
                 color="orange")
        plt.xlabel("ICULOS (Time Index)")
        plt.ylabel(f"{self.target} Value")
        plt.title(f"Forecast for Last {steps} Hours of Patient {patient_id}")
        plt.legend()
        plt.grid(True)
        plt.show()
