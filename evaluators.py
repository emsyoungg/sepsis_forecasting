import pandas as pd
from matplotlib import pyplot as plt
from sktime.distances import dtw_distance
from sktime.distances import ddtw_distance
from tslearn.metrics import dtw
import numpy as np

class Evaluator:

    def __init__(self, y_train, y_pred, forecasts):
        self.y_train = y_train
        self.y_pred = y_pred
        self.forecasts = forecasts

    def sktime_dtw(self):
        for feature in self.y_pred.columns:
            dtw_list = []
            for pid in self.y_pred.index.get_level_values("Patient_ID").unique():
                timeseries = self.y_pred.loc[pid, feature]
                forecasted_timeseries = self.forecasts.loc[pid, feature]

                x = np.array(timeseries)
                y = np.array(forecasted_timeseries)

                dtw_val = dtw_distance(x, y)
                dtw_list.append(dtw_val)

            print(f"Mean sktime DTW distance for feature {feature}: {np.mean(dtw_list)}")

    def sktime_ddtw(self):
        for feature in self.y_pred.columns:
            dtw_list = []
            for pid in self.y_pred.index.get_level_values("Patient_ID").unique():
                timeseries = self.y_pred.loc[pid, feature]
                forecasted_timeseries = self.forecasts.loc[pid, feature]

                x = np.array(timeseries)
                y = np.array(forecasted_timeseries)

                dtw_val = ddtw_distance(x, y)
                dtw_list.append(dtw_val)

            print(f"Mean sktime DDTW distance for feature {feature}: {np.mean(dtw_list)}")

    def tslearn_dtw(self):
        distances = []

        for pid in self.y_pred.index.get_level_values("Patient_ID").unique():
            true = self.y_pred.loc[pid]
            pred = self.forecasts.loc[pid]
            dist = dtw(true, pred)
            distances.append((pid, dist))

        print("Mean tslearn DTW distance:", np.mean([d[1] for d in distances]))

    def plot_multivar(self, num_graphs=4):
        patient_data_combined = pd.concat([self.y_train, self.y_pred], axis=0)

        for pid in self.y_pred.index.get_level_values("Patient_ID").unique():
            if pid > num_graphs:
                break

            plt.figure(figsize=(12, 6))
            for feature in self.y_pred.columns:
                actual_series = patient_data_combined.loc[pid, feature]
                forecast_series = self.forecasts.loc[pid, feature]

                plt.plot(actual_series.index, actual_series, label=f"Actual {feature}", linestyle="-", marker="o")
                plt.plot(forecast_series.index, forecast_series, label=f"Predicted {feature}", linestyle="--", marker="x")

            plt.title(f"Patient {pid} — Forecasts (VAR Model)")
            plt.xlabel("ICULOS")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

