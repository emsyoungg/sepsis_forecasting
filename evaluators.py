import pandas as pd
from matplotlib import pyplot as plt
from sktime.distances import dtw_distance
from sktime.distances import ddtw_distance
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class Evaluator:

    def __init__(self, y_train, y_pred, forecasts):
        self.y_train = y_train
        self.y_pred = y_pred
        self.forecasts = forecasts

    def sktime_dtw(self):
        feature_dtw_dict = {}

        for feature in self.y_pred.columns:
            dtw_list = []
            patient_ids = []

            for pid in self.y_pred.index.get_level_values("Patient_ID").unique():
                timeseries = self.y_pred.loc[pid, feature]
                forecasted_timeseries = self.forecasts.loc[pid, feature]

                x = np.array(timeseries).reshape(-1, 1)
                y = np.array(forecasted_timeseries).reshape(-1, 1)

                dtw_val = dtw_distance(x, y)
                dtw_list.append(dtw_val)
                patient_ids.append(pid)

            feature_dtw_dict[feature] = {
                "dtw": dtw_list,
                "patient_ids": patient_ids
            }

            print(f"Median sktime DTW distance for feature {feature}: {np.median(dtw_list)}")

        return feature_dtw_dict

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


    def box_plot_dtw_interactive(self, feature_dtw_dict):
        num_features = len(feature_dtw_dict)
        fig = make_subplots(
            rows=1,
            cols=num_features,
            subplot_titles=list(feature_dtw_dict.keys())
        )

        for i, (feature, data) in enumerate(feature_dtw_dict.items(), start=1):
            dtw_vals = data["dtw"]
            patient_ids = data["patient_ids"]

            fig.add_trace(
                go.Box(
                    y=dtw_vals,
                    text=[f"Patient ID: {pid}" for pid in patient_ids],
                    hovertemplate="%{text}<br>DTW: %{y:.3f}",
                    name=feature,
                    boxpoints="all",  # shows all the points
                    jitter=0.5,
                    pointpos=-1.8
                ),
                row=1, col=i
            )

        fig.update_layout(
            height=600,
            width=300 * num_features,
            title_text="DTW Distance Boxplots with Patient IDs",
            showlegend=False
        )

        fig.show()

    def box_plot_dtw(self, feature_dtw_dict):
        num_features = len(feature_dtw_dict)
        fig, axs = plt.subplots(1, num_features, figsize=(5 * num_features, 6), constrained_layout=True)

        if num_features == 1:
            axs = [axs]

        for ax, (feature, data) in zip(axs, feature_dtw_dict.items()):
            dtw_list = data["dtw"]
            ax.boxplot(dtw_list)
            ax.set_title(f"Boxplot of DTW distances\nfor {feature}")
            ax.set_ylabel("DTW Distance")
            ax.set_xlabel("Patients")
            ax.grid(True)

        plt.show()



