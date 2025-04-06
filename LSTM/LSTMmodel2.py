import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from sktime.datatypes import check_is_mtype, convert_to
from torchmetrics.functional import mean_absolute_error, mean_squared_error


class LSTMforecaster2:
    def __init__(self, target_df_train, target_df_test, exogenous_df_train, exogenous_df_test, features):
        self.target_df_train = target_df_train
        self.target_df_test = target_df_test
        self.exogenous_df_train = exogenous_df_train
        self.exogenous_df_test = exogenous_df_test
        self.features = features if isinstance(features, list) else [features]
        self.forecaster = NeuralForecastLSTM(max_steps=10, futr_exog_list=features)  # exogenous predictors

    def fit(self):
        fh = ForecastingHorizon([1, 2, 3], is_relative=True)
        self.forecaster.fit(y=self.target_df_train, X=self.exogenous_df_train, fh=fh)

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