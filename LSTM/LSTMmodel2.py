from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.neuralforecast import NeuralForecastLSTM


class LSTMforecaster2:
    def __init__(self, target, exogenous, y_train, y_test=None, X_train=None, X_test=None):
        self.target = target
        self.exogenous = exogenous
        self.y_train = y_train
        self.y_test = y_test
        self.X_train = X_train
        self.X_test = X_test
        self.forecaster = NeuralForecastLSTM(max_steps=10, futr_exog_list=exogenous)  # exogenous predictors

    def fit(self):
        fh = ForecastingHorizon([1, 2, 3], is_relative=True)
        self.forecaster.fit(y=self.y_train, X=self.X_train, fh=fh)

    def predict(self):
        if self.forecaster.is_fitted is False:
            raise ValueError("Model is not fitted. Please call the 'fit' method first.")
        else:
            print("Model is fitted. Generating forecasts...")
        return self.forecaster.predict(self.X_test)

    def fit_predict(self):
        fh = ForecastingHorizon([1, 2, 3], is_relative=True)
        return self.forecaster.fit_predict(y=self.y_train, X=self.X_train, fh=fh, X_pred=self.X_test)
