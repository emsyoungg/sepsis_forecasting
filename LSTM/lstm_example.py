from sktime.split import temporal_train_test_split
from sktime.utils._testing.panel import _make_panel
from sktime.forecasting.neuralforecast import NeuralForecastLSTM

# same shape as my sepsis df
df = _make_panel(n_instances=5, n_columns=6, n_timepoints=30, all_positive=True)

# split into endogenous and exogenous dataframe
y = df.drop(['var_1', 'var_2', 'var_3', 'var_4', 'var_5'], axis=1)
X = df.drop('var_0', axis=1)

y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=4)
model = NeuralForecastLSTM(futr_exog_list=['var_1', 'var_2', 'var_3', 'var_4', 'var_5'], max_steps=5)

model.fit(y_train, X=X_train, fh=[1,2,3,4])

model.predict(X=X_test)