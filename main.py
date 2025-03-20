from statsmodels.tools.sm_exceptions import ValueWarning
import loadDataForSKtime
import ARIMAmodel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ValueWarning)


if __name__ == "__main__":
    loader = loadDataForSKtime.PatientTimeSeriesLoader(
        "C:/Users/emily/Documents/DissertationProject/training/training_setA_csv")
    sktime_df = loader.load_data()
    train_data, test_data = loader.split_train_test(sktime_df)
    train_data_sub, test_data_sub = loader.subset_data(train_data, test_data, max_patient_id=100)

if __name__ == "__main__":
    HRforecaster = ARIMAmodel.ARIMAForecaster('HR', train_data_sub, test_data_sub)
    HRforecaster.fit()
    forecasts = HRforecaster.predict(steps=6)
    for patient_id in range(1, 10):
        HRforecaster.plot_forecast(forecasts, patient_id=patient_id, steps=6)

if __name__ == "__main__":
    HRforecaster.evaluate_model(forecasts)

