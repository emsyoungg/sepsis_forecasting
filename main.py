import loadDataForSKtime
import ARIMAmodel

if __name__ == "__main__":
    loader = loadDataForSKtime.PatientTimeSeriesLoader(
        "C:/Users/emily/Documents/DissertationProject/training/training_setA_csv")
    sktime_df = loader.load_data()
    train_data, test_data = loader.split_train_test(sktime_df)
    train_data_sub, test_data_sub = loader.subset_data(train_data, test_data, max_patient_id=100)

    HRforecaster = ARIMAmodel.ARIMAForecaster('HR', train_data_sub, test_data_sub)
    HRforecaster.fit()
    forecasts = HRforecaster.predict(steps=6)
    HRforecaster.plot_forecast(forecasts, patient_id=8, steps=6)
