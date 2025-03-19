import os
import glob
import pandas as pd
from sktime.datatypes import convert_to


class PatientTimeSeriesLoader:
    """
    A class to load multiple patient time series data from CSV files
    into a sktime-compatible multi-index DataFrame. Each patient has a df.
    """

    def __init__(self, folder):
        self.folder = folder
        self.file_list = self._get_sorted_file_list()

    def _get_sorted_file_list(self):
        file_list = glob.glob(os.path.join(self.folder, "*.csv"))
        return sorted(file_list, key=lambda x: int(os.path.basename(x).split(".")[0]))

    def load_data(self):
        """
        Loads multiple CSV files into a sktime-compatible multi-index DataFrame.

        Returns:
            pd.DataFrame: sktime-compatible multi-index dataframe (Patient_ID, ICULOS as time index).
        """
        df_list = []

        for i, file in enumerate(self.file_list):
            df = pd.read_csv(file)

            if 'ICULOS' not in df.columns:
                raise ValueError(f"File {file} is missing the 'ICULOS' column (time index).")

            df["Patient_ID"] = i
            # change missing data fill method here
            df = df.ffill().bfill()

            df_list.append(df)

        full_df = pd.concat(df_list, ignore_index=True)
        full_df.set_index(["Patient_ID", "ICULOS"], inplace=True)
        return convert_to(full_df, to_type="pd-multiindex")

    def split_train_test(self, data):
        """
        Splits the data into training and testing sets, where the test set is the last 6 hours
        of each patient's data.

        Args:
            data (pd.DataFrame): The sktime-compatible multi-index DataFrame.

        Returns:
            tuple: (train_data, test_data) as separate DataFrames.
        """
        train_list = []
        test_list = []

        for patient_id in data.index.get_level_values("Patient_ID").unique():
            patient_df = data.loc[patient_id]
            split_point = patient_df.index.max() - 6
            train_df = patient_df.loc[:split_point]
            test_df = patient_df.loc[split_point + 1:]

            train_list.append(train_df)
            test_list.append(test_df)

        train_data = pd.concat(train_list, keys=data.index.get_level_values("Patient_ID").unique())
        test_data = pd.concat(test_list, keys=data.index.get_level_values("Patient_ID").unique())

        return train_data, test_data

    def subset_data(self, train_data, test_data, max_patient_id):
        """
        Restricts training and testing data to a subset of patients so can run quicker

        Args:
            max_patient_id (int): Maximum patient ID to include in the subset.
        """
        train_data_subset = train_data.loc[train_data.index.get_level_values("Patient_ID") < max_patient_id]
        test_data_subset = test_data.loc[test_data.index.get_level_values("Patient_ID") < max_patient_id]
        print(train_data_subset.index)
        print(test_data_subset.index)

        return train_data_subset, test_data_subset
