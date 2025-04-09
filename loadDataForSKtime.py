import os
import glob
import pandas as pd
from sktime.datatypes import convert_to
from sktime.datatypes import check_is_mtype

from sktime.transformations.series.impute import Imputer


class PatientTimeSeriesLoader:
    """
    A class to load multiple patient time series data from CSV files
    into a sktime-compatible multi-index DataFrame. Each patient has a df.
    """

    def __init__(self, folder, column):
        self.folder = folder
        self.file_list = self._get_sorted_file_list()
        self.column = column if isinstance(column, list) else [column]

    def _get_sorted_file_list(self):
        file_list = glob.glob(os.path.join(self.folder, "*.csv"))
        return sorted(file_list, key=lambda x: int(os.path.basename(x).split(".")[0]))

    def load_data(self, amount=None):
        """
        Loads multiple CSV files into a sktime-compatible multi-index DataFrame.

        Returns:
            pd.DataFrame: sktime-compatible multi-index dataframe (Patient_ID, ICULOS as time index).
        """
        df_list = []

        print(type(self.column))

        if amount:
            self.file_list = self.file_list[:amount]

        for i, file in enumerate(self.file_list):
            df = pd.read_csv(file)
            df["Patient_ID"] = i + 1

            if 'ICULOS' not in df.columns:
                raise ValueError(f"File {file} is missing the 'ICULOS' column (time index).")

            if not all(col in df.columns for col in self.column):
                print(f"Patient {i + 1}: Missing one of {self.column} — skipping.")
                continue

            # Drop if all values in target columns are NaN
            if df[self.column].isna().all().all():
                print(f"Patient {i + 1}: All {self.column} values are NaN — skipping.")
                continue

            if df["ICULOS"].nunique() < 46:
                print(f"Patient {i + 1}: Less than 46 time points — skipping.")
                continue

            df["ICULOS"] = pd.RangeIndex(start=0, stop=(len(df)), step=1)

            df = df.ffill().bfill()

            if df[self.column].nunique().le(2).any():
                print(f"Patient {i + 1}: Dropping — constant columns found")
                continue

            df_list.append(df[self.column + ["Patient_ID", "ICULOS"]])

        full_df = pd.concat(df_list, ignore_index=True)
        full_df.set_index(["Patient_ID", "ICULOS"], inplace=True)
        # Check whether it's compatible with sktime's "pd-multiindex" format
        check_is_mtype(full_df, mtype="pd-multiindex", scitype="Panel")  # will raise if invalid

        # df_multiindex = convert_to(full_df, to_type="pd-multiindex")
        # check_is_mtype(df_multiindex, mtype="pd-multiindex", scitype="Panel")
        return full_df

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
            train_df = patient_df.loc[:split_point].copy()
            test_df = patient_df.loc[split_point + 1:].copy()

            if train_df[self.column].nunique().le(2).any():
                print(f"Patient {patient_id}: Dropping — constant columns found")
                continue

            train_df["Patient_ID"] = patient_id
            train_df["ICULOS"] = train_df.index
            test_df["Patient_ID"] = patient_id
            test_df["ICULOS"] = test_df.index

            train_list.append(train_df[self.column + ["Patient_ID", "ICULOS"]])
            test_list.append(test_df[self.column + ["Patient_ID", "ICULOS"]])

        train_data = pd.concat(train_list, ignore_index=True)
        test_data = pd.concat(test_list, ignore_index=True)
        train_data.set_index(["Patient_ID", "ICULOS"], inplace=True)
        test_data.set_index(["Patient_ID", "ICULOS"], inplace=True)

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

    def load_data_LSTM(self, amount=None):
        """
        Loads multiple CSV files into a sktime-compatible multi-index DataFrame.

        Returns:
            pd.DataFrame: sktime-compatible multi-index dataframe (Patient_ID, ICULOS as time index).
        """
        df_list = []
        new_patient_id = 0

        print(type(self.column))

        if amount:
            self.file_list = self.file_list[:amount]

        for i, file in enumerate(self.file_list):
            df = pd.read_csv(file)

            if 'ICULOS' not in df.columns:
                raise ValueError(f"File {file} is missing the 'ICULOS' column (time index).")

            if not all(col in df.columns for col in self.column):
                print(f"Patient {i + 1}: Missing one of {self.column} — skipping.")
                continue

            # Drop if all values in target columns are NaN
            if df[self.column].isna().all().all():
                print(f"Patient {i + 1}: All {self.column} values are NaN — skipping.")
                continue

            if df["ICULOS"].nunique() < 30:
                print(f"Patient {i + 1}: Less than 30 time points — skipping.")
                continue

            df["ICULOS"] = pd.RangeIndex(start=0, stop=(len(df)), step=1)

            df = df.ffill().bfill()

            if df[self.column].nunique().le(2).any():
                print(f"Patient {i + 1}: Dropping — constant columns found")
                continue

            # make all time series the same length
            if df["ICULOS"].nunique() > 30:
                df = df[:30]

            df["Patient_ID"] = new_patient_id
            new_patient_id += 1

            df_list.append(df[self.column + ["Patient_ID", "ICULOS"]])

        full_df = pd.concat(df_list, ignore_index=True)
        full_df.set_index(["Patient_ID", "ICULOS"], inplace=True)
        # Check whether it's compatible with sktime's "pd-multiindex" format
        check_is_mtype(full_df, mtype="pd-multiindex", scitype="Panel")  # will raise if invalid

        # df_multiindex = convert_to(full_df, to_type="pd-multiindex")
        # check_is_mtype(df_multiindex, mtype="pd-multiindex", scitype="Panel")
        return full_df

    def split_train_test_LSTM(self, data):
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
        new_patient_id = 0

        for patient_id in data.index.get_level_values("Patient_ID").unique():
            patient_df = data.loc[patient_id]

            split_point = patient_df.index.max() - 6
            train_df = patient_df.loc[:split_point].copy()
            test_df = patient_df.loc[split_point + 1:].copy()

            if train_df[self.column].nunique().le(2).any():
                print(f"Patient {patient_id}: Dropping — constant columns found")
                continue

            train_df["Patient_ID"] = new_patient_id
            train_df["ICULOS"] = train_df.index
            test_df["Patient_ID"] = new_patient_id
            test_df["ICULOS"] = test_df.index
            new_patient_id += 1

            train_list.append(train_df[self.column + ["Patient_ID", "ICULOS"]])
            test_list.append(test_df[self.column + ["Patient_ID", "ICULOS"]])

        train_data = pd.concat(train_list, ignore_index=True)
        test_data = pd.concat(test_list, ignore_index=True)
        train_data.set_index(["Patient_ID", "ICULOS"], inplace=True)
        test_data.set_index(["Patient_ID", "ICULOS"], inplace=True)

        return train_data, test_data

    def properly_split(self, df, target, exogenous):

        y_train_list = []
        X_train_list = []
        y_test_list = []
        X_test_list = []
        new_patient_id = 0

        for patient_id in df.index.get_level_values("Patient_ID").unique():
            patient_df = df.loc[patient_id]

            # Split the data into training and testing sets
            split_point = patient_df.index.max() - 6
            train_df = patient_df.loc[:split_point].copy()
            test_df = patient_df.loc[split_point + 1:].copy()

            # Check if the split is valid
            if train_df.empty or test_df.empty:
                print(f"Patient {patient_id}: Invalid split — skipping.")
                continue

            if train_df[target].nunique().le(2).any():
                print(f"Patient {patient_id}: Dropping — constant columns found")
                continue

            if train_df[exogenous].nunique().le(2).any():
                print(f"Patient {patient_id}: Dropping — constant columns found")
                continue

            train_target_df = train_df[target]
            test_target_df = test_df[target]
            train_exogenous_df = train_df[exogenous]
            test_exogenous_df = test_df[exogenous]

            train_target_df["Patient_ID"] = new_patient_id
            train_target_df["ICULOS"] = train_df.index
            test_target_df["Patient_ID"] = new_patient_id
            test_target_df["ICULOS"] = test_df.index
            train_exogenous_df["Patient_ID"] = new_patient_id
            train_exogenous_df["ICULOS"] = train_df.index
            test_exogenous_df["Patient_ID"] = new_patient_id
            test_exogenous_df["ICULOS"] = test_df.index

            new_patient_id += 1

            y_train_list.append(train_target_df[target + ["Patient_ID", "ICULOS"]])
            y_test_list.append(test_target_df[target + ["Patient_ID", "ICULOS"]])
            X_train_list.append(train_exogenous_df[exogenous + ["Patient_ID", "ICULOS"]])
            X_test_list.append(test_exogenous_df[exogenous + ["Patient_ID", "ICULOS"]])

        y_train = pd.concat(y_train_list, ignore_index=True)
        y_test = pd.concat(y_test_list, ignore_index=True)
        X_train = pd.concat(X_train_list, ignore_index=True)
        X_test = pd.concat(X_test_list, ignore_index=True)

        y_train.set_index(["Patient_ID", "ICULOS"], inplace=True)
        y_test.set_index(["Patient_ID", "ICULOS"], inplace=True)
        X_train.set_index(["Patient_ID", "ICULOS"], inplace=True)
        X_test.set_index(["Patient_ID", "ICULOS"], inplace=True)

        return y_train, y_test, X_train, X_test


