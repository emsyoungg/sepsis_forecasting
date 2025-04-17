import os
import glob
import pandas as pd
from sktime.datatypes import convert_to
from sktime.datatypes import check_is_mtype

from sktime.transformations.series.impute import Imputer


class Preprocessors:
    def __init__(self, folder, column):
        self.folder = folder
        self.file_list = self._get_sorted_file_list()
        self.column = column if isinstance(column, list) else [column]

    def _get_sorted_file_list(self):
        file_list = glob.glob(os.path.join(self.folder, "*.csv"))
        return sorted(file_list, key=lambda x: int(os.path.basename(x).split(".")[0]))

    def ARIMA_VAR_preprocessor(self, max_patients=None, min_time_points=36):
        df_list = []
        for i, file in enumerate(self.file_list):
            if len(df_list) >= max_patients:
                break

            df = pd.read_csv(file)
            df["Patient_ID"] = i
            # Drop if any columns are missing
            if not all(col in df.columns for col in self.column):
                print(f"Patient {i}: Missing one of {self.column} — skipping.")
                continue
            # Drop if all values in target columns are NaN
            if df[self.column].isna().all().all():
                print(f"Patient {i}: All {self.column} values are NaN — skipping.")
                continue
            if df["ICULOS"].nunique() < min_time_points:
                print(f"Patient {i}: Less than {min_time_points} time points — skipping.")
                continue

            df = df.ffill().bfill()

            # Drop if there are constant columns
            if df[self.column].nunique().le(2).any():
                print(f"Patient {i}: Dropping — constant columns found")
                continue

            df_list.append(df[self.column + ["Patient_ID", "ICULOS"]])

        full_df = pd.concat(df_list, ignore_index=True)
        full_df.set_index(["Patient_ID", "ICULOS"], inplace=True)
        # Check whether it's compatible with sktime's "pd-multiindex" format
        check_is_mtype(full_df, mtype="pd-multiindex", scitype="Panel")
        return full_df

    def LSTM_preprocessor(self, max_patients=None, min_time_points=30):
        df_list = []
        new_patient_id = 0

        for i, file in enumerate(self.file_list):
            if len(df_list) >= max_patients:
                break

            df = pd.read_csv(file)

            if not all(col in df.columns for col in self.column):
                print(f"Patient {i + 1}: Missing one of {self.column} — skipping.")
                continue
            # Drop if all values in target columns are NaN
            if df[self.column].isna().all().all():
                print(f"Patient {i + 1}: All {self.column} values are NaN — skipping.")
                continue
            if df["ICULOS"].nunique() < min_time_points:
                print(f"Patient {i + 1}: Less than 30 time points — skipping.")
                continue

            df["ICULOS"] = pd.RangeIndex(start=0, stop=(len(df)), step=1)

            if df[self.column].nunique().le(2).any():
                print(f"Patient {i + 1}: Dropping — constant columns found")
                continue
            # make all time series the same length
            if df["ICULOS"].nunique() > min_time_points:
                df = df[:min_time_points]

            df["Patient_ID"] = new_patient_id
            new_patient_id += 1

            df_list.append(df[self.column + ["Patient_ID", "ICULOS"]])

        full_df = pd.concat(df_list, ignore_index=True)
        full_df.set_index(["Patient_ID", "ICULOS"], inplace=True)
        # Check whether it's compatible with sktime's "pd-multiindex" format
        check_is_mtype(full_df, mtype="pd-multiindex", scitype="Panel")  # will raise if invalid

        return full_df

    def XGBoost_preprocessor(self):
        df_list = [
            pd.read_csv(file, sep="|").assign(Patient_ID=i)
            for i, file in enumerate(self.file_list)
        ]
        hospitalA_df = pd.concat(df_list, ignore_index=True)
        hospitalA_df = hospitalA_df.sort_values(by=["Patient_ID", "ICULOS"]).reset_index(drop=True)

        hospitalA_df = hospitalA_df.ffill().bfill()

        return hospitalA_df

    def split_train_test(self, data, test_size=4, X=None):
        train_list = []
        test_list = []

        for patient_id in data.index.get_level_values("Patient_ID").unique():
            patient_df = data.loc[patient_id]

            split_point = patient_df.index.max() - test_size
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

    def split_y_X(self, train_data, test_data, endogenous, exogenous):
        y_train = train_data.drop(exogenous, axis=1)
        y_test = test_data.drop(exogenous, axis=1)
        X_train = train_data.drop(endogenous, axis=1)
        X_test = test_data.drop(endogenous, axis=1)

        return y_train, X_train, y_test, X_test