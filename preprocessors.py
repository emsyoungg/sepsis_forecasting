import os
import glob
import pandas as pd
from sktime.datatypes import check_is_mtype


class Loader:
    def __init__(self, folder, column=None):
        self.folder = folder
        self.file_list = self._get_sorted_file_list()
        self.column = column if isinstance(column, list) else [column]

    def _get_sorted_file_list(self):
        file_list = glob.glob(os.path.join(self.folder, "*.csv"))
        return sorted(file_list, key=lambda x: int(os.path.basename(x).split(".")[0]))

    def get_panel_df(self, max_instances=None, min_time_points=36, equal_length=False):
        df_list = []
        new_patient_id = 0
        for i, file in enumerate(self.file_list):
            if max_instances is not None:
                if len(df_list) >= max_instances:
                    break
            instance_df = pd.read_csv(file)
            # Drop if any columns are missing
            if not all(col in instance_df.columns for col in self.column):
                continue
            # Drop if all values in target columns are NaN
            if instance_df[self.column].isna().all().all():
                continue
            if instance_df["ICULOS"].nunique() < min_time_points:
                continue
            if instance_df[self.column].nunique().le(2).any():
                continue

            instance_df["ICULOS"] = pd.RangeIndex(start=0, stop=(len(instance_df)), step=1)
            if equal_length and instance_df["ICULOS"].nunique() > min_time_points:
                instance_df = instance_df[:min_time_points]

            instance_df["Patient_ID"] = new_patient_id
            new_patient_id += 1

            df_list.append(instance_df[self.column + ["Patient_ID", "ICULOS"]])

        panel_df = pd.concat(df_list, ignore_index=True)
        panel_df.set_index(["Patient_ID", "ICULOS"], inplace=True)
        check_is_mtype(panel_df, mtype="pd-multiindex", scitype="Panel")
        print(f"Successfully loaded {len(df_list)} patients.")
        return panel_df

    def panel_imputer(self, df, method="ffill"):
        df_list = []
        for patient_id in df.index.get_level_values("Patient_ID").unique():
            instance_df = df.loc[patient_id]
            if method == "ffill":
                instance_df = instance_df.ffill().bfill()
            elif method == "-1":
                instance_df = instance_df.fillna(-1)

            instance_df["Patient_ID"] = patient_id
            instance_df["ICULOS"] = instance_df.index

            df_list.append(instance_df[self.column + ["Patient_ID", "ICULOS"]])
        panel_df = pd.concat(df_list, ignore_index=True)
        panel_df.set_index(["Patient_ID", "ICULOS"], inplace=True)
        check_is_mtype(panel_df, mtype="pd-multiindex", scitype="Panel")
        return panel_df

    def get_flat_df(self):
        df_list = []
        for i, file in enumerate(self.file_list):
            df = pd.read_csv(file)
            df["Patient_ID"] = i
            df_list.append(df)

        hospitalA_df = pd.concat(df_list, ignore_index=True)
        hospitalA_df = hospitalA_df.sort_values(by=["Patient_ID", "ICULOS"]).reset_index(drop=True)

        return hospitalA_df

    def split_by_fh(self, panel_df, fh=4):
        train_list = []
        test_list = []
        new_patient_id = 0

        for patient_id in panel_df.index.get_level_values("Patient_ID").unique():
            patient_df = panel_df.loc[patient_id]

            split_point = patient_df.index.max() - fh
            train_df = patient_df.loc[:split_point].copy()
            test_df = patient_df.loc[split_point + 1:].copy()
            if train_df[self.column].nunique().le(2).any():
                print(f"Warning - Patient {patient_id}: Dropping — constant columns found")
                continue
            #if test_df[self.column].nunique().le(2).any():
             #   print(f"Warning - Patient {patient_id}: Dropping — constant columns found")
              #  continue

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

    def split_y_X(self, train_data, test_data, endogenous, exogenous, drop=None):
        if drop:
            train_data = train_data.drop(drop, axis=1)
            test_data = test_data.drop(drop, axis=1)
        y_train = train_data.drop(exogenous, axis=1)
        y_test = test_data.drop(exogenous, axis=1)
        X_train = train_data.drop(endogenous, axis=1)
        X_test = test_data.drop(endogenous, axis=1)

        return y_train, y_test, X_train, X_test

    def split_by_instance(self, panel_df):

        patient_ids = panel_df.index.get_level_values(0).unique()
        split_idx = int(0.9 * len(patient_ids))

        train_ids = patient_ids[:split_idx]
        test_ids = patient_ids[split_idx:]

        train_data = panel_df.loc[train_ids]
        test_data = panel_df.loc[test_ids]

        return train_data, test_data






