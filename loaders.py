import os
import glob
import pandas as pd
from sktime.datatypes import convert_to
from sktime.datatypes import check_is_mtype

from sktime.transformations.series.impute import Imputer


class DataLoader:

    def __init__(self, folder, column):
        self.folder = folder
        self.file_list = self._get_sorted_file_list()
        self.column = column if isinstance(column, list) else [column]

    def _get_sorted_file_list(self):
        file_list = glob.glob(os.path.join(self.folder, "*.csv"))
        return sorted(file_list, key=lambda x: int(os.path.basename(x).split(".")[0]))

    def get_subset(self, max_patients=None, min_time_points=36, fill_type="NaN"):

        df_list = []

        while len(df_list) < max_patients:
            for i, file in enumerate(self.file_list):
                df = pd.read_csv(file)
                df["PatientID"] = i

                # Drop if any columns are missing
                if not all(col in df.columns for col in self.column):
                    print(f"Patient {i + 1}: Missing one of {self.column} — skipping.")
                    continue

                # Drop if all values in target columns are NaN
                if df[self.column].isna().all().all():
                    print(f"Patient {i}: All {self.column} values are NaN — skipping.")
                    continue

                if df["ICULOS"].nunique() < min_time_points:
                    print(f"Patient {i}: Less than {min_time_points} time points — skipping.")
                    continue

                # Ensure time stamps are continuous
                df["ICULOS"] = pd.RangeIndex(start=0, stop=(len(df)), step=1)

                # fill method
                if fill_type == "ffill":
                    df = df.ffill().bfill()
                elif fill_type == "-1":
                    df = df.fillna(-1)
                # if NaN, left empty

                # Drop if there are constant columns
                if df[self.column].nunique().le(2).any():
                    print(f"Patient {i}: Dropping — constant columns found")
                    continue

                df_list.append(df[self.column + ["Patient_ID", "ICULOS"]])

        panel_df = pd.concat(df_list, ignore_index=True)
        panel_df.set_index(["Patient_ID", "ICULOS"], inplace=True)
        check_is_mtype(panel_df, mtype="pd-multiindex", scitype="Panel")

        return panel_df

    def split_y_X(self, df, endogenous, exogenous):
        y = df.drop(exogenous, axis=1)
        X = df.drop(endogenous, axis=1)

        return y, X