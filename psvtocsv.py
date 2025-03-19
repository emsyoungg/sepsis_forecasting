import os
import pandas as pd


class PSVToCSV:
    def __init__(self, input_dir, output_dir):
        """
        Initialises the converter with input and output directories.
        :param input_dir: Path to the directory containing .psv files.
        :param output_dir: Path to the directory where converted .csv files will be saved.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure output directory exists

    def convert_files(self):
        """
        Converts all .psv files in the input directory to .csv format
        and saves them with sequential filenames (1.csv, 2.csv, ...).
        """
        psv_files = [f for f in os.listdir(self.input_dir) if f.endswith(".psv")]

        for i, filename in enumerate(psv_files, start=1):
            self.convert_file(filename, i)

        print(f"Conversion completed. CSV files are stored in: {self.output_dir}")

    def convert_file(self, filename, index):
        """
        Converts a single .psv file to a .csv file.
        :param filename: Name of the .psv file.
        :param index: Sequential number for naming the output .csv file.
        """
        file_path = os.path.join(self.input_dir, filename)
        df = pd.read_csv(file_path, sep='|')  # Read PSV file
        output_file_path = os.path.join(self.output_dir, f"{index}.csv")  # Sequential filename
        df.to_csv(output_file_path, index=False)  # Save as CSV


if __name__ == "__main__":
    input_directory = "C:/Users/emily/Documents/training/training_setA"
    output_directory = "C:/Users/emily/Documents/training/training_setA_csv"

    converter = PSVToCSV(input_directory, output_directory)
    converter.convert_files()
