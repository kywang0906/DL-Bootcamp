import os
import pandas as pd

SAVE_FOLDER = "raw_data"
CLEANED_FOLDER = "cleaned_data"

def clean():
    files = [f for f in os.listdir(SAVE_FOLDER) if f.endswith(".csv")]

    for file in files:
        file_path = os.path.join(SAVE_FOLDER, file)
        print(f"Cleaning {file_path}...")

        df = pd.read_csv(file_path)
        df["Title"] = df["Title"].astype(str).str.strip()
        df["Title"] = df["Title"].str.lower()
        df = df[~df["Title"].str.startswith(("re:", "fw:"))]
        # Fix repeated `""` issues
        df["Title"] = df["Title"].str.replace(r'[!"#$%&\'()*+,\-./:;<=>?@\\^_`{|}~]', '', regex=True)

        cleaned_file_path = os.path.join(CLEANED_FOLDER, file)
        df.to_csv(cleaned_file_path, index=False, encoding="utf-8-sig")

        print(f"Cleaning completed, saved to {cleaned_file_path}")

if __name__ == '__main__':
    # Make sure cleaned data folder exists
    if not os.path.exists(CLEANED_FOLDER):
        os.makedirs(CLEANED_FOLDER)
    clean()
    print("All CSV files have been cleaned!")
