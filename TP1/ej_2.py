import pandas as pd
import os

# Define the file paths
pkl_file = 'data/Noticias_argentinas.pkl'
excel_file = 'data/Noticias_argentinas.xlsx'

# Check if the .pkl file exists
if os.path.exists(pkl_file):
    # If .pkl file exists, load the DataFrame from it
    df = pd.read_pickle(pkl_file)
    print("Loaded DataFrame from pickle file.")
else:
    # If .pkl file does not exist, read the Excel file and save it to .pkl
    df = pd.read_excel(excel_file)
    df.to_pickle(pkl_file)
    print("Loaded DataFrame from Excel file and saved it to pickle.")

# Display the DataFrame
print(df)
