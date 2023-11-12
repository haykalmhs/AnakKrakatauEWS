import pandas as pd
import os

def load_folder_data(folder_path, label):
    all_data = pd.DataFrame()
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])  # Convert timestamp column to datetime
            df = df.rename(columns={'Unnamed: 0': 'timestamp'})
            df['label'] = label
            all_data = pd.concat([all_data, df])
    return all_data