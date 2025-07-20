import os
import pandas as pd
import kaggle

class KaggleDataSet:
    def __init__(self, local_data_dir='./kaggle_data'):
        self.local_data_dir = local_data_dir
        os.makedirs(self.local_data_dir, exist_ok=True) # Ensure the directory exists
        try:
            kaggle.api.authenticate()
            print("Kaggle API authenticated successfully.")
        except Exception as e:
            print(f"Error authenticating Kaggle API. Please ensure your kaggle.json is correctly set up: {e}")

    def download(self, dataset_slug, file_name_in_dataset, force_download=False):
        local_file_path = os.path.join(self.local_data_dir, dataset_slug.replace('/', '_'), file_name_in_dataset)
        dataset_local_dir = os.path.join(self.local_data_dir, dataset_slug.replace('/', '_'))
        os.makedirs(dataset_local_dir, exist_ok=True)
        if os.path.exists(local_file_path) and not force_download:
            print(f"File '{file_name_in_dataset}' from dataset '{dataset_slug}' already exists locally at '{local_file_path}'. Loading from local storage.")
            try:
                df = pd.read_csv(local_file_path)
                return df
            except Exception as e:
                print(f"Error reading local file {local_file_path}: {e}")
                print("Attempting to re-download the file.")
                return self._perform_download_and_save(dataset_slug, file_name_in_dataset, local_file_path, dataset_local_dir)
        else:
            print(f"File '{file_name_in_dataset}' from dataset '{dataset_slug}' not found locally or force_download is True. Downloading from Kaggle...")
            return self._perform_download_and_save(dataset_slug, file_name_in_dataset, local_file_path, dataset_local_dir)

    def _perform_download_and_save(self, dataset_slug, file_name_in_dataset, local_file_path, download_path):
        try:
            # Kaggle API's dataset_download_file saves directly to the specified path
            # It will download the file directly into 'download_path'
            kaggle.api.dataset_download_file(
                dataset=dataset_slug,
                file_name=file_name_in_dataset,
                path=download_path
            )

            print(f"Download complete. File saved to '{local_file_path}'.")
            df = pd.read_csv(local_file_path)
            return df

        except Exception as e:
            print(f"An error occurred during download or saving: {e}")
            return None