#%%
import pandas as pd
import torch
from torch.utils.data import Dataset


class WeatherDataset(Dataset):

    def __init__(self, dataset_file, day_range, split_date,
                 train_test="train", mean=None, std=None):

        # Accept dataframe OR csv file
        if isinstance(dataset_file, pd.DataFrame):
            df = dataset_file.copy()
        else:
            df = pd.read_csv(dataset_file)

        # --- Preprocessing ---
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values("date")

        # Encode weather column
        if df['weather'].dtype == 'object':
            df['weather'] = df['weather'].astype('category').cat.codes

        # --- Split ---
        if train_test == "train":
            df = df[df['date'] < split_date]
        else:
            df = df[df['date'] >= split_date]

        df = df.drop(columns=['date'])

        # --- NORMALIZATION (IMPORTANT) ---
        if train_test == "train":
            # Compute mean and std from TRAIN data
            self.mean = df.mean().values
            self.std = df.std().values

        else:
            # Use provided mean/std (from training set)
            assert mean is not None and std is not None, \
                "Test dataset requires mean and std from training dataset"

            self.mean = mean
            self.std = std

        # Avoid division by zero
        self.std[self.std == 0] = 1e-8

        # Normalize
        df = (df - self.mean) / self.std

        # Store
        self.data = df.values
        self.day_range = day_range

    def __len__(self):
        return len(self.data) - self.day_range

    def __getitem__(self, idx):

        # Input sequence (14 days)
        x = self.data[idx:idx + self.day_range - 1]

        # Target = temp_max (column index 1)
        y = self.data[idx + self.day_range - 1][1]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# %%