import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test
