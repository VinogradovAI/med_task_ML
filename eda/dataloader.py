import pandas as pd
import missingno
import matplotlib.pyplot as plt

def data_loader(file_path: str) -> pd.DataFrame:
    """ Load the data from the given file path. """
    return pd.read_csv(file_path)

def data_viewer(df: pd.DataFrame) -> None:
    """ Prints general information about the given DataFrame. """
    print(df.head())
    print(df.info())
    print(df.describe())