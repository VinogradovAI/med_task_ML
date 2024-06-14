import pandas as pd
import missingno
import matplotlib.pyplot as plt

def zero_nan_viewer(df: pd.DataFrame, min_abn_percentage: int =25) -> None:
    """
    Calculates and prints the percentage of zeros and NaNs in each column of the given DataFrame.
    If the percentage of zeros or NaNs is greater than or equal to the specified threshold, a recommendation is printed.
    """
    print("\nNull summary in each column:\n", df.isnull().sum())
    missingno.matrix(df)
    plt.show()

    zero_percentages = (df == 0).mean() * 100
    nan_percentages = df.isna().mean() * 100

    for column in df.columns:
        print(f"Column {column} has {zero_percentages[column]:.2f}% zeros and {nan_percentages[column]:.2f}% NaNs")
        if zero_percentages[column] >= min_abn_percentage or nan_percentages[column] >= min_abn_percentage:
            print(f"\033[91m      -> recommended to check expediency to drop the column {column}\033[00m")
        elif (10 >= zero_percentages[column] > 0) or (10 >= nan_percentages[column] > 0):
            print(f"\033[93m      -> recommended to check expediency to fill the column {column}\033[00m")


def data_cleaner(df: pd.DataFrame, drop_columns: list = None, fill_columns: list = None) -> pd.DataFrame:
    """
    Cleans the given DataFrame by dropping specified columns and filling specified columns with the mean value.
    """
    if drop_columns:
        df.drop(drop_columns, axis=1, inplace=True)
    if fill_columns:
        for column in fill_columns:
            df[column].fillna(df[column].mean(), inplace=True)
    return df

def zero_filler(df: pd.DataFrame, fill_columns: list = None) -> pd.DataFrame:
    """
    Fills zeros values in specified columns by mean value.
    """
    if fill_columns:
        for column in fill_columns:
            df[column].replace(0, df[column].mean(), inplace=True)
    return df

def zero_filler_adv(df: pd.DataFrame, fill_columns: list = None, key: bool = False, name: str = "" ) -> pd.DataFrame:
    """
    Fills zeros values in specified columns by mean value.
    If key is True, zeros are replaced by the mean of non-zero values in the column where 'target' is the same.
    """
    if fill_columns and name:
        for column in fill_columns:
            if key:
                for i in df.index:
                    if df.loc[i, column] == 0:
                        mean_val = df[(df[column] != 0) & (df[name] == df.loc[i, name])][column].mean()
                        df.loc[i, column] = mean_val
            else:
                df[column].replace(0, df[column].mean(), inplace=True)
    return df