import logging
import pandas as pd
from enum import Enum
from typing import Optional
from pydantic import BaseModel
import numpy as np
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) ->pd.DataFrame:
        pass

class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, critical_columns=[]):
        self.critical_columns = critical_columns 
        logging.info(f"Dropping rows with missing values in critical columns: {self.critical_columns}")

    def handle(self, df):
        df_cleaned = df.dropna(subset=self.critical_columns)
        n_dropped = len(df) - len(df_cleaned)
        logging.info(f"{n_dropped} has been dropped")
        return df_cleaned


    
class CleanNumericColumnStrategy(MissingValueHandlingStrategy):
    """
    Cleans a "TotalCharges" column by:
    - Replacing blank strings with NaN
    - Converting to float
    - Filling NaN with the column's median
    """
    def __init__(self, column_name: str):
        self.column_name = column_name

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Data types before cleaning:\n{df.dtypes}")

        non_numeric = df[~df[self.column_name].apply(lambda x: str(x).replace('.', '', 1).isdigit())][self.column_name]
        logging.info(f"Non-numeric values in '{self.column_name}': {non_numeric.unique()}")

        df[self.column_name] = df[self.column_name].replace(" ", np.nan)
        df[self.column_name] = df[self.column_name].astype(float)
        df[self.column_name] = df[self.column_name].fillna(df[self.column_name].median())

        logging.info(f"Data types after cleaning:\n{df.dtypes}")
        return df

    
class DropColumnStrategy(MissingValueHandlingStrategy):
    '''
    Remove "customerID" column
    '''
    def __init__(self, column_name: str):
        self.column_name = column_name

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.column_name in df.columns:
            df = df.drop(self.column_name, axis=1)
            logging.info(f"'{self.column_name}' column dropped.")
        else:
            logging.info(f"'{self.column_name}' column not found.")
        return df