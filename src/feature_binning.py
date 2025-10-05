import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Union

logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class FeatureBinningStrategy(ABC):
    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass


class TenureBinnerStrategy(FeatureBinningStrategy):
    def __init__(self, bins: List[float], labels: List[str]):
        if len(bins) != len(labels):
            raise ValueError("Bins and labels must have the same length")
        if not bins or not labels:
            raise ValueError("Bins and labels cannot be empty")
        
        self.bins = sorted(bins)  
        self.labels = labels

    def tenure_category(self, tenure: float) -> str:
        if pd.isna(tenure):
            return "Unknown"
        
        for i, upper in enumerate(self.bins):
            if tenure <= upper:
                return self.labels[i]
        return self.labels[-1]

    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Implementation of the abstract method from FeatureBinningStrategy"""
        return self.add_tenure_category(df, column)

    def add_tenure_category(self, df: pd.DataFrame, tenure_col: str) -> pd.DataFrame:
        if tenure_col not in df.columns:
            raise ValueError(f"Column '{tenure_col}' not found in DataFrame")
        
        logging.info(f"Adding tenure category based on '{tenure_col}' with bins {self.bins} and labels {self.labels}")
        df = df.copy()
        df['TenureCategory'] = df[tenure_col].apply(self.tenure_category)
        return df

    def bin_features(self, df: pd.DataFrame, tenure_col: str) -> pd.DataFrame:
        logging.info("Starting feature binning process")
        df = self.add_tenure_category(df, tenure_col=tenure_col)
        logging.info("Feature binning process completed")
        return df