import logging
import pandas as pd
import os
import json
from enum import Enum
from typing import Dict, List
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class FeatureEncodingStrategy(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame) ->pd.DataFrame:
        pass


class VariableType(str, Enum):
    NOMINAL = 'bincols'
    ORDINAL = 'multicols'

class BinaryEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, bin_cols: List[str]):
        self.bin_cols = bin_cols
        self.mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
        os.makedirs('artifacts/encode', exist_ok=True)

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.bin_cols:
            if col in df.columns:
                df[col] = df[col].map(self.mapping)
                # Save mapping for each column
                encoder_path = os.path.join('artifacts/encode', f"{col}_binary_encoder.json")
                with open(encoder_path, "w") as f:
                    json.dump(self.mapping, f)
                logging.info(f"Binary encoded column '{col}' and saved mapping to {encoder_path}")
        return df

class MultiCategoryEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, multi_cols: List[str]):
        self.multi_cols = multi_cols
        os.makedirs('artifacts/encode', exist_ok=True)

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = pd.get_dummies(df, columns=self.multi_cols, drop_first=True).astype(int)
        encoder_path = os.path.join('artifacts/encode', "multi_category_columns.json")
        with open(encoder_path, "w") as f:
            json.dump(df.columns.tolist(), f)
        logging.info(f"One-hot encoded columns: {self.multi_cols} and saved columns to {encoder_path}")
        return df