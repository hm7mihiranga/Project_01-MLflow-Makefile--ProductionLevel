from typing import Dict, Any
from abc import ABC, abstractmethod
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

MODEL_MAP = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "catboost": CatBoostClassifier,
    "xgboost": XGBClassifier,
}

class ModelFactory:
    @staticmethod
    def get_model(model_type: str, params: Dict[str, Any] = None):
        if model_type not in MODEL_MAP:
            raise ValueError(f"Model type '{model_type}' is not supported.")
        model_class = MODEL_MAP[model_type]
        params = params or {}
        return model_class(**params)