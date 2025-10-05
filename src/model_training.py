import os
import joblib
import logging
import time
import json
from datetime import datetime
from typing import Any, Tuple, Union, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.cv_folds = self.config.get('cv_folds', 5)
        self.random_state = self.config.get('random_state', 42)
        self.validation_split = self.config.get('validation_split', 0.2)
        
        logger.info("ModelTrainer initialized with production-level capabilities")
        if config:
            logger.info(f"Configuration loaded: CV folds={self.cv_folds}, Random state={self.random_state}")
    
    def train(
            self,
            model: BaseEstimator,
            X_train: Union[pd.DataFrame, np.ndarray],
            Y_train: Union[pd.Series, np.ndarray],
            model_name: str = "model",
            use_cross_validation: bool = True
            ) -> Tuple[BaseEstimator, float, Dict]:
        
        logger.info(f"\n{'='*70}")
        logger.info(f"MODEL TRAINING: {model_name.upper()}")
        logger.info(f"{'='*70}")
        
        # Input validation
        self._validate_training_data(X_train, Y_train)
        
        # Log training configuration
        training_info = self._log_training_config(model, X_train, Y_train, model_name, use_cross_validation)
        
        try:
            # Start training
            logger.info("Starting model training...")
            start_time = time.time()
            
            # Fit the model
            model.fit(X_train, Y_train)
            
            training_time = time.time() - start_time
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            
            # Calculate training score
            train_score = self._calculate_training_score(model, X_train, Y_train)
            
            # Perform cross-validation if requested
            cv_scores = None
            if use_cross_validation:
                cv_scores = self._perform_cross_validation(model, X_train, Y_train)
            
            # Update training info
            training_info.update({
                'training_time': training_time,
                'train_score': train_score,
                'cv_scores': cv_scores,
                'training_completed_at': datetime.now().isoformat()
            })
            
            logger.info(f"{model_name} training successful!")
            logger.info(f"{'='*70}\n")
            
            return model, train_score, training_info
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def train_simple(
            self, 
            model: BaseEstimator, 
            X_train: Union[pd.DataFrame, np.ndarray], 
            y_train: Union[pd.Series, np.ndarray]
            ) -> Tuple[BaseEstimator, float]:
        
        trained_model, train_score, _ = self.train(model, X_train, y_train, use_cross_validation=False)
        return trained_model, train_score
    
    def save_model(
            self, 
            model: BaseEstimator, 
            filepath: str, 
            model_info: Optional[Dict] = None,
            save_format: str = 'joblib'
            ) -> None:

        logger.info(f"\n{'='*70}")
        logger.info("MODEL SAVING")
        logger.info(f"{'='*70}")
        
        # Input validation
        self._validate_model_save_inputs(model, filepath, save_format)
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            logger.info(f"Saving model to: {filepath}")
            logger.info(f"Format: {save_format}")
            start_time = time.time()
            
            # Save the model based on format
            if save_format.lower() == 'joblib':
                joblib.dump(model, filepath)
            elif save_format.lower() == 'pickle':
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
            else:
                raise ValueError(f"Unsupported save format: {save_format}")
            
            save_time = time.time() - start_time
            file_size = os.path.getsize(filepath) / (1024**2)  # MB
            
            # Save model metadata if provided
            if model_info:
                metadata_path = filepath.replace('.pkl', '_metadata.json')
                self._save_model_metadata(model_info, metadata_path)
            
            logger.info(f"Model saved successfully!")
            logger.info(f"File Path: {filepath}")
            logger.info(f"File Size: {file_size:.2f} MB")
            logger.info(f"Save Time: {save_time:.2f} seconds")
            logger.info(f"Model Type: {type(model).__name__}")
            logger.info(f"{'='*70}\n")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    def load_model(self, filepath: str, load_format: str = 'auto') -> BaseEstimator:
        logger.info(f"\n{'='*70}")
        logger.info("MODEL LOADING")
        logger.info(f"{'='*70}")
        
        # Input validation
        self._validate_model_load_inputs(filepath)
        
        try:
            logger.info(f"Loading model from: {filepath}")
            start_time = time.time()
            
            # Auto-detect format if not specified
            if load_format == 'auto':
                load_format = 'joblib' if filepath.endswith('.pkl') or filepath.endswith('.joblib') else 'pickle'
            
            # Load the model based on format
            if load_format.lower() == 'joblib':
                model = joblib.load(filepath)
            elif load_format.lower() == 'pickle':
                import pickle
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
            else:
                raise ValueError(f"Unsupported load format: {load_format}")
            
            load_time = time.time() - start_time
            file_size = os.path.getsize(filepath) / (1024**2)  # MB
            
            # Load metadata if available
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            metadata = self._load_model_metadata(metadata_path)
            
            logger.info(f"Model loaded successfully!")
            logger.info(f"Model Type: {type(model).__name__}")
            logger.info(f"File Size: {file_size:.2f} MB")
            logger.info(f"Load Time: {load_time:.2f} seconds")
            logger.info(f"Format: {load_format}")

            if metadata:
                logger.info(f"Metadata: Available")

            logger.info(f"{'='*70}\n")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _validate_training_data(self, X_train, Y_train) -> None:
        """Validate training data inputs."""
        if X_train is None or Y_train is None:
            logger.error("Training data cannot be None")
            raise ValueError("Training data cannot be None")
            
        if len(X_train) == 0 or len(Y_train) == 0:
            logger.error("Training data cannot be empty")
            raise ValueError("Training data cannot be empty")
            
        if len(X_train) != len(Y_train):
            logger.error(f"Feature and target length mismatch: {len(X_train)} vs {len(Y_train)}")
            raise ValueError(f"Feature and target length mismatch: {len(X_train)} vs {len(Y_train)}")
    
    def _log_training_config(self, model, X_train, Y_train, model_name, use_cross_validation) -> Dict:
        """Log training configuration and return training info dictionary."""
        logger.info(f"Training Configuration:")
        logger.info(f"Model Type: {type(model).__name__}")
        logger.info(f"Training Samples: {len(X_train):,}")
        logger.info(f"Features: {X_train.shape[1] if hasattr(X_train, 'shape') else 'Unknown'}")

        # Log target distribution
        if hasattr(Y_train, 'value_counts'):
            target_dist = Y_train.value_counts().to_dict()
        else:
            target_dist = dict(zip(*np.unique(Y_train, return_counts=True)))

        logger.info(f"Target Distribution: {target_dist}")
        logger.info(f"Cross Validation: {'Yes' if use_cross_validation else 'No'}")

        return {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'training_samples': len(X_train),
            'features': X_train.shape[1] if hasattr(X_train, 'shape') else None,
            'target_distribution': target_dist,
            'cross_validation': use_cross_validation,
            'training_started_at': datetime.now().isoformat()
        }
    
    def _calculate_training_score(self, model, X_train, Y_train) -> float:
        """Calculate and log training score."""
        logger.info("Calculating training score...")
        train_score = model.score(X_train, Y_train)
        logger.info(f"Training Score: {train_score:.4f}")
        return train_score
    
    def _perform_cross_validation(self, model, X_train, Y_train) -> Dict:
        """Perform cross-validation and return results."""
        logger.info(f"Performing {self.cv_folds}-fold cross-validation...")
        
        try:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X_train, Y_train, cv=cv, scoring='accuracy')
            
            cv_results = {
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_folds': self.cv_folds
            }

            logger.info(f"Cross-validation completed:")
            logger.info(f"CV Mean Score: {cv_results['cv_mean']:.4f}")
            logger.info(f"CV Std Score: {cv_results['cv_std']:.4f}")
            logger.info(f"CV Scores: {[f'{score:.4f}' for score in cv_scores]}")

            return cv_results
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
            return None
    
    def _validate_model_save_inputs(self, model, filepath, save_format) -> None:
        """Validate model saving inputs."""
        if model is None:
            logger.error("Cannot save None model")
            raise ValueError("Cannot save None model")
            
        if not filepath or not isinstance(filepath, str):
            logger.error("Invalid filepath provided")
            raise ValueError("Invalid filepath provided")
            
        if save_format not in ['joblib', 'pickle']:
            logger.error(f"Unsupported save format: {save_format}")
            raise ValueError(f"Unsupported save format: {save_format}")
    
    def _validate_model_load_inputs(self, filepath) -> None:
        """Validate model loading inputs."""
        if not filepath or not isinstance(filepath, str):
            logger.error("Invalid filepath provided")
            raise ValueError("Invalid filepath provided")
            
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            raise FileNotFoundError(f"Model file not found: {filepath}")
    
    def _save_model_metadata(self, metadata: Dict, metadata_path: str) -> None:
        """Save model metadata to JSON file."""
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save metadata: {str(e)}")

    def _load_model_metadata(self, metadata_path: str) -> Optional[Dict]:
        """Load model metadata from JSON file."""
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {str(e)}")
        return None