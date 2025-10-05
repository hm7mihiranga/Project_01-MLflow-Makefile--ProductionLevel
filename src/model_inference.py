import json
import logging
import os
import joblib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_binning_config, get_encoding_config, get_data_paths, get_inference_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelInference:
    """  
    input format for single customer prediction:
    {
        "customerID": "7590-VHVEG",
        "gender": "Female", 
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No", 
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No", 
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", 
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }
    """
    
    def __init__(self, model_path: Optional[str] = None, encoders_dir: Optional[str] = None):
        logger.info(f"\n{'='*60}")
        logger.info("INITIALIZING TELCO CHURN MODEL INFERENCE")
        logger.info(f"{'='*60}")
        
        # Load configuration
        self.config = get_inference_config()
        self.data_paths = get_data_paths()
        
        # Set paths from config if not provided
        if model_path is None:
            artifacts_dir = self.data_paths.get('model_artifacts_dir', 'artifacts/models')
            model_name = self.config.get('model_name', 'random_forest_cv_model')
            model_path = os.path.join(artifacts_dir, f"{model_name}.pkl")
            
        if encoders_dir is None:
            encoders_dir = os.path.join(self.data_paths.get('artifacts_dir', 'artifacts'), 'encode')
        
        self.model_path = model_path
        self.encoders_dir = encoders_dir
        self.encoders = {}
        self.model = None
        
        logger.info(f"Configuration:")
        logger.info(f"  • Model Path: {model_path}")
        logger.info(f"  • Encoders Directory: {encoders_dir}")
        logger.info(f"  • Batch Size: {self.config.get('batch_size', 1000)}")
        logger.info(f"  • Return Probabilities: {self.config.get('return_proba', True)}")
        
        try:
            self.load_model()
            self.load_encoders()
            
            # Load configurations for preprocessing
            self.binning_config = get_binning_config()
            self.encoding_config = get_encoding_config()
            
            logger.info("✓ Telco churn model inference system initialized successfully")
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize model inference: {str(e)}")
            raise

    def load_model(self) -> None:
        logger.info("Loading trained churn prediction model...")
        
        if not self.model_path or not isinstance(self.model_path, str):
            logger.error("✗ Invalid model path provided")
            raise ValueError("Invalid model path provided")
            
        if not os.path.exists(self.model_path):
            logger.error(f"✗ Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            start_time = time.time()
            self.model = joblib.load(self.model_path)
            load_time = time.time() - start_time
            
            file_size = os.path.getsize(self.model_path) / (1024**2)  # MB
            
            logger.info(f"✓ Model loaded successfully:")
            logger.info(f"  • Model Type: {type(self.model).__name__}")
            logger.info(f"  • File Size: {file_size:.2f} MB")
            logger.info(f"  • Load Time: {load_time:.2f} seconds")
            logger.info(f"  • Has predict_proba: {hasattr(self.model, 'predict_proba')}")
            
        except Exception as e:
            logger.error(f"✗ Failed to load model: {str(e)}")
            raise

    def load_encoders(self) -> None:
        logger.info(f"\n{'='*50}")
        logger.info("LOADING FEATURE ENCODERS")
        logger.info(f"{'='*50}")
        
        if not os.path.exists(self.encoders_dir):
            logger.error(f"✗ Encoders directory not found: {self.encoders_dir}")
            raise FileNotFoundError(f"Encoders directory not found: {self.encoders_dir}")
        
        try:
            encoder_files = [f for f in os.listdir(self.encoders_dir) if f.endswith('_encoder.json')]
            
            if not encoder_files:
                logger.warning("⚠ No encoder files found in directory")
                return
            
            logger.info(f"Found {len(encoder_files)} encoder files")
            
            for file in encoder_files:
                feature_name = file.split('_encoder.json')[0]
                file_path = os.path.join(self.encoders_dir, file)
                
                with open(file_path, 'r') as f:
                    encoder_data = json.load(f)
                    self.encoders[feature_name] = encoder_data
                    
                logger.info(f"  ✓ Loaded encoder for '{feature_name}': {len(encoder_data)} mappings")
            
            # Load multi-category columns if available
            multi_cat_path = os.path.join(self.encoders_dir, 'multi_category_columns.json')
            if os.path.exists(multi_cat_path):
                with open(multi_cat_path, 'r') as f:
                    multi_cat_data = json.load(f)
                    # Check if it's a list (final feature names) or dict (mapping)
                    if isinstance(multi_cat_data, list):
                        self.final_feature_names = multi_cat_data
                        logger.info(f"  ✓ Loaded final feature names: {len(multi_cat_data)} features")
                    else:
                        self.multi_category_encoders = multi_cat_data
                        logger.info(f"  ✓ Loaded multi-category encoders: {len(multi_cat_data)} features")
            else:
                self.final_feature_names = []
                self.multi_category_encoders = {}
            
            logger.info(f"✓ All encoders loaded successfully")
            logger.info(f"{'='*50}\n")
            
        except Exception as e:
            logger.error(f"✗ Failed to load encoders: {str(e)}")
            raise

    def preprocess_single_record(self, data: Dict[str, Any]) -> pd.DataFrame:
        logger.info(f"\n{'='*50}")
        logger.info("PREPROCESSING CUSTOMER DATA")
        logger.info(f"{'='*50}")
        
        if not data or not isinstance(data, dict):
            logger.error("✗ Input data must be a non-empty dictionary")
            raise ValueError("Input data must be a non-empty dictionary")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([data])
            logger.info(f"✓ Input data converted to DataFrame: {df.shape}")
            logger.info(f"  • Customer ID: {data.get('customerID', 'N/A')}")
            logger.info(f"  • Input features: {len(df.columns)}")
            
            # Drop customerID if present (not needed for prediction)
            if 'customerID' in df.columns:
                df = df.drop('customerID', axis=1)
                logger.info("  ✓ Dropped customerID column")
            
            # Handle TotalCharges conversion (common issue in telecom data)
            if 'TotalCharges' in df.columns:
                original_value = df['TotalCharges'].iloc[0]
                # Convert to numeric, replacing empty strings/spaces with 0
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
                new_value = df['TotalCharges'].iloc[0]
                
                if original_value != new_value:
                    logger.info(f"  ✓ TotalCharges converted: '{original_value}' → {new_value}")
            
            # Apply tenure binning if configured
            if 'tenure' in df.columns and self.binning_config:
                logger.info("Applying tenure category binning...")
                original_tenure = df['tenure'].iloc[0]
                
                tenure_config = self.binning_config.get('tenure_category', {})
                
                def categorize_tenure(tenure):
                    if tenure <= 12:
                        return 'New'
                    elif tenure <= 48:
                        return 'Established'
                    else:
                        return 'Loyal'
                
                df['TenureCategory'] = df['tenure'].apply(categorize_tenure)
                logger.info(f"  ✓ Tenure categorized: {original_tenure} months → {df['TenureCategory'].iloc[0]}")
            
            # Apply binary encoding
            logger.info("Applying binary feature encoders...")
            
            # Map feature names to their proper encoder
            feature_encoder_mapping = {
                'gender': 'gender_binary',
                'Partner': 'Partner_binary', 
                'Dependents': 'Dependents_binary',
                'PhoneService': 'PhoneService_binary',
                'PaperlessBilling': 'PaperlessBilling_binary'
            }
            
            for feature, encoder_name in feature_encoder_mapping.items():
                if feature in df.columns and encoder_name in self.encoders:
                    original_value = df[feature].iloc[0]
                    encoder = self.encoders[encoder_name]
                    
                    # Apply encoding, default to 0 if value not found
                    if original_value in encoder:
                        df[feature] = encoder[original_value]
                        encoded_value = df[feature].iloc[0]
                        logger.info(f"  ✓ Encoded '{feature}': '{original_value}' → {encoded_value}")
                    else:
                        df[feature] = 0
                        logger.warning(f"  ⚠ Value '{original_value}' not found in encoder for '{feature}', defaulting to 0")
            
            # Apply multi-category encoding if available
            if hasattr(self, 'final_feature_names') and self.final_feature_names:
                logger.info("Applying multi-category encoders...")
                
                # Define the multi-category features and their possible values
                multi_category_features = {
                    'MultipleLines': ['No phone service', 'No', 'Yes'],
                    'InternetService': ['DSL', 'Fiber optic', 'No'],
                    'OnlineSecurity': ['No', 'Yes', 'No internet service'],
                    'OnlineBackup': ['No', 'Yes', 'No internet service'],
                    'DeviceProtection': ['No', 'Yes', 'No internet service'],
                    'TechSupport': ['No', 'Yes', 'No internet service'],
                    'StreamingTV': ['No', 'Yes', 'No internet service'],
                    'StreamingMovies': ['No', 'Yes', 'No internet service'],
                    'Contract': ['Month-to-month', 'One year', 'Two year'],
                    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
                }
                
                for feature, categories in multi_category_features.items():
                    if feature in df.columns:
                        original_value = df[feature].iloc[0]
                        
                        for category in categories[1:]:  
                            col_name = f"{feature}_{category}"
                            df[col_name] = (df[feature] == category).astype(int)
                        
                        # Drop original column
                        df = df.drop(feature, axis=1)
                        logger.info(f"Multi-category encoded '{feature}': '{original_value}' → {len(categories)-1} binary columns")
            
            # Ensure all numeric columns are properly typed
            numeric_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            if hasattr(self, 'final_feature_names') and self.final_feature_names:
                prediction_features = [col for col in self.final_feature_names if col != 'Churn']
                
                available_features = [col for col in prediction_features if col in df.columns]
                missing_features = [col for col in prediction_features if col not in df.columns]
                
                for feature in missing_features:
                    df[feature] = 0
                    logger.info(f"  ✓ Added missing feature '{feature}' with default value 0")
                
                df = df[prediction_features]
                logger.info(f"  ✓ Features reordered to match training data: {len(prediction_features)} features")
            
            logger.info(f"✓ Preprocessing completed:")
            logger.info(f"  • Final shape: {df.shape}")
            logger.info(f"  • Final features: {list(df.columns)}")
            logger.info(f"  • Missing values: {df.isnull().sum().sum()}")
            
            if df.isnull().sum().sum() > 0:
                logger.warning("⚠ Missing values detected after preprocessing")
                df = df.fillna(0)
                logger.info("  ✓ Missing values filled with 0")
            
            logger.info(f"{'='*50}\n")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Preprocessing failed: {str(e)}")
            raise
    
    def predict_single_customer(self, data: Dict[str, Any]) -> Dict[str, Union[str, float, int]]:
        logger.info(f"\n{'='*60}")
        logger.info("PREDICTING CUSTOMER CHURN")
        logger.info(f"{'='*60}")
        
        if not data:
            logger.error("✗ Input data cannot be empty")
            raise ValueError("Input data cannot be empty")
        
        if self.model is None:
            logger.error("✗ Model not loaded")
            raise ValueError("Model not loaded")
        
        try:
            # Preprocess input data
            processed_data = self.preprocess_single_record(data)
            
            # Make prediction
            logger.info("Generating churn prediction...")
            start_time = time.time()
            
            y_pred = self.model.predict(processed_data)
            prediction_time = time.time() - start_time
            
            # Get probability if available
            if hasattr(self.model, 'predict_proba') and self.config.get('return_proba', True):
                y_proba = self.model.predict_proba(processed_data)
                churn_probability = float(y_proba[0][1])  # Probability of churn (class 1)
                retain_probability = float(y_proba[0][0])  # Probability of retention (class 0)
            else:
                churn_probability = float(y_pred[0])  # Binary prediction as probability
                retain_probability = 1.0 - churn_probability
            
            # Process results
            prediction = int(y_pred[0])
            churn_status = 'Will Churn' if prediction == 1 else 'Will Retain'
            confidence_score = max(churn_probability, retain_probability) * 100
            
            # Create comprehensive result
            result = {
                "customer_id": data.get('customerID', 'Unknown'),
                "prediction": prediction,
                "churn_status": churn_status,
                "churn_probability": round(churn_probability, 4),
                "retain_probability": round(retain_probability, 4),
                "confidence_score": round(confidence_score, 2),
                "prediction_time_ms": round(prediction_time * 1000, 2),
                "model_type": type(self.model).__name__,
                "features_used": len(processed_data.columns)
            }
            
            logger.info("✓ Prediction completed successfully:")
            logger.info(f"  • Customer ID: {result['customer_id']}")
            logger.info(f"  • Churn Status: {result['churn_status']}")
            logger.info(f"  • Churn Probability: {result['churn_probability']:.1%}")
            logger.info(f"  • Confidence Score: {result['confidence_score']:.1f}%")
            logger.info(f"  • Prediction Time: {result['prediction_time_ms']:.2f}ms")
            logger.info(f"  • Features Used: {result['features_used']}")
            logger.info(f"{'='*60}\n")
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Prediction failed: {str(e)}")
            raise
    
    def predict_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Union[str, float, int]]]:
        logger.info(f"\n{'='*60}")
        logger.info("BATCH CHURN PREDICTION")
        logger.info(f"{'='*60}")
        
        if not data_list or not isinstance(data_list, list):
            logger.error("✗ Input must be a non-empty list of dictionaries")
            raise ValueError("Input must be a non-empty list of dictionaries")
        
        batch_size = self.config.get('batch_size', 1000)
        total_customers = len(data_list)
        
        logger.info(f"Processing {total_customers} customers in batches of {batch_size}")
        
        try:
            results = []
            start_time = time.time()
            
            for i in range(0, total_customers, batch_size):
                batch = data_list[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_customers + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} customers)")
                
                for customer_data in batch:
                    try:
                        result = self.predict_single_customer(customer_data)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Failed to predict for customer {customer_data.get('customerID', 'Unknown')}: {str(e)}")
                        # Add error result
                        error_result = {
                            "customer_id": customer_data.get('customerID', 'Unknown'),
                            "prediction": -1,
                            "churn_status": "Error",
                            "error": str(e)
                        }
                        results.append(error_result)
            
            total_time = time.time() - start_time
            successful_predictions = len([r for r in results if r.get('prediction', -1) != -1])
            
            logger.info(f"✓ Batch prediction completed:")
            logger.info(f"  • Total Customers: {total_customers}")
            logger.info(f"  • Successful Predictions: {successful_predictions}")
            logger.info(f"  • Failed Predictions: {total_customers - successful_predictions}")
            logger.info(f"  • Total Time: {total_time:.2f} seconds")
            logger.info(f"  • Average Time per Customer: {(total_time/total_customers)*1000:.2f}ms")
            logger.info(f"{'='*60}\n")
            
            return results
            
        except Exception as e:
            logger.error(f"✗ Batch prediction failed: {str(e)}")
            raise

    # Legacy methods for backward compatibility
    def predict(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> Union[np.ndarray, Dict]:
        if isinstance(X, dict):
            return self.predict_single_customer(X)
        elif isinstance(X, list):
            return self.predict_batch(X)
        else:
            return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            logger.warning("Model does not support predict_proba. Using predict instead.")
            return self.model.predict(X)