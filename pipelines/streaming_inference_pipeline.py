import os
import sys
import pandas as pd
import logging
import json
from typing import Dict, List, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_inference import ModelInference
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_model_config, get_inference_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_customer_data() -> List[Dict[str, Any]]:
    """
    Create sample customer data for streaming inference demonstration.
    """
    return [
        {
            "customerID": "DEMO-001",
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
            "TotalCharges": "29.85"
        },
        {
            "customerID": "DEMO-002",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 34,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 56.95,
            "TotalCharges": "1889.5"
        },
        {
            "customerID": "DEMO-003",
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 45,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 89.10,
            "TotalCharges": "4009.5"
        }
    ]

def main():
    """
    Main streaming inference pipeline function.
    """
    logger.info("="*60)
    logger.info("TELCO CHURN STREAMING INFERENCE PIPELINE")
    logger.info("="*60)
    
    try:
        # Load configuration
        inference_config = get_inference_config()
        
        # Configuration parameters
        model_name = inference_config.get('model_name', 'random_forest_cv_model')
        sample_size = inference_config.get('sample_size', 100)
        save_path = inference_config.get('save_path', 'artifacts/predictions/streaming_predictions.csv')
        batch_size = inference_config.get('batch_size', 1000)
        return_proba = inference_config.get('return_proba', True)
        
        logger.info(f"Configuration loaded:")
        logger.info(f"  • Model: {model_name}")
        logger.info(f"  • Batch Size: {batch_size}")
        logger.info(f"  • Return Probabilities: {return_proba}")
        
        # Initialize ModelInference with enhanced capabilities
        logger.info("Initializing enhanced model inference engine...")
        model_infer = ModelInference()
        
        # Create sample customer data for demonstration
        logger.info("Creating sample customer data for streaming inference...")
        sample_customers = create_sample_customer_data()
        
        logger.info(f"Processing {len(sample_customers)} sample customers...")
        
        # Process customers using the enhanced inference engine
        predictions = []
        
        # Single customer predictions
        logger.info("\n" + "="*50)
        logger.info("SINGLE CUSTOMER PREDICTIONS")
        logger.info("="*50)
        
        for i, customer in enumerate(sample_customers, 1):
            logger.info(f"\nProcessing Customer {i}: {customer['customerID']}")
            try:
                result = model_infer.predict_single_customer(customer)
                predictions.append({
                    'customer_id': result['customer_id'],
                    'churn_prediction': result['prediction'],
                    'churn_status': result['churn_status'],
                    'churn_probability': result['churn_probability'],
                    'confidence_score': result['confidence_score'],
                    'prediction_time_ms': result['prediction_time_ms']
                })
                
                # Log individual result
                logger.info(f"  Result: {result['churn_status']} (Confidence: {result['confidence_score']:.1f}%)")
                
            except Exception as e:
                logger.error(f"  Failed to predict for customer {customer['customerID']}: {str(e)}")
                predictions.append({
                    'customer_id': customer['customerID'],
                    'churn_prediction': -1,
                    'churn_status': 'Error',
                    'error': str(e)
                })
        
        # Batch prediction demonstration
        logger.info("\n" + "="*50)
        logger.info("BATCH PREDICTION DEMONSTRATION")
        logger.info("="*50)
        
        try:
            batch_results = model_infer.predict_batch(sample_customers)
            logger.info(f"Batch processing completed: {len(batch_results)} results")
            
            # Update predictions with batch results
            for result in batch_results:
                if result.get('prediction', -1) != -1:  # Successful prediction
                    # Find existing prediction and update with batch result
                    for pred in predictions:
                        if pred['customer_id'] == result['customer_id']:
                            pred['batch_prediction'] = result['prediction']
                            pred['batch_churn_status'] = result['churn_status']
                            break
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
        
        # Save predictions
        logger.info("\n" + "="*50)
        logger.info("SAVING PREDICTIONS")
        logger.info("="*50)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as CSV
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(save_path, index=False)
        logger.info(f"✓ Predictions saved to CSV: {save_path}")
        
        # Save as JSON for API consumption
        json_save_path = save_path.replace('.csv', '.json')
        with open(json_save_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"✓ Predictions saved to JSON: {json_save_path}")
        
        # Summary statistics
        successful_predictions = [p for p in predictions if p.get('churn_prediction', -1) != -1]
        churn_predictions = [p for p in successful_predictions if p['churn_prediction'] == 1]
        retain_predictions = [p for p in successful_predictions if p['churn_prediction'] == 0]
        
        logger.info("\n" + "="*60)
        logger.info("STREAMING INFERENCE SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Customers Processed: {len(sample_customers)}")
        logger.info(f"Successful Predictions: {len(successful_predictions)}")
        logger.info(f"Predicted to Churn: {len(churn_predictions)} ({len(churn_predictions)/len(successful_predictions)*100:.1f}%)")
        logger.info(f"Predicted to Retain: {len(retain_predictions)} ({len(retain_predictions)/len(successful_predictions)*100:.1f}%)")
        
        if successful_predictions:
            avg_confidence = sum(p['confidence_score'] for p in successful_predictions) / len(successful_predictions)
            avg_time = sum(p['prediction_time_ms'] for p in successful_predictions) / len(successful_predictions)
            logger.info(f"Average Confidence: {avg_confidence:.1f}%")
            logger.info(f"Average Prediction Time: {avg_time:.2f}ms")
        
        logger.info("✓ Streaming inference pipeline completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Streaming inference pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()