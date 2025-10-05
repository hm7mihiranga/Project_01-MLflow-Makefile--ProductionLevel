"""
Test script for the Telco Customer Churn Prediction API.

This script tests the deployed API endpoints with sample data.
"""

import requests
import json
import time
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API configuration
BASE_URL = "http://localhost:8080"
HEADERS = {"Content-Type": "application/json"}


def create_sample_customer_data() -> Dict:
    """Create sample customer data for testing."""
    return {
        "customerID": "TEST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 50.50,
        "TotalCharges": "606.00"
    }


def create_batch_customer_data() -> List[Dict]:
    """Create batch customer data for testing."""
    return [
        {
            "customerID": "BATCH-001",
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.25,
            "TotalCharges": "170.50"
        },
        {
            "customerID": "BATCH-002",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 36,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 65.75,
            "TotalCharges": "2367.00"
        }
    ]


def test_health_check():
    """Test the health check endpoint."""
    logger.info("Testing health check endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Health check passed: {data}")
            return True
        else:
            logger.error(f"✗ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Health check failed: {str(e)}")
        return False


def test_root_endpoint():
    """Test the root endpoint."""
    logger.info("Testing root endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Root endpoint response: {data}")
            return True
        else:
            logger.error(f"✗ Root endpoint failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Root endpoint failed: {str(e)}")
        return False


def test_model_info():
    """Test the model info endpoint."""
    logger.info("Testing model info endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/models/info", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Model info: {data}")
            return True
        else:
            logger.error(f"✗ Model info failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Model info failed: {str(e)}")
        return False


def test_single_prediction():
    """Test single customer prediction."""
    logger.info("Testing single customer prediction...")
    
    try:
        customer_data = create_sample_customer_data()
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict",
            json=customer_data,
            headers=HEADERS,
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Single prediction successful:")
            logger.info(f"  Customer ID: {data['customer_id']}")
            logger.info(f"  Churn Status: {data['churn_status']}")
            logger.info(f"  Churn Probability: {data['churn_probability']:.1%}")
            logger.info(f"  Risk Level: {data['risk_level']}")
            logger.info(f"  Recommendation: {data['recommendation']}")
            logger.info(f"  Response Time: {(end_time - start_time)*1000:.2f}ms")
            return True
        else:
            logger.error(f"✗ Single prediction failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Single prediction failed: {str(e)}")
        return False


def test_batch_prediction():
    """Test batch customer prediction."""
    logger.info("Testing batch customer prediction...")
    
    try:
        batch_data = {
            "customers": create_batch_customer_data()
        }
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=batch_data,
            headers=HEADERS,
            timeout=60
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Batch prediction successful:")
            logger.info(f"  Total Customers: {data['total_customers']}")
            logger.info(f"  Successful Predictions: {data['successful_predictions']}")
            logger.info(f"  Response Time: {(end_time - start_time)*1000:.2f}ms")
            
            # Show individual results
            for i, pred in enumerate(data['predictions'][:3], 1):  # Show first 3
                logger.info(f"  Customer {i}: {pred['customer_id']} - {pred['churn_status']} ({pred['confidence_score']:.1f}% confidence)")
            
            return True
        else:
            logger.error(f"✗ Batch prediction failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Batch prediction failed: {str(e)}")
        return False


def test_invalid_data():
    """Test API with invalid data to check error handling."""
    logger.info("Testing error handling with invalid data...")
    
    try:
        invalid_data = {
            "customerID": "INVALID-001",
            "gender": "Invalid",  # Invalid value
            "SeniorCitizen": 2,   # Invalid range
            "tenure": -1          # Invalid value
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data,
            headers=HEADERS,
            timeout=10
        )
        
        if response.status_code == 422:  # Validation error expected
            logger.info("✓ Error handling working correctly - validation error caught")
            return True
        else:
            logger.warning(f"⚠ Unexpected response for invalid data: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Error handling test failed: {str(e)}")
        return False


def run_comprehensive_test():
    """Run comprehensive API testing."""
    logger.info("="*60)
    logger.info("TELCO CHURN API COMPREHENSIVE TESTING")
    logger.info("="*60)
    
    # Test results
    tests = {
        "Root Endpoint": test_root_endpoint,
        "Health Check": test_health_check,
        "Model Info": test_model_info,
        "Single Prediction": test_single_prediction,
        "Batch Prediction": test_batch_prediction,
        "Error Handling": test_invalid_data
    }
    
    results = {}
    passed_tests = 0
    
    for test_name, test_func in tests.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"RUNNING: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed_tests += 1
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {str(e)}")
            results[test_name] = False
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("API TESTING SUMMARY")
    logger.info(f"{'='*60}")
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:<20}: {status}")
    
    logger.info(f"\nTotal Tests: {len(tests)}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {len(tests) - passed_tests}")
    
    if passed_tests == len(tests):
        logger.info("\n🎉 ALL API TESTS PASSED!")
        logger.info("The Telco Churn Prediction API is working correctly.")
    else:
        logger.info(f"\n⚠ {len(tests) - passed_tests} tests failed.")
        logger.info("Please check the API server and model loading.")
    
    logger.info("="*60)
    
    return results


def main():
    """Main function to run API tests."""
    try:
        # Check if API server is running
        logger.info("Checking if API server is running...")
        
        try:
            response = requests.get(f"{BASE_URL}/", timeout=5)
            logger.info("✓ API server is responding")
        except requests.exceptions.RequestException:
            logger.error("✗ API server is not responding")
            logger.error(f"Please make sure the API server is running at {BASE_URL}")
            logger.error("Run 'make deploy-local' to start the API server")
            return False
        
        # Run comprehensive tests
        results = run_comprehensive_test()
        
        # Return overall success
        return all(results.values())
        
    except KeyboardInterrupt:
        logger.info("\nTesting interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Testing failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)