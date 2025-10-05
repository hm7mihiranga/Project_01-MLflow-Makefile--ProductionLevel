"""
Quick test script to validate the Telco Churn Analysis project setup.

This script performs basic validation of the project structure, 
configuration, and core functionality.
"""

import os
import sys
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_project_structure() -> bool:
    """
    Test if the project structure is correctly set up.
    
    Returns:
        bool: True if all required directories and files exist
    """
    logger.info("Testing project structure...")
    
    required_dirs = [
        'src',
        'utils', 
        'pipelines',
        'artifacts',
        'artifacts/data',
        'artifacts/models',
        'artifacts/evaluation',
        'artifacts/predictions',
        'artifacts/encode'
    ]
    
    required_files = [
        'config.yaml',
        'requirements.txt',
        'Makefile'
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_dirs:
        logger.error(f"Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return False
    
    logger.info("✓ Project structure validation passed")
    return True


def test_configuration() -> bool:
    """
    Test if configuration can be loaded properly.
    
    Returns:
        bool: True if configuration loads successfully
    """
    logger.info("Testing configuration loading...")
    
    try:
        # Add utils to path
        sys.path.append('utils')
        import config
        
        # Test basic config loading
        config_data = config.get_config()
        data_paths = config.get_data_paths()
        columns_config = config.get_columns()
        
        # Validate required config sections
        required_sections = ['data_paths', 'columns', 'training']
        missing_sections = []
        
        for section in required_sections:
            if section not in config_data:
                missing_sections.append(section)
        
        if missing_sections:
            logger.error(f"Missing configuration sections: {missing_sections}")
            return False
        
        logger.info("✓ Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration loading failed: {str(e)}")
        return False


def test_imports() -> bool:
    """
    Test if all required modules can be imported.
    
    Returns:
        bool: True if all imports are successful
    """
    logger.info("Testing module imports...")
    
    try:
        # Test core libraries
        import pandas
        import numpy
        import sklearn
        import matplotlib
        import seaborn
        import yaml
        import joblib
        
        logger.info("✓ Core libraries import successful")
        
        # Test ML libraries
        try:
            import xgboost
            import lightgbm
            logger.info("✓ Boosting libraries import successful")
        except ImportError as e:
            logger.warning(f"⚠ Some ML libraries missing: {str(e)}")
        
        # Test project modules
        sys.path.append('src')
        sys.path.append('utils')
        
        try:
            from data_ingestion import DataIngestorCSV
            from model_inference import ModelInference
            import config
            logger.info("✓ Project modules import successful")
        except ImportError as e:
            logger.error(f"Project modules import failed: {str(e)}")
            return False
        
        return True
        
    except ImportError as e:
        logger.error(f"Core library import failed: {str(e)}")
        return False


def test_data_availability() -> bool:
    """
    Test if required data files are available.
    
    Returns:
        bool: True if data files exist
    """
    logger.info("Testing data availability...")
    
    # Check for raw data
    raw_data_file = 'Telco-Customer-Churn.csv'
    if not os.path.exists(raw_data_file):
        logger.warning(f"⚠ Raw data file missing: {raw_data_file}")
        logger.info("  This is expected if you haven't downloaded the dataset yet")
    else:
        logger.info(f"✓ Raw data file found: {raw_data_file}")
    
    # Check for processed data (optional)
    processed_files = [
        'artifacts/data/X_train.csv',
        'artifacts/data/X_test.csv', 
        'artifacts/data/Y_train.csv',
        'artifacts/data/Y_test.csv'
    ]
    
    processed_count = 0
    for file_path in processed_files:
        if os.path.exists(file_path):
            processed_count += 1
    
    if processed_count == len(processed_files):
        logger.info("✓ All processed data files found")
    elif processed_count > 0:
        logger.info(f"✓ {processed_count}/{len(processed_files)} processed data files found")
    else:
        logger.info("ℹ No processed data files found (run data pipeline to generate)")
    
    return True


def test_models_availability() -> bool:
    """
    Test if trained models are available.
    
    Returns:
        bool: Always returns True (models are optional for testing)
    """
    logger.info("Testing model availability...")
    
    models_dir = 'artifacts/models'
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        if model_files:
            logger.info(f"✓ {len(model_files)} trained models found")
            for model_file in model_files[:3]:  # Show first 3
                logger.info(f"  • {model_file}")
        else:
            logger.info("ℹ No trained models found (run training pipeline to generate)")
    else:
        logger.info("ℹ Models directory not found (run training pipeline to generate)")
    
    return True


def run_quick_validation() -> Dict[str, bool]:
    """
    Run all validation tests.
    
    Returns:
        Dict[str, bool]: Results of each test
    """
    logger.info("="*60)
    logger.info("TELCO CHURN ANALYSIS - PROJECT VALIDATION")
    logger.info("="*60)
    
    results = {
        'project_structure': test_project_structure(),
        'configuration': test_configuration(),
        'imports': test_imports(),
        'data_availability': test_data_availability(),
        'models_availability': test_models_availability()
    }
    
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    passed_tests = 0
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name.replace('_', ' ').title():<20}: {status}")
        if result:
            passed_tests += 1
    
    logger.info("-" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        logger.info("\n🎉 ALL TESTS PASSED! Project is ready for use.")
    else:
        logger.info(f"\n⚠ {total_tests - passed_tests} tests failed. Please check the issues above.")
    
    logger.info("="*60)
    
    return results


def main():
    """
    Main function to run project validation.
    """
    try:
        results = run_quick_validation()
        
        # Provide recommendations based on results
        if not results['imports']:
            print("\n💡 RECOMMENDATION:")
            print("   Run 'make install' to install required dependencies")
        
        if results['project_structure'] and results['configuration'] and results['imports']:
            print("\n🚀 NEXT STEPS:")
            print("   1. Run 'make data-pipeline' to process the data")
            print("   2. Run 'make train-pipeline' to train models") 
            print("   3. Run 'make streaming-inference' to test inference")
            print("   4. Or run 'make run-all' to execute complete pipeline")
        
    except Exception as e:
        logger.error(f"Validation script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()