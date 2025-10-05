import os
import sys
import logging
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from model_building import ModelFactory
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from config import get_model_config

# MLflow imports (optional - will work without MLflow if not installed)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Initialize MLflow tracking if available"""
    if not MLFLOW_AVAILABLE:
        logger.info("MLflow not available - running without experiment tracking")
        return None
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Set experiment name
        experiment_name = "telco_churn_prediction"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
            experiment_name = "Default"
        
        mlflow.set_experiment(experiment_name)
        return experiment_name
    except Exception as e:
        logger.error(f"Failed to setup MLflow: {e}")
        return None

def main():
    logger.info("="*60)
    logger.info("TELCO CHURN MODEL TRAINING PIPELINE WITH MLFLOW")
    logger.info("="*60)
    
    try:
        # Setup MLflow tracking
        experiment_name = setup_mlflow()
        
        # Load configuration
        model_config = get_model_config()
        model_types = model_config.get('model_types', {})
        default_metrics = ["accuracy", "precision", "recall", "f1", "auc", "log_loss"]
        
        logger.info(f"Configuration loaded:")
        if experiment_name:
            logger.info(f"  • MLflow Experiment: {experiment_name}")
        logger.info(f"  • Models to train: {list(model_types.keys())}")
        logger.info(f"  • Evaluation metrics: {default_metrics}")
        
        # Load training and test data
        logger.info("Loading training and test data...")
        X_train = pd.read_csv("artifacts/data/X_train.csv")
        X_test = pd.read_csv("artifacts/data/X_test.csv")
        y_train = pd.read_csv("artifacts/data/Y_train.csv").squeeze()
        y_test = pd.read_csv("artifacts/data/Y_test.csv").squeeze()
        
        data_info = {
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "features": X_train.shape[1],
            "positive_class_ratio": (y_train == 1).mean()
        }
        
        logger.info(f"Data loaded:")
        logger.info(f"  • Training set: {X_train.shape}")
        logger.info(f"  • Test set: {X_test.shape}")
        logger.info(f"  • Target distribution: {y_train.value_counts().to_dict()}")
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Train and evaluate each model
        trained_models = {}
        
        for model_type, model_info in model_types.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"PROCESSING MODEL: {model_type.upper()}")
            logger.info(f"{'='*50}")
            
            # Start MLflow run for this model if available
            mlflow_context = None
            if MLFLOW_AVAILABLE and experiment_name:
                try:
                    mlflow_context = mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                except Exception as e:
                    logger.warning(f"Failed to start MLflow run: {e}")
            
            try:
                # Log dataset information to MLflow
                if mlflow_context:
                    try:
                        mlflow.log_params(data_info)
                    except Exception as e:
                        logger.warning(f"Failed to log data params to MLflow: {e}")
                
                # Build model
                params = model_info.get('params', {})
                logger.info(f"Building {model_type} with params: {params}")
                model = ModelFactory.get_model(model_type, params)
                
                # Log model parameters to MLflow
                if mlflow_context:
                    try:
                        mlflow.log_params({f"model_{k}": v for k, v in params.items()})
                        mlflow.log_param("model_type", model_type)
                    except Exception as e:
                        logger.warning(f"Failed to log model params to MLflow: {e}")
                
                # Train model using enhanced trainer
                logger.info(f"Training {model_type}...")
                train_result = trainer.train(model, X_train, y_train, model_type)
                
                # Handle different return formats from ModelTrainer
                if isinstance(train_result, tuple) and len(train_result) == 3:
                    trained_model, train_score, training_info = train_result
                elif isinstance(train_result, tuple) and len(train_result) == 2:
                    trained_model, train_score = train_result
                    training_info = {}
                else:
                    logger.error(f"Unexpected return format from trainer for {model_type}")
                    continue
                    
                logger.info(f"Training completed - Score: {train_score:.4f}")
                
                # Log training metrics to MLflow
                if mlflow_context:
                    try:
                        mlflow.log_metric("train_cv_score", train_score)
                        # Log additional training information if available
                        if training_info:
                            if 'cv_mean' in training_info:
                                mlflow.log_metric("cv_mean_score", training_info['cv_mean'])
                            if 'cv_std' in training_info:
                                mlflow.log_metric("cv_std_score", training_info['cv_std'])
                            if 'training_time' in training_info:
                                mlflow.log_metric("training_time_seconds", training_info['training_time'])
                    except Exception as e:
                        logger.warning(f"Failed to log training metrics to MLflow: {e}")
                
                # Save trained model
                model_path = f"artifacts/models/{model_type}_cv_model.pkl"
                logger.info(f"Saving model to: {model_path}")
                trainer.save_model(trained_model, model_path)
                
                # Create model-specific evaluator
                logger.info(f"Creating evaluator for {model_type}...")
                evaluator = ModelEvaluator(
                    model=trained_model,
                    model_name=f"{model_type}_cv_model",
                    metrics=default_metrics
                )
                
                # Evaluate model with comprehensive metrics
                logger.info(f"Evaluating {model_type}...")
                evaluation_path = f"artifacts/evaluation/{model_type}_cv_evaluation_report"
                
                eval_results = evaluator.evaluate(
                    X_test=X_test,
                    Y_test=y_test,
                    save_results=True,
                    save_path=evaluation_path
                )
                
                logger.info(f"Evaluation completed for {model_type}")
                logger.info(f"Results: {eval_results}")
                
                # Log evaluation metrics to MLflow
                if mlflow_context:
                    try:
                        for metric_name, metric_value in eval_results.items():
                            if isinstance(metric_value, (int, float)):
                                mlflow.log_metric(f"test_{metric_name}", metric_value)
                        
                        # Log model artifacts
                        mlflow.sklearn.log_model(
                            sk_model=trained_model,
                            artifact_path="model",
                            registered_model_name=f"telco_churn_{model_type}"
                        )
                        
                        # Log evaluation artifacts
                        mlflow.log_artifacts(f"artifacts/evaluation", artifact_path="evaluation")
                        
                        # Add model tags
                        mlflow.set_tags({
                            "model_type": model_type,
                            "project": "telco_churn_prediction",
                            "version": "1.0",
                            "framework": "scikit-learn"
                        })
                    except Exception as e:
                        logger.warning(f"Failed to log evaluation results to MLflow: {e}")
                
                # Store trained model for comparison
                run_id = None
                if mlflow_context:
                    try:
                        run_id = mlflow.active_run().info.run_id
                    except:
                        pass
                        
                trained_models[model_type] = {
                    'model': trained_model,
                    'evaluator': evaluator,
                    'train_score': train_score,
                    'eval_results': eval_results,
                    'run_id': run_id,
                    'training_info': training_info if 'training_info' in locals() else {}
                }
                
                # Display evaluation summary
                logger.info(f"\n{model_type} Evaluation Summary:")
                logger.info(evaluator.get_evaluation_summary())
                
                if mlflow_context and run_id:
                    logger.info(f"✓ MLflow run completed for {model_type}")
                    logger.info(f"  Run ID: {run_id}")
                
            except Exception as e:
                logger.error(f"Failed to train/evaluate {model_type}: {str(e)}")
                if mlflow_context:
                    try:
                        mlflow.log_param("error", str(e))
                    except:
                        pass
                continue
            finally:
                # End MLflow run
                if mlflow_context:
                    try:
                        mlflow.end_run()
                    except:
                        pass
        
        # Model comparison if multiple models were trained
        if len(trained_models) > 1:
            logger.info(f"\n{'='*60}")
            logger.info("MODEL COMPARISON")
            logger.info(f"{'='*60}")
            
            # Compare models pairwise
            model_names = list(trained_models.keys())
            for i in range(len(model_names) - 1):
                model1_name = model_names[i]
                model2_name = model_names[i + 1]
                
                evaluator1 = trained_models[model1_name]['evaluator']
                evaluator2 = trained_models[model2_name]['evaluator']
                
                try:
                    comparison = evaluator1.compare_models(evaluator2)
                    
                    logger.info(f"\nComparison: {comparison['model_1']} vs {comparison['model_2']}")
                    logger.info("-" * 50)
                    for metric, comp_data in comparison['comparison'].items():
                        logger.info(f"{metric.capitalize()}:")
                        logger.info(f"  {comparison['model_1']}: {comp_data['model_1_value']:.4f}")
                        logger.info(f"  {comparison['model_2']}: {comp_data['model_2_value']:.4f}")
                        logger.info(f"  Better: {comp_data['better_model']}")
                        
                except Exception as e:
                    logger.warning(f"Failed to compare {model1_name} vs {model2_name}: {str(e)}")
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING PIPELINE SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Successfully trained models: {len(trained_models)}")
        if experiment_name:
            logger.info(f"MLflow Experiment: {experiment_name}")
            logger.info(f"MLflow UI: http://localhost:5000")
        
        for model_name, model_data in trained_models.items():
            logger.info(f"  • {model_name}:")
            logger.info(f"    - Training Score: {model_data['train_score']:.4f}")
            if model_data.get('run_id'):
                logger.info(f"    - MLflow Run ID: {model_data['run_id']}")
            eval_results = model_data['eval_results']
            if 'accuracy' in eval_results:
                logger.info(f"    - Test Accuracy: {eval_results['accuracy']:.4f}")
            if 'f1' in eval_results:
                logger.info(f"    - F1 Score: {eval_results['f1']:.4f}")
        
        logger.info("\n" + "="*60)
        logger.info("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        if experiment_name:
            logger.info("Next steps:")
            logger.info("1. Run 'make mlflow-ui' to view experiment results")
            logger.info("2. Visit http://localhost:5000 to explore MLflow UI")
            logger.info("3. Compare models and select the best performer")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()