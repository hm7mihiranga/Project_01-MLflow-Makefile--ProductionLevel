import os
import joblib
import logging
import time
from typing import Any, Dict, Optional, Union, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    log_loss
)

# Configure warnings and logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(
            self,
            model: BaseEstimator,
            model_name: str,
            metrics: Optional[List[str]] = None
    ):
        self.model = model
        self.model_name = model_name
        self.evaluation_results = {}
        
        if metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1", "auc", "log_loss"]
        else:
            self.metrics = metrics
            
        logger.info(f"ModelEvaluator initialized for model: {self.model_name}")
        logger.info(f"Available metrics: {', '.join(self.metrics)}")

    def evaluate(
            self,
            X_test: Union[pd.DataFrame, np.ndarray],
            Y_test: Union[pd.Series, np.ndarray],
            save_results: bool = True,
            save_path: Optional[str] = None
    ) -> Dict[str, Any]:

        logger.info(f"\n{'='*60}")
        logger.info("MODEL EVALUATION")
        logger.info(f"{'='*60}")
        
        # Input validation
        if X_test is None or Y_test is None:
            logger.error("✗ Test data cannot be None")
            raise ValueError("Test data cannot be None")
            
        if len(X_test) == 0 or len(Y_test) == 0:
            logger.error("✗ Test data cannot be empty")
            raise ValueError("Test data cannot be empty")
            
        if len(X_test) != len(Y_test):
            logger.error(f"✗ Feature and target length mismatch: {len(X_test)} vs {len(Y_test)}")
            raise ValueError(f"Feature and target length mismatch: {len(X_test)} vs {len(Y_test)}")
        
        try:
            # Log evaluation information
            logger.info(f"Evaluation Configuration:")
            logger.info(f"  • Model: {self.model_name}")
            logger.info(f"  • Test Samples: {len(X_test):,}")
            logger.info(f"  • Features: {X_test.shape[1] if hasattr(X_test, 'shape') else 'Unknown'}")
            logger.info(f"  • Target Distribution: {np.bincount(Y_test)}")
            
            # Start evaluation
            logger.info("Starting model evaluation...")
            start_time = time.time()
            
            # Make predictions
            logger.info("Generating predictions...")
            Y_pred = self.model.predict(X_test)
            
            # Get prediction probabilities if available
            Y_pred_proba = None
            if hasattr(self.model, "predict_proba"):
                Y_pred_proba = self.model.predict_proba(X_test)
                logger.info("✓ Prediction probabilities obtained")
            
            # Calculate confusion matrix
            cm = confusion_matrix(Y_test, Y_pred)
            
            # Calculate all metrics
            results = {
                'model_name': self.model_name,
                'evaluation_timestamp': datetime.now().isoformat(),
                'test_samples': len(X_test),
                'confusion_matrix': cm
            }
            
            # Basic classification metrics
            if "accuracy" in self.metrics:
                accuracy = accuracy_score(Y_test, Y_pred)
                results['accuracy'] = accuracy
                logger.info(f"  • Accuracy: {accuracy:.4f}")
            
            if "precision" in self.metrics:
                precision = precision_score(Y_test, Y_pred, average='binary' if len(np.unique(Y_test)) == 2 else 'macro')
                results['precision'] = precision
                logger.info(f"  • Precision: {precision:.4f}")
            
            if "recall" in self.metrics:
                recall = recall_score(Y_test, Y_pred, average='binary' if len(np.unique(Y_test)) == 2 else 'macro')
                results['recall'] = recall
                logger.info(f"  • Recall: {recall:.4f}")
            
            if "f1" in self.metrics:
                f1 = f1_score(Y_test, Y_pred, average='binary' if len(np.unique(Y_test)) == 2 else 'macro')
                results['f1'] = f1
                logger.info(f"  • F1-Score: {f1:.4f}")
            
            # Probabilistic metrics
            if Y_pred_proba is not None:
                if "auc" in self.metrics and len(np.unique(Y_test)) == 2:
                    auc = roc_auc_score(Y_test, Y_pred_proba[:, 1])
                    results['auc'] = auc
                    logger.info(f"  • AUC-ROC: {auc:.4f}")
                
                if "log_loss" in self.metrics:
                    logloss = log_loss(Y_test, Y_pred_proba)
                    results['log_loss'] = logloss
                    logger.info(f"  • Log Loss: {logloss:.4f}")
                
                if "average_precision" in self.metrics and len(np.unique(Y_test)) == 2:
                    avg_precision = average_precision_score(Y_test, Y_pred_proba[:, 1])
                    results['average_precision'] = avg_precision
                    logger.info(f"  • Average Precision: {avg_precision:.4f}")
            
            # Classification report
            class_report = classification_report(Y_test, Y_pred, output_dict=True)
            results['classification_report'] = class_report
            
            evaluation_time = time.time() - start_time
            logger.info(f"✓ Evaluation completed in {evaluation_time:.2f} seconds")
            
            # Store results
            self.evaluation_results = results
            
            # Save results if requested
            if save_results:
                self._save_evaluation_artifacts(results, Y_test, Y_pred, Y_pred_proba, save_path)
            
            logger.info("Model evaluation successful!")
            logger.info(f"{'='*60}\n")
            
            return results
            
        except Exception as e:
            logger.error(f"✗ Model evaluation failed: {str(e)}")
            raise

    def _save_evaluation_artifacts(
            self,
            results: Dict[str, Any],
            Y_test: np.ndarray,
            Y_pred: np.ndarray,
            Y_pred_proba: Optional[np.ndarray] = None,
            save_path: Optional[str] = None
    ) -> None:
        logger.info("Saving evaluation artifacts...")
        
        if save_path is None:
            save_path = f"artifacts/evaluation"
        
        # Create directory structure
        os.makedirs(save_path, exist_ok=True)
        
        # Save evaluation report
        report_path = os.path.join(save_path, f"{self.model_name}_evaluation_report.txt")
        self._save_evaluation_report(results, report_path)
        
        # Save classification report
        class_report_path = os.path.join(save_path, f"{self.model_name}_classification_report.txt")
        class_report_str = classification_report(Y_test, Y_pred)
        with open(class_report_path, 'w') as f:
            f.write(f"Classification Report for {self.model_name}\n")
            f.write(f"{'='*50}\n")
            f.write(class_report_str)
        
        # Save confusion matrix plot
        cm_path = os.path.join(save_path, f"{self.model_name}_confusion_matrix.png")
        self._plot_confusion_matrix(Y_test, Y_pred, cm_path)
        
        # Save ROC curve if binary classification and probabilities available
        if Y_pred_proba is not None and len(np.unique(Y_test)) == 2:
            roc_path = os.path.join(save_path, f"{self.model_name}_roc_curve.png")
            self._plot_roc_curve(Y_test, Y_pred_proba[:, 1], roc_path)
            
            pr_path = os.path.join(save_path, f"{self.model_name}_precision_recall_curve.png")
            self._plot_precision_recall_curve(Y_test, Y_pred_proba[:, 1], pr_path)
        
        logger.info(f"✓ Evaluation artifacts saved to: {save_path}")

    def _save_evaluation_report(self, results: Dict[str, Any], filepath: str) -> None:
        """Save comprehensive evaluation report to file."""
        try:
            with open(filepath, 'w') as f:
                f.write(f"Model Evaluation Report\n")
                f.write(f"{'='*50}\n")
                f.write(f"Model Name: {results.get('model_name', 'Unknown')}\n")
                f.write(f"Evaluation Date: {results.get('evaluation_timestamp', 'Unknown')}\n")
                f.write(f"Test Samples: {results.get('test_samples', 'Unknown'):,}\n")
                f.write(f"\nMetrics:\n")
                f.write(f"---------\n")
                
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'log_loss', 'average_precision']:
                    if metric in results:
                        f.write(f"{metric.capitalize()}: {results[metric]:.4f}\n")
                
                f.write(f"\nConfusion Matrix:\n")
                f.write(f"-----------------\n")
                cm = results.get('confusion_matrix', [])
                for row in cm:
                    f.write(f"{row}\n")
            
            logger.info(f"✓ Evaluation report saved: {filepath}")
            
        except Exception as e:
            logger.error(f"✗ Failed to save evaluation report: {str(e)}")

    def _plot_confusion_matrix(self, Y_test: np.ndarray, Y_pred: np.ndarray, save_path: str) -> None:
        """Plot and save confusion matrix."""
        try:
            cm = confusion_matrix(Y_test, Y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
            plt.title(f'Confusion Matrix - {self.model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ Confusion matrix plot saved: {save_path}")
            
        except Exception as e:
            logger.error(f"✗ Failed to save confusion matrix plot: {str(e)}")

    def _plot_roc_curve(self, Y_test: np.ndarray, Y_pred_proba: np.ndarray, save_path: str) -> None:
        try:
            fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba)
            auc_score = roc_auc_score(Y_test, Y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.model_name}')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ ROC curve plot saved: {save_path}")
            
        except Exception as e:
            logger.error(f"✗ Failed to save ROC curve plot: {str(e)}")

    def _plot_precision_recall_curve(self, Y_test: np.ndarray, Y_pred_proba: np.ndarray, save_path: str) -> None:
        try:
            precision, recall, _ = precision_recall_curve(Y_test, Y_pred_proba)
            avg_precision = average_precision_score(Y_test, Y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {self.model_name}')
            plt.legend(loc="lower left")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ Precision-Recall curve plot saved: {save_path}")
            
        except Exception as e:
            logger.error(f"✗ Failed to save Precision-Recall curve plot: {str(e)}")

    def get_evaluation_summary(self) -> str:
        if not self.evaluation_results:
            return "No evaluation results available. Please run evaluate() first."
        
        results = self.evaluation_results
        summary = f"\nModel Evaluation Summary - {self.model_name}\n"
        summary += "="*50 + "\n"
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'log_loss']:
            if metric in results:
                summary += f"{metric.capitalize()}: {results[metric]:.4f}\n"
        
        summary += f"Test Samples: {results.get('test_samples', 'Unknown'):,}\n"
        
        return summary

    def compare_models(self, other_evaluator: 'ModelEvaluator') -> Dict[str, Any]:
        if not self.evaluation_results or not other_evaluator.evaluation_results:
            raise ValueError("Both models must be evaluated before comparison")
        
        logger.info(f"Comparing {self.model_name} vs {other_evaluator.model_name}")
        
        comparison = {
            'model_1': self.model_name,
            'model_2': other_evaluator.model_name,
            'comparison': {}
        }
        
        # Compare common metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            if metric in self.evaluation_results and metric in other_evaluator.evaluation_results:
                val1 = self.evaluation_results[metric]
                val2 = other_evaluator.evaluation_results[metric]
                difference = val1 - val2
                winner = self.model_name if val1 > val2 else other_evaluator.model_name
                
                comparison['comparison'][metric] = {
                    'model_1_value': val1,
                    'model_2_value': val2,
                    'difference': difference,
                    'better_model': winner
                }
        
        return comparison


# Example usage and testing functionality
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    logger.info("Generating sample data for testing...")
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple model
    logger.info("Training sample model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create evaluator and evaluate
    logger.info("Creating ModelEvaluator and evaluating...")
    evaluator = ModelEvaluator(
        model=model,
        model_name="test_random_forest"
    )
    
    # Perform comprehensive evaluation
    results = evaluator.evaluate(X_test, y_test, save_results=True)
    
    # Print summary
    print(evaluator.get_evaluation_summary())
    
    logger.info("✓ ModelEvaluator test completed successfully!")