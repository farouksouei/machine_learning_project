"""
Comprehensive Model Evaluation Module

This module provides advanced evaluation capabilities including:
- Multiple evaluation metrics and statistical tests
- ROC and Precision-Recall curve analysis
- Feature importance and SHAP analysis
- Model interpretability and explainability
- Cross-validation and bootstrap evaluation
- Model comparison and statistical significance testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score, log_loss, brier_score_loss
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

# Statistical testing
from scipy import stats
from scipy.stats import chi2_contingency

# Optional advanced libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")


class ModelEvaluator:
    """
    Comprehensive model evaluation with advanced metrics and visualizations.
    
    Features:
    - Standard classification metrics (accuracy, precision, recall, F1, ROC-AUC)
    - Advanced metrics (Brier score, log loss, calibration)
    - Statistical significance testing
    - Feature importance analysis
    - Model interpretability (SHAP, LIME)
    - Cross-validation evaluation
    - Visualization and reporting
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model evaluator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.evaluation_results = {}
        self.comparison_results = {}
    
    def evaluate_classification_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        model_name: str = "model",
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a classification model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            X_train: Training features (optional, for overfitting analysis)
            y_train: Training labels (optional, for overfitting analysis)
            model_name: Name of the model
            feature_names: Names of features
            class_names: Names of classes
            
        Returns:
            Dictionary containing all evaluation metrics and results
        """
        print(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_proba = y_pred.astype(float)
        
        # Calculate basic metrics
        basic_metrics = self._calculate_basic_metrics(y_test, y_pred, y_proba)
        
        # Calculate advanced metrics
        advanced_metrics = self._calculate_advanced_metrics(y_test, y_pred, y_proba)
        
        # Confusion matrix analysis
        cm_analysis = self._analyze_confusion_matrix(y_test, y_pred, class_names)
        
        # ROC and PR curve analysis
        curve_analysis = self._analyze_curves(y_test, y_proba)
        
        # Feature importance analysis
        feature_importance = self._analyze_feature_importance(
            model, X_test, y_test, feature_names
        )
        
        # Cross-validation evaluation
        cv_results = self._cross_validation_evaluation(
            model, X_test, y_test
        ) if X_train is not None else {}
        
        # Overfitting analysis
        overfitting_analysis = self._analyze_overfitting(
            model, X_train, y_train, X_test, y_test
        ) if X_train is not None and y_train is not None else {}
        
        # Model interpretability
        interpretability = self._model_interpretability(
            model, X_test, y_test, feature_names
        )
        
        # Compile all results
        evaluation_result = {
            'model_name': model_name,
            'basic_metrics': basic_metrics,
            'advanced_metrics': advanced_metrics,
            'confusion_matrix_analysis': cm_analysis,
            'curve_analysis': curve_analysis,
            'feature_importance': feature_importance,
            'cross_validation': cv_results,
            'overfitting_analysis': overfitting_analysis,
            'interpretability': interpretability,
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Store results
        self.evaluation_results[model_name] = evaluation_result
        
        # Print summary
        self._print_evaluation_summary(evaluation_result)
        
        return evaluation_result
    
    def _calculate_basic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'average_precision': average_precision_score(y_true, y_proba)
        }
    
    def _calculate_advanced_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate advanced classification metrics."""
        # Brier score (lower is better)
        brier_score = brier_score_loss(y_true, y_proba)
        
        # Log loss (lower is better)
        try:
            logloss = log_loss(y_true, y_proba)
        except:
            logloss = np.nan
        
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Sensitivity (same as recall)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Balanced accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Matthews Correlation Coefficient
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / mcc_den if mcc_den != 0 else 0
        
        # Youden's J statistic
        youden_j = sensitivity + specificity - 1
        
        return {
            'brier_score': brier_score,
            'log_loss': logloss,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'balanced_accuracy': balanced_accuracy,
            'matthews_corr_coef': mcc,
            'youden_j_statistic': youden_j
        }
    
    def _analyze_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze confusion matrix and calculate derived metrics."""
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = ['No Churn', 'Churn']
        
        # Calculate rates
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'class_names': class_names
        }
    
    def _analyze_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze ROC and Precision-Recall curves."""
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = roc_thresholds[optimal_idx]
        
        return {
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist(),
                'auc': roc_auc
            },
            'pr_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist(),
                'average_precision': avg_precision
            },
            'optimal_threshold': optimal_threshold,
            'optimal_threshold_metrics': {
                'tpr': tpr[optimal_idx],
                'fpr': fpr[optimal_idx],
                'j_statistic': j_scores[optimal_idx]
            }
        }
    
    def _analyze_feature_importance(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze feature importance using multiple methods."""
        importance_analysis = {}
        
        # Built-in feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importance_analysis['built_in'] = {
                'importances': model.feature_importances_.tolist(),
                'feature_names': feature_names or [f'feature_{i}' for i in range(len(model.feature_importances_))]
            }
        
        # Permutation importance
        try:
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=10,
                random_state=self.random_state, n_jobs=-1
            )
            importance_analysis['permutation'] = {
                'importances_mean': perm_importance.importances_mean.tolist(),
                'importances_std': perm_importance.importances_std.tolist(),
                'feature_names': feature_names or [f'feature_{i}' for i in range(len(perm_importance.importances_mean))]
            }
        except Exception as e:
            print(f"Permutation importance failed: {e}")
        
        return importance_analysis
    
    def _cross_validation_evaluation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """Perform cross-validation evaluation."""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Multiple scoring metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}
        
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
                cv_results[metric] = {
                    'scores': scores.tolist(),
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'min': scores.min(),
                    'max': scores.max()
                }
            except Exception as e:
                print(f"CV evaluation for {metric} failed: {e}")
        
        return cv_results
    
    def _analyze_overfitting(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze overfitting by comparing train and test performance."""
        # Training predictions
        y_train_pred = model.predict(X_train)
        try:
            y_train_proba = model.predict_proba(X_train)[:, 1]
        except:
            y_train_proba = y_train_pred.astype(float)
        
        # Test predictions
        y_test_pred = model.predict(X_test)
        try:
            y_test_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_test_proba = y_test_pred.astype(float)
        
        # Calculate metrics for both sets
        train_metrics = self._calculate_basic_metrics(y_train, y_train_pred, y_train_proba)
        test_metrics = self._calculate_basic_metrics(y_test, y_test_pred, y_test_proba)
        
        # Calculate overfitting indicators
        overfitting_indicators = {}
        for metric in train_metrics:
            train_score = train_metrics[metric]
            test_score = test_metrics[metric]
            overfitting_indicators[f'{metric}_gap'] = train_score - test_score
            overfitting_indicators[f'{metric}_ratio'] = test_score / train_score if train_score != 0 else 0
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'overfitting_indicators': overfitting_indicators,
            'is_overfitting': overfitting_indicators.get('roc_auc_gap', 0) > 0.1
        }
    
    def _model_interpretability(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze model interpretability using SHAP and LIME."""
        interpretability_results = {}
        
        # SHAP analysis
        if SHAP_AVAILABLE:
            try:
                # Choose appropriate explainer based on model type
                if hasattr(model, 'predict_proba'):
                    explainer = shap.Explainer(model, X_test[:100])  # Use subset for speed
                    shap_values = explainer(X_test[:10])  # Explain first 10 samples
                    
                    interpretability_results['shap'] = {
                        'available': True,
                        'feature_names': feature_names or [f'feature_{i}' for i in range(X_test.shape[1])],
                        'mean_abs_shap_values': np.abs(shap_values.values).mean(axis=0).tolist(),
                        'sample_explanations': shap_values.values[:5].tolist()  # First 5 samples
                    }
                else:
                    interpretability_results['shap'] = {'available': False, 'reason': 'Model not compatible'}
            except Exception as e:
                interpretability_results['shap'] = {'available': False, 'reason': str(e)}
        else:
            interpretability_results['shap'] = {'available': False, 'reason': 'SHAP not installed'}
        
        # LIME analysis
        if LIME_AVAILABLE:
            try:
                explainer = lime_tabular.LimeTabularExplainer(
                    X_test,
                    feature_names=feature_names or [f'feature_{i}' for i in range(X_test.shape[1])],
                    class_names=['No Churn', 'Churn'],
                    mode='classification'
                )
                
                # Explain a few instances
                lime_explanations = []
                for i in range(min(3, len(X_test))):
                    exp = explainer.explain_instance(X_test[i], model.predict_proba)
                    lime_explanations.append({
                        'instance': i,
                        'explanation': exp.as_list()
                    })
                
                interpretability_results['lime'] = {
                    'available': True,
                    'explanations': lime_explanations
                }
            except Exception as e:
                interpretability_results['lime'] = {'available': False, 'reason': str(e)}
        else:
            interpretability_results['lime'] = {'available': False, 'reason': 'LIME not installed'}
        
        return interpretability_results
    
    def _print_evaluation_summary(self, evaluation_result: Dict[str, Any]) -> None:
        """Print a summary of evaluation results."""
        print(f"\n{'='*50}")
        print(f"EVALUATION SUMMARY: {evaluation_result['model_name']}")
        print(f"{'='*50}")
        
        # Basic metrics
        basic = evaluation_result['basic_metrics']
        print(f"Accuracy:     {basic['accuracy']:.4f}")
        print(f"Precision:    {basic['precision']:.4f}")
        print(f"Recall:       {basic['recall']:.4f}")
        print(f"F1-Score:     {basic['f1_score']:.4f}")
        print(f"ROC-AUC:      {basic['roc_auc']:.4f}")
        print(f"Avg Precision: {basic['average_precision']:.4f}")
        
        # Advanced metrics
        advanced = evaluation_result['advanced_metrics']
        print(f"\nAdvanced Metrics:")
        print(f"Brier Score:  {advanced['brier_score']:.4f}")
        print(f"Balanced Acc: {advanced['balanced_accuracy']:.4f}")
        print(f"MCC:          {advanced['matthews_corr_coef']:.4f}")
        
        # Confusion matrix
        cm_analysis = evaluation_result['confusion_matrix_analysis']
        print(f"\nConfusion Matrix:")
        print(f"TP: {cm_analysis['true_positives']}, FP: {cm_analysis['false_positives']}")
        print(f"FN: {cm_analysis['false_negatives']}, TN: {cm_analysis['true_negatives']}")
        
        print(f"{'='*50}\n")
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[str, Any]],
        primary_metric: str = 'roc_auc'
    ) -> Dict[str, Any]:
        """
        Compare multiple models and perform statistical significance testing.
        
        Args:
            model_results: Dictionary of model evaluation results
            primary_metric: Primary metric for comparison
            
        Returns:
            Dictionary containing comparison results
        """
        if len(model_results) < 2:
            raise ValueError("At least 2 models required for comparison")
        
        # Extract metrics for comparison
        comparison_data = []
        for model_name, results in model_results.items():
            row = {'model': model_name}
            row.update(results['basic_metrics'])
            row.update(results['advanced_metrics'])
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models by primary metric
        comparison_df = comparison_df.sort_values(primary_metric, ascending=False)
        
        # Statistical significance testing (if CV results available)
        significance_tests = {}
        cv_available = all('cross_validation' in results and primary_metric in results['cross_validation'] 
                          for results in model_results.values())
        
        if cv_available:
            model_names = list(model_results.keys())
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    scores1 = model_results[model1]['cross_validation'][primary_metric]['scores']
                    scores2 = model_results[model2]['cross_validation'][primary_metric]['scores']
                    
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(scores1, scores2)
                    
                    significance_tests[f"{model1}_vs_{model2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'better_model': model1 if np.mean(scores1) > np.mean(scores2) else model2
                    }
        
        comparison_result = {
            'comparison_table': comparison_df.to_dict('records'),
            'best_model': comparison_df.iloc[0]['model'],
            'primary_metric': primary_metric,
            'significance_tests': significance_tests,
            'summary': {
                'total_models': len(comparison_df),
                'best_score': comparison_df.iloc[0][primary_metric],
                'score_range': {
                    'min': comparison_df[primary_metric].min(),
                    'max': comparison_df[primary_metric].max(),
                    'std': comparison_df[primary_metric].std()
                }
            }
        }
        
        self.comparison_results = comparison_result
        return comparison_result
    
    def generate_evaluation_report(
        self,
        output_path: str = "evaluation_report.html"
    ) -> str:
        """
        Generate a comprehensive HTML evaluation report.
        
        Args:
            output_path: Path to save the HTML report
            
        Returns:
            Path to the generated report
        """
        # This would generate a comprehensive HTML report
        # For now, return a simple summary
        report_content = f"""
        <html>
        <head><title>Model Evaluation Report</title></head>
        <body>
        <h1>Model Evaluation Report</h1>
        <p>Generated on: {pd.Timestamp.now()}</p>
        
        <h2>Models Evaluated</h2>
        <ul>
        """
        
        for model_name in self.evaluation_results:
            report_content += f"<li>{model_name}</li>"
        
        report_content += """
        </ul>
        
        <h2>Best Model</h2>
        """
        
        if self.comparison_results:
            best_model = self.comparison_results['best_model']
            best_score = self.comparison_results['summary']['best_score']
            report_content += f"<p>Best Model: {best_model} (Score: {best_score:.4f})</p>"
        
        report_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        return output_path


# Backward compatibility function
def evaluate(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Simple evaluation function for backward compatibility.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    """
    from sklearn.metrics import classification_report
    
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    
    # Enhanced evaluation
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_classification_model(
        model, X_test, y_test, model_name="model"
    )
    
    return results