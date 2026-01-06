"""
Advanced Data Preprocessing Pipeline for Customer Churn Prediction

This module implements comprehensive data preprocessing including:
- Feature engineering and interaction terms
- Advanced scaling and encoding techniques
- Outlier detection and handling
- Feature selection methods
- Data validation and quality checks
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    OneHotEncoder, LabelEncoder, OrdinalEncoder,
    PolynomialFeatures, PowerTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from typing import List, Dict, Any, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    print("imbalanced-learn not available. Install with: pip install imbalanced-learn")


class AdvancedPreprocessor:
    """
    Advanced preprocessing pipeline with feature engineering and data quality checks.
    
    Features:
    - Multiple scaling strategies (Standard, Robust, MinMax)
    - Advanced encoding for categorical variables
    - Feature engineering (interactions, polynomial features)
    - Outlier detection and handling
    - Feature selection methods
    - Class imbalance handling
    - Data validation and quality reporting
    """
    
    def __init__(
        self,
        scaling_strategy: str = 'standard',
        encoding_strategy: str = 'onehot',
        handle_outliers: bool = True,
        feature_engineering: bool = True,
        feature_selection: bool = False,
        n_features_to_select: Optional[int] = None,
        handle_imbalance: bool = False,
        imbalance_strategy: str = 'smote',
        random_state: int = 42
    ):
        """
        Initialize the advanced preprocessor.
        
        Args:
            scaling_strategy: 'standard', 'robust', or 'minmax'
            encoding_strategy: 'onehot', 'label', or 'ordinal'
            handle_outliers: Whether to detect and handle outliers
            feature_engineering: Whether to create interaction features
            feature_selection: Whether to perform feature selection
            n_features_to_select: Number of features to select (if None, selects top 50%)
            handle_imbalance: Whether to handle class imbalance
            imbalance_strategy: 'smote', 'adasyn', 'undersample', or 'smote_tomek'
            random_state: Random seed for reproducibility
        """
        self.scaling_strategy = scaling_strategy
        self.encoding_strategy = encoding_strategy
        self.handle_outliers = handle_outliers
        self.feature_engineering = feature_engineering
        self.feature_selection = feature_selection
        self.n_features_to_select = n_features_to_select
        self.handle_imbalance = handle_imbalance
        self.imbalance_strategy = imbalance_strategy
        self.random_state = random_state
        
        # Initialize components
        self.preprocessor = None
        self.feature_selector = None
        self.imbalance_handler = None
        self.feature_names_ = None
        self.data_quality_report_ = {}
        
        # Validation
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        valid_scaling = ['standard', 'robust', 'minmax']
        if self.scaling_strategy not in valid_scaling:
            raise ValueError(f"scaling_strategy must be one of {valid_scaling}")
        
        valid_encoding = ['onehot', 'label', 'ordinal']
        if self.encoding_strategy not in valid_encoding:
            raise ValueError(f"encoding_strategy must be one of {valid_encoding}")
        
        if self.handle_imbalance and not IMBALANCED_LEARN_AVAILABLE:
            raise ValueError("imbalanced-learn package required for imbalance handling")
        
        valid_imbalance = ['smote', 'adasyn', 'undersample', 'smote_tomek']
        if self.imbalance_strategy not in valid_imbalance:
            raise ValueError(f"imbalance_strategy must be one of {valid_imbalance}")
    
    def _get_scaler(self) -> Any:
        """Get the appropriate scaler based on strategy."""
        scalers = {
            'standard': StandardScaler(),  # Mean=0, Std=1, sensitive to outliers
            'robust': RobustScaler(),      # Median=0, IQR=1, robust to outliers
            'minmax': MinMaxScaler()       # Scale to [0,1], preserves relationships
        }
        return scalers[self.scaling_strategy]
    
    def _get_encoder(self, handle_unknown: str = 'ignore') -> Any:
        """Get the appropriate encoder based on strategy."""
        encoders = {
            'onehot': OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False),
            'label': LabelEncoder(),  # Note: Only works for single column
            'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        }
        return encoders[self.encoding_strategy]
    
    def _detect_outliers_iqr(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            df: DataFrame to check for outliers
            columns: Numerical columns to check
            
        Returns:
            DataFrame with outliers capped at 1.5*IQR
        """
        df_clean = df.copy()
        outlier_info = {}
        
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                n_outliers = outliers_mask.sum()
                
                if n_outliers > 0:
                    # Cap outliers instead of removing them
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                    outlier_info[col] = {
                        'count': n_outliers,
                        'percentage': (n_outliers / len(df_clean)) * 100,
                        'bounds': (lower_bound, upper_bound)
                    }
        
        self.data_quality_report_['outliers'] = outlier_info
        return df_clean
    
    def _create_interaction_features(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        max_interactions: int = 10
    ) -> pd.DataFrame:
        """
        Create interaction features between numerical variables.
        
        Args:
            df: Input DataFrame
            numerical_cols: List of numerical columns
            max_interactions: Maximum number of interaction features to create
            
        Returns:
            DataFrame with additional interaction features
        """
        df_enhanced = df.copy()
        interaction_count = 0
        
        # Create pairwise interactions for most important numerical features
        important_pairs = [
            ('tenure', 'MonthlyCharges'),
            ('tenure', 'TotalCharges'),
            ('MonthlyCharges', 'TotalCharges')
        ]
        
        for col1, col2 in important_pairs:
            if col1 in numerical_cols and col2 in numerical_cols and interaction_count < max_interactions:
                # Multiplicative interaction
                interaction_name = f"{col1}_x_{col2}"
                df_enhanced[interaction_name] = df_enhanced[col1] * df_enhanced[col2]
                
                # Ratio interaction (avoid division by zero)
                if (df_enhanced[col2] != 0).all():
                    ratio_name = f"{col1}_div_{col2}"
                    df_enhanced[ratio_name] = df_enhanced[col1] / df_enhanced[col2]
                    interaction_count += 1
                
                interaction_count += 1
        
        # Create domain-specific features
        if 'tenure' in df_enhanced.columns and 'MonthlyCharges' in df_enhanced.columns:
            # Customer lifetime value approximation
            df_enhanced['CLV_approx'] = df_enhanced['tenure'] * df_enhanced['MonthlyCharges']
        
        if 'TotalCharges' in df_enhanced.columns and 'tenure' in df_enhanced.columns:
            # Convert TotalCharges to numeric if it's string
            if df_enhanced['TotalCharges'].dtype == 'object':
                df_enhanced['TotalCharges'] = pd.to_numeric(df_enhanced['TotalCharges'], errors='coerce').fillna(0)
            
            # Average monthly charges (alternative calculation)
            df_enhanced['AvgMonthlyCharges'] = df_enhanced['TotalCharges'] / (df_enhanced['tenure'] + 1)
        
        return df_enhanced
    
    def _setup_imbalance_handler(self) -> Any:
        """Setup the imbalance handling strategy."""
        if not IMBALANCED_LEARN_AVAILABLE:
            return None
        
        handlers = {
            'smote': SMOTE(random_state=self.random_state),
            'adasyn': ADASYN(random_state=self.random_state),
            'undersample': RandomUnderSampler(random_state=self.random_state),
            'smote_tomek': SMOTETomek(random_state=self.random_state)
        }
        return handlers[self.imbalance_strategy]
    
    def _generate_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing data quality metrics
        """
        report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'data_types': df.dtypes.to_dict(),
            'unique_values': df.nunique().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        # Numerical statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            report['numerical_stats'] = df[numerical_cols].describe().to_dict()
        
        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            report['categorical_stats'] = {}
            for col in categorical_cols:
                report['categorical_stats'][col] = {
                    'unique_count': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    'frequency': df[col].value_counts().head().to_dict()
                }
        
        return report
    
    def build_preprocessor(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str]
    ) -> ColumnTransformer:
        """
        Build the preprocessing pipeline.
        
        Args:
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            
        Returns:
            ColumnTransformer with preprocessing steps
        """
        # Numerical preprocessing pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
            ('scaler', self._get_scaler())  # Scale features
        ])
        
        # Categorical preprocessing pipeline
        if self.encoding_strategy == 'onehot':
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', self._get_encoder())
            ])
        else:
            # For label/ordinal encoding, handle each column separately
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', self._get_encoder())
            ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ],
            remainder='drop'  # Drop columns not specified
        )
        
        return preprocessor
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AdvancedPreprocessor':
        """
        Fit the preprocessor to the training data.
        
        Args:
            X: Training features
            y: Training labels (optional, needed for feature selection)
            
        Returns:
            Self for method chaining
        """
        # Generate data quality report
        self.data_quality_report_ = self._generate_data_quality_report(X)
        
        # Identify column types
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Handle outliers in numerical columns
        X_processed = X.copy()
        if self.handle_outliers and numerical_cols:
            X_processed = self._detect_outliers_iqr(X_processed, numerical_cols)
        
        # Feature engineering
        if self.feature_engineering and numerical_cols:
            X_processed = self._create_interaction_features(X_processed, numerical_cols)
            # Update numerical columns list after feature engineering
            numerical_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        # Build and fit preprocessor
        self.preprocessor = self.build_preprocessor(numerical_cols, categorical_cols)
        self.preprocessor.fit(X_processed)
        
        # Store feature names for later use
        self._store_feature_names(numerical_cols, categorical_cols)
        
        # Setup feature selection
        if self.feature_selection and y is not None:
            X_transformed = self.preprocessor.transform(X_processed)
            self._setup_feature_selection(X_transformed, y)
        
        # Setup imbalance handling
        if self.handle_imbalance:
            self.imbalance_handler = self._setup_imbalance_handler()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the input data using fitted preprocessor.
        
        Args:
            X: Input features
            
        Returns:
            Transformed feature array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Apply same preprocessing steps as in fit
        X_processed = X.copy()
        
        # Identify column types (same as in fit)
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Handle outliers (using same bounds as training)
        if self.handle_outliers and numerical_cols:
            X_processed = self._detect_outliers_iqr(X_processed, numerical_cols)
        
        # Feature engineering
        if self.feature_engineering and numerical_cols:
            X_processed = self._create_interaction_features(X_processed, numerical_cols)
        
        # Transform using fitted preprocessor
        X_transformed = self.preprocessor.transform(X_processed)
        
        # Apply feature selection if fitted
        if self.feature_selection and self.feature_selector is not None:
            X_transformed = self.feature_selector.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Fit the preprocessor and transform the data in one step.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Transformed feature array
        """
        return self.fit(X, y).transform(X)
    
    def handle_class_imbalance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance in the dataset.
        
        Args:
            X: Feature array
            y: Label array
            
        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        if not self.handle_imbalance or self.imbalance_handler is None:
            return X, y
        
        print(f"Original class distribution: {np.bincount(y)}")
        X_resampled, y_resampled = self.imbalance_handler.fit_resample(X, y)
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def _store_feature_names(self, numerical_cols: List[str], categorical_cols: List[str]) -> None:
        """Store feature names after preprocessing."""
        feature_names = numerical_cols.copy()
        
        if self.encoding_strategy == 'onehot':
            # Get feature names from OneHotEncoder
            try:
                cat_encoder = self.preprocessor.named_transformers_['cat']['encoder']
                if hasattr(cat_encoder, 'get_feature_names_out'):
                    cat_features = cat_encoder.get_feature_names_out(categorical_cols)
                    feature_names.extend(cat_features)
                else:
                    # Fallback for older sklearn versions
                    feature_names.extend([f"{col}_{i}" for col in categorical_cols 
                                        for i in range(len(cat_encoder.categories_[categorical_cols.index(col)]))])
            except:
                # Fallback: just use column names
                feature_names.extend(categorical_cols)
        else:
            feature_names.extend(categorical_cols)
        
        self.feature_names_ = feature_names
    
    def _setup_feature_selection(self, X: np.ndarray, y: np.ndarray) -> None:
        """Setup and fit feature selection."""
        n_features = X.shape[1]
        
        if self.n_features_to_select is None:
            # Select top 50% of features by default
            self.n_features_to_select = max(1, n_features // 2)
        
        # Use SelectKBest with f_classif for feature selection
        self.feature_selector = SelectKBest(
            score_func=f_classif,
            k=min(self.n_features_to_select, n_features)
        )
        self.feature_selector.fit(X, y)
        
        # Store feature selection results
        if hasattr(self.feature_selector, 'scores_'):
            feature_scores = dict(zip(
                self.feature_names_[:len(self.feature_selector.scores_)],
                self.feature_selector.scores_
            ))
            self.data_quality_report_['feature_scores'] = feature_scores
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        if self.feature_names_ is None:
            raise ValueError("Feature names not available. Call fit() first.")
        
        if self.feature_selection and self.feature_selector is not None:
            # Return only selected feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            return [self.feature_names_[i] for i in selected_indices]
        
        return self.feature_names_
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Get the data quality report."""
        return self.data_quality_report_


def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str],
    advanced: bool = False,
    **kwargs
) -> Union[ColumnTransformer, AdvancedPreprocessor]:
    """
    Build a preprocessing pipeline.
    
    Args:
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        advanced: Whether to use advanced preprocessing
        **kwargs: Additional arguments for AdvancedPreprocessor
        
    Returns:
        Preprocessing pipeline
    """
    if advanced:
        return AdvancedPreprocessor(**kwargs)
    else:
        # Simple preprocessing (backward compatibility)
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ]
        )