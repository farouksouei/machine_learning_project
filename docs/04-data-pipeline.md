# Data Pipeline Documentation

The data pipeline is the foundation of the Customer Churn Prediction System, responsible for loading, validating, and preparing customer data for machine learning. This document covers every aspect of data handling in the system.

## 📊 Data Overview

### Dataset Structure

The system expects customer data in CSV format with the following structure:

```csv
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
5575-GNVDE,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.5,No
...
```

### Data Schema

#### Customer Demographics
| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `customerID` | String | Unique customer identifier | "7590-VHVEG" |
| `gender` | String | Customer gender | "Male", "Female" |
| `SeniorCitizen` | Integer | Senior citizen flag | 0, 1 |
| `Partner` | String | Has partner | "Yes", "No" |
| `Dependents` | String | Has dependents | "Yes", "No" |

#### Service Information
| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `tenure` | Integer | Months with company | 1, 12, 34, 72 |
| `PhoneService` | String | Has phone service | "Yes", "No" |
| `MultipleLines` | String | Multiple phone lines | "Yes", "No", "No phone service" |
| `InternetService` | String | Internet service type | "DSL", "Fiber optic", "No" |
| `OnlineSecurity` | String | Online security service | "Yes", "No", "No internet service" |
| `OnlineBackup` | String | Online backup service | "Yes", "No", "No internet service" |
| `DeviceProtection` | String | Device protection service | "Yes", "No", "No internet service" |
| `TechSupport` | String | Technical support service | "Yes", "No", "No internet service" |
| `StreamingTV` | String | TV streaming service | "Yes", "No", "No internet service" |
| `StreamingMovies` | String | Movie streaming service | "Yes", "No", "No internet service" |

#### Contract and Billing
| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `Contract` | String | Contract type | "Month-to-month", "One year", "Two year" |
| `PaperlessBilling` | String | Paperless billing | "Yes", "No" |
| `PaymentMethod` | String | Payment method | "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)" |
| `MonthlyCharges` | Float | Monthly charges in dollars | 29.85, 56.95, 89.10 |
| `TotalCharges` | String/Float | Total charges (may be string) | "29.85", "1889.5", " " (empty) |

#### Target Variable
| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `Churn` | String | Customer churned | "Yes", "No" |

## 🔧 Data Loading (`data_loader.py`)

### Basic Data Loading

```python
from data_loader import load_data

# Load the dataset
df = load_data('data/dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
```

### Implementation Details

```python
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load customer churn dataset from CSV file.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        pandas DataFrame with customer data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file format is invalid
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {path}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Data file is empty: {path}")
    except Exception as e:
        raise pd.errors.ParserError(f"Error parsing data file: {e}")
```

### Advanced Data Loading

For production use, you might want more robust data loading:

```python
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

def load_data_advanced(
    path: str,
    validate_schema: bool = True,
    handle_missing: bool = True,
    encoding: str = 'utf-8'
) -> pd.DataFrame:
    """
    Advanced data loading with validation and error handling.
    
    Args:
        path: Path to the CSV file
        validate_schema: Whether to validate data schema
        handle_missing: Whether to handle missing values
        encoding: File encoding
        
    Returns:
        Cleaned and validated DataFrame
    """
    # Load data with error handling
    try:
        df = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        # Try different encodings
        for enc in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(path, encoding=enc)
                print(f"Successfully loaded with encoding: {enc}")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode file with any common encoding")
    
    # Validate schema
    if validate_schema:
        df = validate_data_schema(df)
    
    # Handle missing values
    if handle_missing:
        df = handle_missing_values(df)
    
    return df

def validate_data_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the DataFrame has the expected schema."""
    required_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges', 'Churn'
    ]
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate data types
    if df['SeniorCitizen'].dtype not in ['int64', 'int32']:
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    
    if df['tenure'].dtype not in ['int64', 'int32']:
        df['tenure'] = df['tenure'].astype(int)
    
    if df['MonthlyCharges'].dtype not in ['float64', 'float32']:
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    # Handle TotalCharges (often has empty strings)
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with tenure * MonthlyCharges
    missing_total = df['TotalCharges'].isnull()
    df.loc[missing_total, 'TotalCharges'] = (
        df.loc[missing_total, 'tenure'] * df.loc[missing_total, 'MonthlyCharges']
    )
    
    return df
```

## 🔍 Data Validation and Quality Checks

### Data Quality Assessment

```python
def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive data quality assessment.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'unique_values': df.nunique().to_dict()
    }
    
    # Categorical value distributions
    categorical_cols = df.select_dtypes(include=['object']).columns
    quality_report['categorical_distributions'] = {}
    for col in categorical_cols:
        quality_report['categorical_distributions'][col] = df[col].value_counts().to_dict()
    
    # Numerical statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    quality_report['numerical_stats'] = df[numerical_cols].describe().to_dict()
    
    # Data quality issues
    issues = []
    
    # Check for high missing value percentage
    for col, pct in quality_report['missing_percentage'].items():
        if pct > 10:
            issues.append(f"High missing values in {col}: {pct:.1f}%")
    
    # Check for potential data entry errors
    if 'tenure' in df.columns:
        if df['tenure'].min() < 0 or df['tenure'].max() > 100:
            issues.append("Unusual tenure values detected")
    
    if 'MonthlyCharges' in df.columns:
        if df['MonthlyCharges'].min() < 0 or df['MonthlyCharges'].max() > 1000:
            issues.append("Unusual MonthlyCharges values detected")
    
    quality_report['issues'] = issues
    
    return quality_report

# Usage
df = load_data('data/dataset.csv')
quality_report = assess_data_quality(df)
print(f"Data quality issues: {len(quality_report['issues'])}")
for issue in quality_report['issues']:
    print(f"- {issue}")
```

### Data Validation Rules

```python
def validate_business_rules(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate business logic rules in the data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_records': len(df),
        'valid_records': 0,
        'violations': []
    }
    
    violations = []
    
    # Rule 1: TotalCharges should be >= MonthlyCharges for tenure > 0
    if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns:
        invalid_charges = df[
            (df['tenure'] > 0) & 
            (df['TotalCharges'] < df['MonthlyCharges'])
        ]
        if len(invalid_charges) > 0:
            violations.append({
                'rule': 'TotalCharges >= MonthlyCharges for tenure > 0',
                'violations': len(invalid_charges),
                'percentage': len(invalid_charges) / len(df) * 100
            })
    
    # Rule 2: Customers with no internet service shouldn't have internet-dependent services
    if 'InternetService' in df.columns:
        no_internet = df['InternetService'] == 'No'
        internet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                           'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        for service in internet_services:
            if service in df.columns:
                invalid_service = df[no_internet & (df[service] != 'No internet service')]
                if len(invalid_service) > 0:
                    violations.append({
                        'rule': f'{service} should be "No internet service" when InternetService is "No"',
                        'violations': len(invalid_service),
                        'percentage': len(invalid_service) / len(df) * 100
                    })
    
    # Rule 3: Customers with no phone service shouldn't have MultipleLines
    if 'PhoneService' in df.columns and 'MultipleLines' in df.columns:
        no_phone = df['PhoneService'] == 'No'
        invalid_lines = df[no_phone & (df['MultipleLines'] != 'No phone service')]
        if len(invalid_lines) > 0:
            violations.append({
                'rule': 'MultipleLines should be "No phone service" when PhoneService is "No"',
                'violations': len(invalid_lines),
                'percentage': len(invalid_lines) / len(df) * 100
            })
    
    validation_results['violations'] = violations
    validation_results['valid_records'] = len(df) - sum(v['violations'] for v in violations)
    validation_results['validity_percentage'] = validation_results['valid_records'] / len(df) * 100
    
    return validation_results
```

## 🔄 Data Preprocessing Integration

### Data Splitting

The training pipeline automatically handles data splitting:

```python
from train import ChurnTrainer

trainer = ChurnTrainer(
    data_path="data/dataset.csv",
    test_size=0.2,        # 20% for testing
    validation_size=0.2,  # 20% of remaining for validation
    random_state=42
)

# Load and split data
trainer.load_and_split_data()

print(f"Training set: {trainer.X_train.shape}")
print(f"Validation set: {trainer.X_val.shape}")
print(f"Test set: {trainer.X_test.shape}")
```

### Target Variable Handling

The system automatically handles target variable encoding:

```python
# Automatic encoding of string targets
# "Yes" -> 1, "No" -> 0
y = df['Churn'].map({"Yes": 1, "No": 0})

# Fallback for other formats
if y.isnull().any():
    unique_values = df['Churn'].unique()
    print(f"Target values: {unique_values}")
    y = (df['Churn'] == unique_values[1]).astype(int)
```

## 📈 Data Exploration and Analysis

### Exploratory Data Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        df: Input DataFrame
    """
    print("=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    missing_pct = missing / len(df) * 100
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    }).sort_values('Missing', ascending=False)
    print(missing_df[missing_df['Missing'] > 0])
    
    print("\n=== TARGET DISTRIBUTION ===")
    if 'Churn' in df.columns:
        churn_dist = df['Churn'].value_counts()
        churn_pct = df['Churn'].value_counts(normalize=True) * 100
        print(f"No Churn: {churn_dist['No']} ({churn_pct['No']:.1f}%)")
        print(f"Churn: {churn_dist['Yes']} ({churn_pct['Yes']:.1f}%)")
    
    print("\n=== NUMERICAL FEATURES ===")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numerical_cols].describe())
    
    print("\n=== CATEGORICAL FEATURES ===")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'customerID':  # Skip ID column
            print(f"\n{col}:")
            print(df[col].value_counts())

def plot_data_distribution(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Create visualizations for data distribution.
    
    Args:
        df: Input DataFrame
        save_path: Path to save plots (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Churn distribution
    if 'Churn' in df.columns:
        df['Churn'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Churn Distribution')
        axes[0,0].set_xlabel('Churn')
        axes[0,0].set_ylabel('Count')
    
    # Tenure distribution
    if 'tenure' in df.columns:
        df['tenure'].hist(bins=30, ax=axes[0,1])
        axes[0,1].set_title('Tenure Distribution')
        axes[0,1].set_xlabel('Tenure (months)')
        axes[0,1].set_ylabel('Frequency')
    
    # Monthly charges distribution
    if 'MonthlyCharges' in df.columns:
        df['MonthlyCharges'].hist(bins=30, ax=axes[1,0])
        axes[1,0].set_title('Monthly Charges Distribution')
        axes[1,0].set_xlabel('Monthly Charges ($)')
        axes[1,0].set_ylabel('Frequency')
    
    # Contract type distribution
    if 'Contract' in df.columns:
        df['Contract'].value_counts().plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Contract Type Distribution')
        axes[1,1].set_xlabel('Contract Type')
        axes[1,1].set_ylabel('Count')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Usage
df = load_data('data/dataset.csv')
explore_data(df)
plot_data_distribution(df, 'data_distribution.png')
```

## 🔧 Data Pipeline Configuration

### Custom Data Loading

```python
class DataPipeline:
    """
    Configurable data pipeline for customer churn prediction.
    """
    
    def __init__(
        self,
        data_path: str,
        target_column: str = 'Churn',
        id_column: str = 'customerID',
        validation_rules: bool = True,
        quality_checks: bool = True
    ):
        self.data_path = data_path
        self.target_column = target_column
        self.id_column = id_column
        self.validation_rules = validation_rules
        self.quality_checks = quality_checks
        
        self.data = None
        self.quality_report = None
        self.validation_report = None
    
    def load_data(self) -> pd.DataFrame:
        """Load and process the data."""
        print(f"Loading data from {self.data_path}...")
        
        # Load raw data
        self.data = load_data_advanced(self.data_path)
        print(f"Loaded {len(self.data)} records with {len(self.data.columns)} columns")
        
        # Quality checks
        if self.quality_checks:
            self.quality_report = assess_data_quality(self.data)
            print(f"Data quality issues: {len(self.quality_report['issues'])}")
        
        # Validation rules
        if self.validation_rules:
            self.validation_report = validate_business_rules(self.data)
            print(f"Business rule violations: {len(self.validation_report['violations'])}")
        
        return self.data
    
    def get_features_and_target(self) -> tuple:
        """Split data into features and target."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Remove ID column and target from features
        feature_columns = [col for col in self.data.columns 
                          if col not in [self.id_column, self.target_column]]
        
        X = self.data[feature_columns]
        
        # Handle target encoding
        if self.data[self.target_column].dtype == 'object':
            y = self.data[self.target_column].map({"Yes": 1, "No": 0})
            if y.isnull().any():
                unique_values = self.data[self.target_column].unique()
                y = (self.data[self.target_column] == unique_values[1]).astype(int)
        else:
            y = self.data[self.target_column]
        
        return X, y
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get data quality report."""
        return self.quality_report
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get validation report."""
        return self.validation_report

# Usage
pipeline = DataPipeline('data/dataset.csv')
df = pipeline.load_data()
X, y = pipeline.get_features_and_target()

print(f"Features shape: {X.shape}")
print(f"Target distribution: {y.value_counts()}")
```

## 📊 Data Monitoring and Drift Detection

### Data Drift Detection

```python
def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Detect data drift between reference and current datasets.
    
    Args:
        reference_data: Reference dataset (e.g., training data)
        current_data: Current dataset (e.g., new production data)
        threshold: Significance threshold for drift detection
        
    Returns:
        Dictionary with drift detection results
    """
    from scipy.stats import ks_2samp, chi2_contingency
    
    drift_results = {
        'numerical_drift': {},
        'categorical_drift': {},
        'overall_drift': False
    }
    
    # Numerical features drift (Kolmogorov-Smirnov test)
    numerical_cols = reference_data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col in current_data.columns:
            statistic, p_value = ks_2samp(
                reference_data[col].dropna(),
                current_data[col].dropna()
            )
            
            drift_results['numerical_drift'][col] = {
                'statistic': statistic,
                'p_value': p_value,
                'drift_detected': p_value < threshold
            }
    
    # Categorical features drift (Chi-square test)
    categorical_cols = reference_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in current_data.columns and col != 'customerID':
            try:
                # Create contingency table
                ref_counts = reference_data[col].value_counts()
                curr_counts = current_data[col].value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(curr_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                
                # Chi-square test
                contingency_table = [ref_aligned, curr_aligned]
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                
                drift_results['categorical_drift'][col] = {
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'drift_detected': p_value < threshold
                }
            except Exception as e:
                drift_results['categorical_drift'][col] = {
                    'error': str(e),
                    'drift_detected': False
                }
    
    # Overall drift assessment
    all_drifts = []
    all_drifts.extend([v['drift_detected'] for v in drift_results['numerical_drift'].values()])
    all_drifts.extend([v['drift_detected'] for v in drift_results['categorical_drift'].values() if 'drift_detected' in v])
    
    drift_results['overall_drift'] = any(all_drifts)
    drift_results['drift_percentage'] = sum(all_drifts) / len(all_drifts) * 100 if all_drifts else 0
    
    return drift_results

# Usage
reference_df = load_data('data/training_dataset.csv')
current_df = load_data('data/current_dataset.csv')

drift_report = detect_data_drift(reference_df, current_df)
print(f"Overall drift detected: {drift_report['overall_drift']}")
print(f"Drift percentage: {drift_report['drift_percentage']:.1f}%")
```

## 🚀 Best Practices

### 1. Data Loading Best Practices

```python
# Always use try-catch for data loading
try:
    df = load_data('data/dataset.csv')
except FileNotFoundError:
    print("Data file not found. Please check the path.")
except Exception as e:
    print(f"Error loading data: {e}")

# Validate data immediately after loading
quality_report = assess_data_quality(df)
if quality_report['issues']:
    print("Data quality issues detected:")
    for issue in quality_report['issues']:
        print(f"- {issue}")
```

### 2. Memory Optimization

```python
# Optimize data types to reduce memory usage
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame data types for memory efficiency."""
    optimized_df = df.copy()
    
    # Convert object columns with few unique values to category
    for col in optimized_df.select_dtypes(include=['object']).columns:
        if col != 'customerID':  # Skip ID columns
            unique_ratio = optimized_df[col].nunique() / len(optimized_df)
            if unique_ratio < 0.5:  # Less than 50% unique values
                optimized_df[col] = optimized_df[col].astype('category')
    
    # Downcast numerical columns
    for col in optimized_df.select_dtypes(include=['int64']).columns:
        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
    
    for col in optimized_df.select_dtypes(include=['float64']).columns:
        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
    
    return optimized_df

# Usage
df_optimized = optimize_dtypes(df)
print(f"Memory reduction: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB -> {df_optimized.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

### 3. Data Versioning

```python
import hashlib
import json

def create_data_fingerprint(df: pd.DataFrame) -> str:
    """Create a unique fingerprint for the dataset."""
    # Create a hash based on data shape, column names, and sample of data
    fingerprint_data = {
        'shape': df.shape,
        'columns': sorted(df.columns.tolist()),
        'dtypes': df.dtypes.to_dict(),
        'sample_hash': hashlib.md5(df.head(100).to_string().encode()).hexdigest()
    }
    
    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
    return hashlib.md5(fingerprint_str.encode()).hexdigest()

# Usage
data_fingerprint = create_data_fingerprint(df)
print(f"Data fingerprint: {data_fingerprint}")

# Save fingerprint with model metadata
metadata = {
    'data_fingerprint': data_fingerprint,
    'data_path': 'data/dataset.csv',
    'data_shape': df.shape,
    'created_at': pd.Timestamp.now().isoformat()
}
```

---

**Next**: Continue to [Model Architecture](05-model-architecture.md) to understand the machine learning algorithms used in the system.