# 1. Initial Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from scipy.stats import chi2_contingency, uniform, randint
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE



def explore_data(df, title="Dataset"):
    """
    Performs initial exploration of the dataset
    """
    print(f"\n{title} Exploration:")
    print("-" * 50)
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nBasic Statistics:")
    print(df.describe())


def analyze_distributions(df):
    """
    Analyzes and plots distributions of numerical features
    :param df: DataFrame
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    plt.figure(figsize=(15, len(numerical_cols)*4))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(len(numerical_cols), 2, i*2-1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')

        plt.subplot(len(numerical_cols), 2, i*2)
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()


def analyze_non_numeric_distributions(df):
    """
    Analyzes and plots distributions of non-numerical features
    :param df: DataFrame
    """
    numerical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns

    plt.figure(figsize=(15, len(numerical_cols)*4))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(len(numerical_cols), 2, i*2-1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')

        plt.subplot(len(numerical_cols), 2, i*2)
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()


def analyze_non_numeric_distributions(df):
    """
    Analyzes and plots distributions of non-numerical features

    Parameters:
    df (pandas.DataFrame): Input DataFrame
    """
    # Select non-numeric columns (categorical)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(categorical_cols) == 0:
        print("No categorical columns found in the dataset.")
        return

    # Create subplots for each categorical column
    plt.figure(figsize=(15, 5*len(categorical_cols)))

    for i, col in enumerate(categorical_cols, 1):
        # Get value counts and calculate percentages
        value_counts = df[col].value_counts()
        value_percentages = df[col].value_counts(normalize=True) * 100

        # Create subplot for bar plot
        plt.subplot(len(categorical_cols), 2, i*2-1)
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Count')

        # Add percentage labels on top of bars - CORRECTED THIS PART
        for idx, (j, v) in enumerate(zip(value_counts.index, value_counts.values)):
            plt.text(idx, v, f'{value_percentages.iloc[idx]:.1f}%',
                     ha='center', va='bottom')

        # Create subplot for pie chart
        plt.subplot(len(categorical_cols), 2, i*2)
        plt.pie(value_counts.values, labels=value_counts.index,
                autopct='%1.1f%%', startangle=90)
        plt.title(f'Percentage Distribution of {col}')

        # Print summary statistics
        print(f"\nSummary for {col}:")
        print("-" * 50)
        print(f"Number of unique values: {df[col].nunique()}")
        print(f"Most common value: {df[col].mode().iloc[0]} ({value_percentages.iloc[0]:.1f}%)")
        print(f"Missing values: {df[col].isnull().sum()} ({(df[col].isnull().sum()/len(df))*100:.1f}%)")
        print("\nValue Counts:")
        print(pd.DataFrame({
            'Count': value_counts,
            'Percentage': value_percentages
        }))

    plt.tight_layout()
    plt.show()

    # Additional analysis for high-cardinality categorical variables
    high_cardinality_cols = [col for col in categorical_cols
                             if df[col].nunique() > 10]
    if high_cardinality_cols:
        print("\nHigh Cardinality Columns Analysis:")
        print("-" * 50)
        for col in high_cardinality_cols:
            print(f"\n{col}:")
            print(f"Number of unique values: {df[col].nunique()}")
            print("Top 10 most frequent values:")
            print(df[col].value_counts().head(10))



def check_cardinality(df):
    """
    Checks cardinality of categorical columns
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    cardinality = pd.DataFrame({
        'nunique': df[categorical_cols].nunique(),
        'percent_unique': df[categorical_cols].nunique() / len(df) * 100
    }).sort_values('nunique', ascending=False)

    return cardinality



# def analyze_categorical_relationships(df, target_col):
#     """
#     Analyzes relationships between categorical variables and target
#     """
#     categorical_cols = df.select_dtypes(include=['object', 'category']).columns
#     categorical_cols = [col for col in categorical_cols if col != target_col]

#     for col in categorical_cols:
#         plt.figure(figsize=(10, 6))
#         pd.crosstab(df[col], df[target_col], normalize='index').plot(kind='bar')
#         plt.title(f'{col} vs {target_col}')
#         plt.xlabel(col)
#         plt.ylabel(f'Proportion of {target_col}')
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.show()

def analyze_categorical_relationships(df, target_col):
    """
    Analyzes relationships between categorical variables and target using subplots

    Parameters:
    df (pandas.DataFrame): Input DataFrame
    target_col (str): Name of the target column
    """
    # Get categorical columns excluding target
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if col != target_col]

    if len(categorical_cols) == 0:
        print("No categorical columns found in the dataset.")
        return

    # Calculate number of rows and columns for subplots
    n_cols = 2  # You can adjust this
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(15, 5*n_rows),
                             squeeze=False)

    # Flatten axes array for easier iteration
    axes_flat = axes.flatten()

    # Create plots
    for idx, col in enumerate(categorical_cols):
        # Create crosstab
        ct = pd.crosstab(df[col], df[target_col], normalize='index')

        # Plot on corresponding subplot
        ct.plot(kind='bar', ax=axes_flat[idx])

        # Customize subplot
        axes_flat[idx].set_title(f'{col} vs {target_col}')
        axes_flat[idx].set_xlabel(col)
        axes_flat[idx].set_ylabel(f'Proportion of {target_col}')
        axes_flat[idx].tick_params(axis='x', rotation=45)

        # Add percentage labels
        for container in axes_flat[idx].containers:
            axes_flat[idx].bar_label(container, fmt='%.1f%%',
                                     padding=3)

        # Print chi-square test results
        chi2, p_value = chi2_contingency(pd.crosstab(df[col], df[target_col]))[:2]
        print(f"\nChi-square test results for {col}:")
        print(f"Chi-square statistic: {chi2:.2f}")
        print(f"p-value: {p_value:.4f}")

    # Remove empty subplots if any
    for idx in range(len(categorical_cols), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Print additional statistics
    print("\nDetailed Statistics:")
    print("-" * 50)
    for col in categorical_cols:
        print(f"\n{col} breakdown:")
        print(pd.crosstab(df[col], df[target_col],
                          normalize='index').round(3) * 100)


def engineer_features(df):
    """
    Creates new features based on existing data
    :param df: DataFrame
    :return: DataFrame with new features
    """
    df_new = df.copy()

    # Engagement Score
    if 'avg_time_spent' in df.columns and 'points_in_wallet' in df.columns:
        df_new['engagement_score'] = df_new['avg_time_spent'] * df_new['points_in_wallet']

    # Complaint Ratio
    if 'past_complaint' in df.columns and 'complaint_status' in df.columns:
        df_new['complaint_resolution_rate'] = (
            df_new['complaint_status'].map({'Resolved': 1, 'Not Resolved': 0, 'Pending': 0.5})
        )

    # Membership Value
    df_new['membership_value'] = df_new['membership_category'].map({
        'Basic Membership': 1,
        'Silver Membership': 2,
        'Gold Membership': 3,
        'Platinum Membership': 4,
        'Premium Membership': 5,
        'No Membership': 0
    })

    # Activity Level
    if 'days_since_last_login' in df.columns:
        df_new['activity_level'] = pd.cut(
            df_new['days_since_last_login'],
            bins=[-float('inf'), 7, 30, 90, float('inf')],
            labels=['Very Active', 'Active', 'Moderate', 'Inactive']
        )

    return df_new



def preprocess_data(df, is_training=True, fitted_transformers=None, target_col='churn_risk_score'):
    """
    Comprehensive preprocessing pipeline with proper datetime handling
    """
    # Initialize
    if fitted_transformers is None:
        fitted_transformers = {}

    # Copy the dataframe
    df_processed = df.copy()

    # Separate target if it exists
    if target_col in df_processed.columns:
        target = df_processed[target_col].copy()
        df_processed = df_processed.drop(columns=[target_col])

    # 1. First handle date columns
    # Convert joining_date to days_since_joining
    if 'joining_date' in df_processed.columns:
        df_processed['joining_date'] = pd.to_datetime(df_processed['joining_date'])
        df_processed['days_since_joining'] = (pd.Timestamp.now() - df_processed['joining_date']).dt.days
        df_processed = df_processed.drop('joining_date', axis=1)

    # Convert last_visit_time to days_since_last_visit
    if 'last_visit_time' in df_processed.columns:
        df_processed['last_visit_time'] = pd.to_datetime(df_processed['last_visit_time'])
        df_processed['days_since_last_visit'] = (pd.Timestamp.now() - df_processed['last_visit_time']).dt.days
        df_processed = df_processed.drop('last_visit_time', axis=1)

    # 2. Drop unnecessary columns
    id_cols = ['customer_id', 'Name', 'security_no', 'referral_id']
    df_processed = df_processed.drop(columns=[col for col in id_cols if col in df_processed.columns])

    # 3. Handle avg_frequency_login_days before other processing
    if 'avg_frequency_login_days' in df_processed.columns:
        # Convert to numeric
        df_processed['avg_frequency_login_days'] = pd.to_numeric(df_processed['avg_frequency_login_days'], errors='coerce')

        if is_training:
            # Store statistics for test set
            fitted_transformers['avg_freq_mean'] = df_processed['avg_frequency_login_days'].mean()
            fitted_transformers['avg_freq_std'] = df_processed['avg_frequency_login_days'].std()

        # Standardize the values
        mean = fitted_transformers.get('avg_freq_mean', df_processed['avg_frequency_login_days'].mean())
        std = fitted_transformers.get('avg_freq_std', df_processed['avg_frequency_login_days'].std())
        df_processed['avg_frequency_login_days'] = (df_processed['avg_frequency_login_days'] - mean) / std

    # 4. Handle missing values
    numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns

    for col in numerical_cols:
        if df_processed[col].isnull().any():
            if is_training:
                fitted_transformers[f'{col}_median'] = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(fitted_transformers.get(f'{col}_median', 0))

    for col in categorical_cols:
        if df_processed[col].isnull().any():
            if is_training:
                fitted_transformers[f'{col}_mode'] = df_processed[col].mode()[0]
            df_processed[col] = df_processed[col].fillna(fitted_transformers.get(f'{col}_mode', 'Unknown'))

    # 5. Feature Engineering
    if all(col in df_processed.columns for col in ['avg_time_spent', 'points_in_wallet']):
        df_processed['engagement_score'] = df_processed['avg_time_spent'] * df_processed['points_in_wallet']

    # 6. Encode categorical variables
    for col in categorical_cols:
        if df_processed[col].nunique() <= 2:
            if is_training:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                fitted_transformers[f'{col}_encoder'] = le
            else:
                le = fitted_transformers.get(f'{col}_encoder')
                if le is not None:
                    try:
                        df_processed[col] = le.transform(df_processed[col])
                    except ValueError:
                        df_processed[col] = 0
        else:
            dummies = pd.get_dummies(df_processed[col], prefix=col)
            if is_training:
                fitted_transformers[f'{col}_columns'] = dummies.columns.tolist()
            else:
                expected_columns = fitted_transformers.get(f'{col}_columns', [])
                for exp_col in expected_columns:
                    if exp_col not in dummies.columns:
                        dummies[exp_col] = 0
                dummies = dummies[expected_columns]
            df_processed = pd.concat([df_processed.drop(col, axis=1), dummies], axis=1)

    # 7. Scale remaining numerical features
    numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 0:
        if is_training:
            scaler = StandardScaler()
            fitted_transformers['scaler'] = scaler.fit(df_processed[numerical_cols])

        if 'scaler' in fitted_transformers:
            df_processed[numerical_cols] = fitted_transformers['scaler'].transform(df_processed[numerical_cols])

    # 8. Ensure column consistency
    if not is_training and 'training_columns' in fitted_transformers:
        missing_cols = set(fitted_transformers['training_columns']) - set(df_processed.columns)
        for col in missing_cols:
            df_processed[col] = 0
        df_processed = df_processed[fitted_transformers['training_columns']]
    elif is_training:
        fitted_transformers['training_columns'] = df_processed.columns.tolist()

    # Add back target
    if target_col in df.columns:
        df_processed[target_col] = target

    return df_processed, fitted_transformers



def create_model_pipeline(y=None):
    """
    Create a dictionary of models suitable for imbalanced classification
    """
    # Calculate class weights if y is provided
    if y is not None:
        class_weights = get_class_weights(y)
    else:
        class_weights = None

    n_classes = len(np.unique(y)) if y is not None else 6

    models = {
        'xgboost': xgb.XGBClassifier(
            learning_rate=0.1,
            n_estimators=200,
            max_depth=6,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            enable_categorical=True,
            objective='multi:softmax',
            num_class=n_classes,  # For classes 0-5
            random_state=42,
            tree_method='hist',
            seed=42,
            class_weight=class_weights
        ),

        'lightgbm': lgb.LGBMClassifier(
            objective='multiclass',
            num_class=n_classes,
            learning_rate=0.1,
            n_estimators=300,
            max_depth=8,
            num_leaves=31,
            class_weight=class_weights,
            random_state=42,
            verbose=-1,
            boosting_type='gbdt',
            is_unbalance=True
        ),

        'random_forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight=class_weights if class_weights else 'balanced',
            random_state=42
        )
    }

    return models

def custom_metrics(y_true, y_pred):
    """
    Calculate multiple classification metrics
    """
    metrics = {}
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # Calculate confusion matrix and derived metrics
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm

    # Per-class metrics
    metrics['per_class_precision'] = np.diag(cm) / np.sum(cm, axis=0)
    metrics['per_class_recall'] = np.diag(cm) / np.sum(cm, axis=1)

    return metrics


def plot_confusion_matrices(y_true, predictions):
    """
    Plot confusion matrices. Can handle both dictionary of predictions and single prediction array.
    """
    if not isinstance(predictions, dict):
        # If single array is passed, convert to dictionary
        predictions = {'model': predictions}

    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))

    # Convert axes to array if single plot
    if n_models == 1:
        axes = np.array([axes])

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title(f'{name} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.tight_layout()
    plt.show()


def plot_roc_curves(y_true, predictions_proba_dict):
    plt.figure(figsize=(10, 6))

    for name, y_pred_proba in predictions_proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.show()


def print_classification_reports(y_true, predictions):
    """
    Print classification reports. Can handle both dictionary of predictions and single prediction array.
    """
    reports = {}

    if not isinstance(predictions, dict):
        # If single array is passed, convert to dictionary
        predictions = {'model': predictions}

    for name, y_pred in predictions.items():
        print(f"\n{name} Classification Report:")
        print(classification_report(y_true, y_pred))
        reports[name] = classification_report(y_true, y_pred, output_dict=True)

    return reports


def compare_model_metrics(reports):
    metrics = ['precision', 'recall', 'f1-score']
    comparison = {}

    for model, report in reports.items():
        comparison[model] = {metric: report['weighted avg'][metric]
                             for metric in metrics}

    return pd.DataFrame(comparison).round(3)


def train_evaluate_models(X, y, models, n_splits=5):
    """
    Train and evaluate models using both cross-validation and train-test split
    """
    print("Shape of X:", X.shape)
    print("Unique classes in y:", np.unique(y))
    print("Class distribution:", pd.Series(y).value_counts().sort_index())

    # Convert categorical columns to numeric
    X_processed = X.copy()
    categorical_columns = X_processed.select_dtypes(include=['object']).columns

    print("\nCategorical columns being processed:", categorical_columns.tolist())

    # Convert each categorical column
    for col in categorical_columns:
        if X_processed[col].nunique() == 2:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col])
        else:
            dummies = pd.get_dummies(X_processed[col], prefix=col, drop_first=True)
            X_processed = pd.concat([X_processed.drop(col, axis=1), dummies], axis=1)

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    results = {}
    feature_importance = {}
    validation_results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        try:
            # Train on training set
            model.fit(X_train, y_train)

            # Make predictions on validation set
            val_predictions = model.predict(X_val)

            # Calculate validation metrics
            validation_results[name] = {
                'accuracy': accuracy_score(y_val, val_predictions),
                'weighted_f1': f1_score(y_val, val_predictions, average='weighted'),
                'confusion_matrix': confusion_matrix(y_val, val_predictions)
            }

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(validation_results[name]['confusion_matrix'],
                        annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} Validation Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

            # Cross-validation
            cv_results = cross_validate(
                model, X_processed, y,
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
                scoring={
                    'weighted_f1': make_scorer(f1_score, average='weighted'),
                    'balanced_accuracy': make_scorer(balanced_accuracy_score)
                },
                return_train_score=True,
                n_jobs=-1
            )

            # Store CV results
            results[name] = {
                'test_weighted_f1_mean': cv_results['test_weighted_f1'].mean(),
                'test_weighted_f1_std': cv_results['test_weighted_f1'].std(),
                'test_balanced_accuracy_mean': cv_results['test_balanced_accuracy'].mean(),
                'test_balanced_accuracy_std': cv_results['test_balanced_accuracy'].std(),
            }

            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance[name] = pd.Series(
                    model.feature_importances_,
                    index=X_processed.columns
                ).sort_values(ascending=False)

            # Print validation results
            print(f"\nValidation Results for {name}:")
            print(f"Accuracy: {validation_results[name]['accuracy']:.4f}")
            print(f"Weighted F1: {validation_results[name]['weighted_f1']:.4f}")

        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue

    return results, feature_importance, validation_results


def create_ensemble(models, voting='soft'):
    """
    Create an ensemble model using voting
    """
    estimators = [(name, model) for name, model in models.items()]
    ensemble = VotingClassifier(estimators=estimators, voting=voting)
    return ensemble


def plot_feature_importance(feature_importance_dict):
    """
    Plot feature importance for each model
    """
    plt.figure(figsize=(15, 10))

    for i, (name, importance) in enumerate(feature_importance_dict.items(), 1):
        plt.subplot(len(feature_importance_dict), 1, i)
        importance.head(20).plot(kind='barh')
        plt.title(f'Top 20 Important Features - {name}')
        plt.tight_layout()

    plt.show()


def run_model_pipeline(X_train, y_train):
    """
    Run the complete modeling pipeline
    """
    # Create models
    models = create_model_pipeline(y_train)

    # Train and evaluate individual models
    print("Training and evaluating individual models...")
    results, feature_importance, validation_results = train_evaluate_models(X_train, y_train, models)

    # Access validation results
    for name, metrics in validation_results.items():
        print(f"\nValidation metrics for {name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")

    # Print results
    print("\nModel Performance Summary:")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"\n{name} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # Create and evaluate ensemble
    print("\nTraining ensemble model...")
    ensemble = create_ensemble(models)
    ensemble_results, _, ensemble_validation = train_evaluate_models(
        X_train, y_train, {'ensemble': ensemble}
    )

    # Plot feature importance
    print("\nPlotting feature importance...")
    plot_feature_importance(feature_importance)

    return results, feature_importance, ensemble


# Additional utility functions for hyperparameter tuning
# def tune_xgboost(X, y):
#     """
#     Perform hyperparameter tuning for XGBoost
#     """
#     param_grid = {
#         'max_depth': [3, 5, 7],
#         'learning_rate': [0.01, 0.1],
#         'n_estimators': [100, 200],
#         'min_child_weight': [1, 3],
#         'gamma': [0, 0.1],
#         'subsample': [0.8, 0.9],
#         'colsample_bytree': [0.8, 0.9]
#     }
#
#     xgb_model = xgb.XGBClassifier(random_state=42)
#
#     # Use RandomizedSearchCV for faster tuning
#     search = RandomizedSearchCV(
#         xgb_model, param_grid,
#         n_iter=10,
#         scoring='f1_weighted',
#         n_jobs=-1,
#         cv=StratifiedKFold(n_splits=5),
#         random_state=42
#     )
#
#     search.fit(X, y)
#     return search.best_params_

def tune_xgboost(X, y):
    """
    Perform hyperparameter tuning for XGBoost with proper categorical handling
    """
    # Process categorical columns first
    X_processed = X.copy()
    categorical_columns = X_processed.select_dtypes(include=['object']).columns

    print(f"Processing categorical columns: {list(categorical_columns)}")

    # Convert categorical columns
    for col in categorical_columns:
        if X_processed[col].nunique() == 2:
            # Binary categories
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col])
        else:
            # Multi-class categories
            dummies = pd.get_dummies(X_processed[col], prefix=col, drop_first=True)
            X_processed = pd.concat([X_processed.drop(col, axis=1), dummies], axis=1)

    print("Shape after processing:", X_processed.shape)
    print("Data types after processing:\n", X_processed.dtypes)

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'tree_method': ['hist'],  # Use histogram-based algorithm
        'objective': ['multi:softmax'],
        'num_class': [len(np.unique(y))]
    }

    xgb_model = xgb.XGBClassifier(random_state=42)

    search = RandomizedSearchCV(
        xgb_model,
        param_grid,
        n_iter=10,
        scoring='f1_weighted',
        n_jobs=-1,
        cv=StratifiedKFold(n_splits=5),
        random_state=42,
        error_score='raise'  # This will help with debugging
    )

    try:
        search.fit(X_processed, y)
        print("Best parameters found:", search.best_params_)
        return search.best_params_
    except Exception as e:
        print("Error during tuning:", str(e))
        print("Falling back to default parameters")
        return {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'objective': 'multi:softmax',
            'num_class': len(np.unique(y))
        }



def tune_lightgbm(X, y):
    """
    Perform hyperparameter tuning for LightGBM with proper categorical handling
    """
    # Process categorical columns first
    X_processed = X.copy()
    categorical_columns = X_processed.select_dtypes(include=['object']).columns

    print(f"Processing categorical columns: {list(categorical_columns)}")

    # Convert categorical columns
    for col in categorical_columns:
        if X_processed[col].nunique() == 2:
            # Binary categories
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col])
        else:
            # Multi-class categories
            dummies = pd.get_dummies(X_processed[col], prefix=col, drop_first=True)
            X_processed = pd.concat([X_processed.drop(col, axis=1), dummies], axis=1)

    print("Shape after processing:", X_processed.shape)
    print("Data types after processing:\n", X_processed.dtypes)

    param_grid = {
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'num_leaves': randint(20, 100),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'objective': ['multiclass'],
        'num_class': [len(np.unique(y))]
    }

    try:
        search = RandomizedSearchCV(
            estimator=lgb.LGBMClassifier(objective='multiclass'),
            param_distributions=param_grid,
            n_iter=20,
            cv=5,
            random_state=42,
            n_jobs=-1,
            error_score='raise'
        )

        search.fit(X_processed, y)
        print("Best parameters found:", search.best_params_)
        return search.best_params_

    except Exception as e:
        print("Error during tuning:", str(e))
        print("Falling back to default parameters")
        return {
            'learning_rate': 0.1,
            'n_estimators': 200,
            'max_depth': 6,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multiclass',
            'num_class': len(np.unique(y))
        }


# def tune_lightgbm(X, y):
#     param_grid = {
#         'learning_rate': uniform(0.01, 0.3),
#         'n_estimators': randint(100, 500),
#         'max_depth': randint(3, 10),
#         'num_leaves': randint(20, 100),
#         'subsample': uniform(0.6, 0.4),
#         'colsample_bytree': uniform(0.6, 0.4)
#     }
#
#     search = RandomizedSearchCV(
#         estimator=lgb.LGBMClassifier(),
#         param_distributions=param_grid,
#         n_iter=20,
#         cv=5,
#         random_state=42,
#         n_jobs=-1
#     )
#
#     search.fit(X, y)
#     return search.best_params_

def create_optimized_ensemble(models, X, y):
    """Create a voting classifier with optimized weights"""
    from sklearn.ensemble import VotingClassifier
    from sklearn.model_selection import cross_val_score

    # Get base predictions
    predictions = {}
    for name, model in models.items():
        predictions[name] = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    # Calculate weights based on mean CV scores
    weights = {name: scores.mean() for name, scores in predictions.items()}
    total = sum(weights.values())
    weights = [w/total for w in weights.values()]

    # Create voting classifier
    estimators = [(name, model) for name, model in models.items()]
    ensemble = VotingClassifier(
        estimators=estimators,
        weights=weights,
        voting='soft'
    )

    return ensemble


def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def prepare_data_for_smote(X, y):
    """
    Prepare data for SMOTE by handling NaN values and categorical variables.
    SMOTE requires all features to be numeric and cannot handle missing values or categorical variables directly.
    If I tried to apply SMOTE to raw data containing categorical variables or NaN values,
    it would fail since SMOTE works by creating synthetic samples through interpolation between existing samples
    in feature space. Consistent data preprocessing ensures that SMOTE can effectively generate synthetic
    samples that maintain the meaningful relationships between features.
    """
    X_processed = X.copy()

    # First handle NaN values
    numeric_cols = X_processed.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_processed.select_dtypes(include=['object']).columns

    print("Handling missing values...")
    print("Numeric columns:", numeric_cols.tolist())
    print("Categorical columns:", categorical_cols.tolist())

    # Handle numeric columns
    for col in numeric_cols:
        median_val = X_processed[col].median()
        X_processed[col] = X_processed[col].fillna(median_val)

    # Handle categorical columns
    for col in categorical_cols:
        # Replace 'Error' values with NaN first
        X_processed[col] = X_processed[col].replace('Error', np.nan)
        # Then fill NaN with mode
        mode_val = X_processed[col].mode()[0]
        X_processed[col] = X_processed[col].fillna(mode_val)

        # Encode categorical variables
        if X_processed[col].nunique() == 2:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col])
        else:
            dummies = pd.get_dummies(X_processed[col], prefix=col, drop_first=True)
            X_processed = pd.concat([X_processed.drop(col, axis=1), dummies], axis=1)

    print("\nData shape after processing:", X_processed.shape)
    print("Any remaining NaN values:", X_processed.isnull().sum().sum())

    return X_processed, y
