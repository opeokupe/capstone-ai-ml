#  Initial Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from scipy.stats import chi2_contingency, uniform, randint
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# Import all functions
from my_functions import *

#

# Load the data
train_url = 'https://raw.githubusercontent.com/opeokupe/capstone-ai-ml/main/data/train.csv'
test_url = 'https://raw.githubusercontent.com/opeokupe/capstone-ai-ml/main/data/test.csv'
# train_url = 'https://raw.githubusercontent.com/opeokupe/capstone-ai-ml/main/data/train_2.csv'
# test_url = 'https://raw.githubusercontent.com/opeokupe/capstone-ai-ml/main/data/test_2.csv'
train_data = pd.read_csv(train_url, on_bad_lines='skip')
test_data = pd.read_csv(test_url, on_bad_lines='skip' )

# Explore the data
explore_data(train_data, "Training Dataset")

# Plot distributions
analyze_distributions(train_data)

# Check gender distribution
gender_counts = train_data['gender'].value_counts()
plt.figure(figsize=(8, 5))
gender_counts.plot(kind='bar', color=['blue', 'pink'], edgecolor='black', alpha=0.7)
plt.title('Gender Representation')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Provide numerical breakdown
# age_summary = train_data['age'].describe()
gender_summary = gender_counts.to_dict()

# age_summary,
gender_summary

excluded_columns = ['customer_id','security_no', 'Name','referral_id']
filtered_columns_train = [col for col in train_data if col not in excluded_columns]
filtered_columns_test = [col for col in test_data if col not in excluded_columns]
filtered_data_train = train_data[filtered_columns_train]
filtered_data_test = test_data[filtered_columns_test]

check_cardinality(filtered_data_train)

analyze_categorical_relationships(filtered_data_train, 'churn_risk_score')

analyze_non_numeric_distributions(filtered_data_train)

"""Feature Engineering"""
# Apply feature engineering
train_data_processed = engineer_features(filtered_data_train)
test_data_processed = engineer_features(filtered_data_test)


print(f"original train data shape: {train_data.shape}")
print(f"filtered_columns_train data shape: {filtered_data_train.shape}")
print(f"train_data_processed data train data shape: {train_data_processed.shape}")
print(train_data_processed.columns.tolist())

print(f"original test data shape: {test_data.shape}")
print(f"filtered_columns_test data shape: {filtered_data_test.shape}")
print(f"test_data_processed data train data shape: {test_data_processed.shape}")
print(test_data_processed.columns.tolist())


""" Data Processing"""

# For training data:
train_processed, fitted_transformers = preprocess_data(train_data, is_training=True)
train_columns = train_processed.columns.tolist()

# Save columns for later use
np.save('train_columns.npy', train_columns)

# For test/validation data (using same transformations):
test_processed, _ = preprocess_data(test_data, is_training=False, fitted_transformers=fitted_transformers)
train_columns = np.load('train_columns.npy')

# Align test columns with train columns
for col in train_columns:
    if col not in test_processed.columns:
        test_processed[col] = 0
test_processed = test_processed[train_columns]

# Print summary of preprocessing
print("\nPreprocessing Summary:")
print(f"Original train shape: {train_data.shape}")
print(f"Processed train shape: {train_processed.shape}")
print(f"Original test shape: {test_data.shape}")
print(f"Processed test shape: {test_processed.shape}")

# Print feature names to verify
print("\nFeatures after preprocessing:")
print(train_processed.columns.tolist())

print(f"train processed: {train_processed.columns.tolist()}")
print(f"test processed: {test_processed.columns.tolist()}")

# Prepare  data (after preprocessing)
X = train_processed.drop('churn_risk_score', axis=1)
y = train_processed['churn_risk_score']
XT = test_processed.drop('churn_risk_score', axis=1)
yt = test_processed['churn_risk_score']

# Print distributions
print("Training data distribution:")
print(y.value_counts().sort_index())
print("\nTest data distribution:")
print(yt.value_counts().sort_index())


# results, feature_importance, ensemble = run_model_pipeline(X, y)



# Assuming 'y' is your target variable Series
print("Original target distribution:", y.value_counts())
unique_classes = sorted(y.unique())
print("Unique classes:", unique_classes)

# Replace -1 with 0 in both training and test data
y = y.replace(-1, 0)
yt = yt.replace(-1, 0)

"""*****************************"""

# Check for NaN values before processing
print("\nNaN values in X before processing:")
print(X.isnull().sum()[X.isnull().sum() > 0])


# Process data before SMOTE
X_processed, y = prepare_data_for_smote(X, y)

# Explicitly handle complaint_resolution_rate NaN values
if 'complaint_resolution_rate' in X_processed.columns:
    print("\nHandling complaint_resolution_rate NaN values")
    median_value = X_processed['complaint_resolution_rate'].median()
    if np.isnan(median_value):  # If median is also NaN
        X_processed['complaint_resolution_rate'] = X_processed['complaint_resolution_rate'].fillna(0)
    else:
        X_processed['complaint_resolution_rate'] = X_processed['complaint_resolution_rate'].fillna(median_value)

# Check for any remaining NaN values in any column
columns_with_nan = X_processed.columns[X_processed.isnull().any()].tolist()
if columns_with_nan:
    print("\nFilling remaining NaN values in columns:", columns_with_nan)
    for col in columns_with_nan:
        if X_processed[col].dtype.kind in 'fc':  # float or complex
            X_processed[col] = X_processed[col].fillna(0)  # Using 0 instead of mean
        else:
            X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0])

# Final verification
print("\nVerifying no NaN values remain:")
final_nan_check = X_processed.isnull().sum().sum()
print("NaN values:", final_nan_check)

if final_nan_check > 0:
    print("WARNING: Still have NaN values. Removing rows with NaN values as last resort.")
    X_processed = X_processed.dropna()
    y = y[X_processed.index]


# Verify no NaN values remain
print("\nVerifying no NaN values remain:")
print("NaN values:", X_processed.isnull().sum().sum())
# Print original distribution
print("\nOriginal class distribution:")
print(pd.Series(y).value_counts().sort_index())

# Now apply SMOTE to the processed data
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_processed, y)

# Convert back to DataFrame to maintain column names
X_balanced = pd.DataFrame(X_balanced, columns=X_processed.columns)

print("\nBalanced class distribution:")
print(pd.Series(y_balanced).value_counts().sort_index())

# Update X and y
X = X_balanced
y = y_balanced

"""*****************************"""
# Create and run models
models = create_model_pipeline(y)
# # Run the complete pipeline
# results, feature_importance, ensemble = run_model_pipeline(X, y)


# Print new distribution to verify balancing
print("\nBalanced class distribution:")
print(pd.Series(y).value_counts())

# Use balanced data
results, feature_importance, ensemble = run_model_pipeline(X, y)

print(f'\n results: {results} '
      f'\n feature importance {feature_importance} '
      f'\n ensemble {ensemble},')


"""Tune Hyperparameters"""

# print('Tuning XGBoost Hyperparameters')
#  Tune XGBoost hyperparameters
best_xgb_params = tune_xgboost(X, y)
print("\nBest XGBoost parameters:")
print(best_xgb_params)

# Tune lightgbm hyperparameters
best_lgb_params = tune_lightgbm(X, y)
print("\nBest lightgbm parameters:")
print(best_lgb_params)


"""Model Development"""
# First, process categorical columns
X_processed = X.copy()
XT_processed = XT.copy()
categorical_columns = X_processed.select_dtypes(include=['object']).columns
print("Processing categorical columns:", categorical_columns.tolist())

# Convert categorical columns for both training and test data
for col in categorical_columns:
    print(f"\nProcessing column: {col}")

    # Convert all values to strings first to ensure uniform type
    X_processed[col] = X_processed[col].astype(str)
    XT_processed[col] = XT_processed[col].astype(str)

    print(f"Unique values in training: {X_processed[col].unique()}")
    print(f"Unique values in test: {XT_processed[col].unique()}")

    if X_processed[col].nunique() == 2:
        # Binary categories
        le = LabelEncoder()
        # Combine unique values from both training and test
        all_values = pd.concat([X_processed[col], XT_processed[col]]).unique()
        le.fit(all_values)
        X_processed[col] = le.transform(X_processed[col])
        XT_processed[col] = le.transform(XT_processed[col])
    else:
        # Multi-class categories
        dummies_train = pd.get_dummies(X_processed[col], prefix=col)
        dummies_test = pd.get_dummies(XT_processed[col], prefix=col)

        # Add missing columns to each
        missing_in_train = set(dummies_test.columns) - set(dummies_train.columns)
        missing_in_test = set(dummies_train.columns) - set(dummies_test.columns)

        for col_name in missing_in_train:
            dummies_train[col_name] = 0
        for col_name in missing_in_test:
            dummies_test[col_name] = 0

        # Ensure columns are in the same order
        dummies_train = dummies_train.reindex(columns=sorted(dummies_train.columns))
        dummies_test = dummies_test.reindex(columns=sorted(dummies_test.columns))

        # Drop one category to avoid multicollinearity
        dummies_train = dummies_train.iloc[:, 1:]
        dummies_test = dummies_test.iloc[:, 1:]

        X_processed = pd.concat([X_processed.drop(col, axis=1), dummies_train], axis=1)
        XT_processed = pd.concat([XT_processed.drop(col, axis=1), dummies_test], axis=1)

print("\nFinal shapes:")
print(f"Training data shape: {X_processed.shape}")
print(f"Test data shape: {XT_processed.shape}")


# Create optimized models with processed data
optimized_models = {
    'xgboost': xgb.XGBClassifier(**best_xgb_params),
    'lightgbm': lgb.LGBMClassifier(**best_lgb_params)
}

# Fit models with processed data
print("\nFitting models...")
for name, model in optimized_models.items():
    print(f"Fitting {name}...")
    model.fit(X_processed, y)

# Create and fit ensemble with processed data
print("\nCreating and fitting ensemble...")
final_ensemble = create_optimized_ensemble(optimized_models, X_processed, y)
final_ensemble.fit(X_processed, y)

# Make predictions using processed data
print("\nMaking predictions...")
predictions = {}
for name, model in optimized_models.items():
    print(f"Predicting with {name}...")
    predictions[name] = model.predict(XT_processed)

# Ensemble predictions
predictions['ensemble'] = final_ensemble.predict(XT_processed)

"""Saving Predictions """
# Load original test CSV to keep customer IDs
original_test = pd.read_csv('https://raw.githubusercontent.com/opeokupe/capstone-ai-ml/main/data/test.csv')

# Create predictions DataFrame
predictions_df = pd.DataFrame({
    'customer_id': original_test['customer_id'],
    'predicted_churn_risk': predictions['ensemble']
})

# Save predictions
predictions_df.to_csv('final_predictions.csv', index=False)

# Show distribution of predictions
print("\nPredictions distribution:")
print(pd.Series(predictions['ensemble']).value_counts().sort_index())




""" yt """
yt = predictions['ensemble']
print("\nTrain variable unique values:")
print(y.unique())
print("\nTest target variable unique values:")
print(pd.Series(yt).unique())


# test_processed.to_csv('processed_test.csv', index=False)
#
# print('\nPredictions')
# print(predictions)
# print('\nend of predictions')
#
# print('\n predictions are:')
# print(predictions['ensemble'])


# 7. Correlation Analysis
plt.figure(figsize=(12, 8))

# Select only numerical features for correlation analysis
numerical_features = train_processed.select_dtypes(include=['number']).columns
sns.heatmap(train_processed[numerical_features].corr(), annot=True, cmap='coolwarm')


plt.title('Feature Correlation Matrix')
plt.show()
