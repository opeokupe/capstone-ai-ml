# Churn Rate Analysis & Data Preprocessing
## Imperial College Capstone Project

# 1. Initial Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')


# 2. Initial Data Exploration
## 2.1 Basic Data Overview
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


# Explore training data
explore_data(train_data, "Training Dataset")


## 2.2 Data Distribution Analysis


def analyze_distributions(df):
    """
   Analyzes and plots distributions of numerical features
   :param df:
   :return:
   """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(15, len(numerical_cols) * 4))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(len(numerical_cols), 2, i * 2 - 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')

        plt.subplot(len(numerical_cols), 2, i * 2)
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()


# Plot distributions
analyze_distributions(train_data)

# 3. Bias Analysis
"""
Document potential biases in the dataset:
- Selection bias: [Your analysis here]
- Measurement bias: [Your analysis here]
- Demographic representation: [Your analysis here]
- Time-related biases: [Your analysis here]
"""


# 4. Feature Engineering
def engineer_features(df):
    """
    Creates new features based on existing data
    Returns DataFrame with new features
    """
    df_processed = df.copy()

    # Example feature engineering (modify based on your specific needs):

    # 4.1 Time-based features
    if 'last_login' in df.columns:
        df_processed['days_since_login'] = (pd.to_datetime('now') - pd.to_datetime(df['last_login'])).dt.days

    # 4.2 Interaction features
    if 'visit_frequency' in df.columns and 'duration' in df.columns:
        df_processed['engagement_score'] = df['visit_frequency'] * df['duration']

    # 4.3 Categorical encoding
    # Add your categorical encoding logic here

    return df_processed


# Apply feature engineering
train_data_processed = engineer_features(train_data)
test_data_processed = engineer_features(test_data)


# 5. Data Preprocessing
def preprocess_data(df):
    """
    Applies all preprocessing steps:
    - Handles missing values
    - Scales numerical features
    - Encodes categorical variables
    Returns preprocessed DataFrame
    """
    df_processed = df.copy()

    # 5.1 Handle missing values
    # Add your missing value handling logic here

    # 5.2 Scale numerical features
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])

    # 5.3 Additional preprocessing steps
    # Add any additional preprocessing steps here

    return df_processed


# Apply preprocessing
train_data_final = preprocess_data(train_data_processed)
test_data_final = preprocess_data(test_data_processed)

# 6. Save Processed Data
# Save the processed datasets
train_data_final.to_csv('../data/processed_train.csv', index=False)
test_data_final.to_csv('../data/processed_test.csv', index=False)

# 7. Correlation Analysis
plt.figure(figsize=(12, 8))
# Select only numerical features for correlation analysis
numerical_features = train_data_final.select_dtypes(include=['number']).columns
sns.heatmap(train_data_final[numerical_features].corr(), annot=True, cmap='coolwarm')

plt.title('Feature Correlation Matrix')
plt.show()

"""
Final Notes and Observations:

1. Data Quality:
   - [Document your findings about data quality]
   - [Note any concerning patterns or issues]

2. Feature Engineering Decisions:
   - [Document why you created each new feature]
   - [Explain the expected impact on the model]

3. Preprocessing Decisions:
   - [Explain your choice of scaling methods]
   - [Document handling of outliers]
   - [Explain treatment of missing values]

4. Next Steps:
   - [List any additional analyses needed]
   - [Note potential model considerations based on this analysis]
"""


# excluded_columns = ['customer_id','security_no', 'Name']
# non_numerical_cols = train_data.select_dtypes(exclude=['int64', 'float64']).columns
# filtered_columns = [col for col in non_numerical_cols if col not in excluded_columns]


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
    plt.figure(figsize=(15, 5 * len(categorical_cols)))

    for i, col in enumerate(categorical_cols, 1):
        # Get value counts and calculate percentages
        value_counts = df[col].value_counts()
        value_percentages = df[col].value_counts(normalize=True) * 100

        # Create subplot for bar plot
        plt.subplot(len(categorical_cols), 2, i * 2 - 1)
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Count')

        # Add percentage labels on top of bars - CORRECTED THIS PART
        for idx, (j, v) in enumerate(zip(value_counts.index, value_counts.values)):
            plt.text(idx, v, f'{value_percentages.iloc[idx]:.1f}%',
                     ha='center', va='bottom')

        # Create subplot for pie chart
        plt.subplot(len(categorical_cols), 2, i * 2)
        plt.pie(value_counts.values, labels=value_counts.index,
                autopct='%1.1f%%', startangle=90)
        plt.title(f'Percentage Distribution of {col}')

        # Print summary statistics
        print(f"\nSummary for {col}:")
        print("-" * 50)
        print(f"Number of unique values: {df[col].nunique()}")
        print(f"Most common value: {df[col].mode().iloc[0]} ({value_percentages.iloc[0]:.1f}%)")
        print(f"Missing values: {df[col].isnull().sum()} ({(df[col].isnull().sum() / len(df)) * 100:.1f}%)")
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


def analyze_categorical_relationships(df, target_col):
    """
    Analyzes relationships between categorical variables and target
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if col != target_col]

    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        pd.crosstab(df[col], df[target_col], normalize='index').plot(kind='bar')
        plt.title(f'{col} vs {target_col}')
        plt.xlabel(col)
        plt.ylabel(f'Proportion of {target_col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


excluded_columns = ['customer_id', 'security_no', 'Name']
filtered_columns = [col for col in train_data if col not in excluded_columns]
filtered_data = train_data[filtered_columns]

check_cardinality(train_data[filtered_data])

analyze_categorical_relationships(filtered_data, 'churn_risk_score')

analyze_non_numeric_distributions(filtered_data)
