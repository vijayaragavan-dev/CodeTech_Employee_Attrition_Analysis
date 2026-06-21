"""
===============================================================================
EMPLOYEE ATTRITION ANALYSIS USING MACHINE LEARNING
===============================================================================
Project : Employee Attrition Prediction System
Author  : Vijayaragavan U
Domain  : HR Analytics, Machine Learning
Tool    : Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn

Objective:
    Predict whether an employee is likely to leave the organization
    based on various demographic, job-related, and satisfaction factors.

Algorithms Used:
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier

Dataset:
    IBM HR Analytics Employee Attrition Dataset (Synthetic)
===============================================================================
"""

# =============================================================================
# PHASE 1: IMPORT LIBRARIES
# =============================================================================

import pandas as pd                    # Data manipulation and analysis
import numpy as np                     # Numerical computing
import matplotlib.pyplot as plt        # Data visualization
import seaborn as sns                  # Statistical data visualization
import warnings                        # Suppress warnings for cleaner output

from sklearn.model_selection import train_test_split   # Splitting dataset
from sklearn.preprocessing import LabelEncoder         # Encode categorical data
from sklearn.linear_model import LogisticRegression    # Logistic Regression model
from sklearn.tree import DecisionTreeClassifier        # Decision Tree model
from sklearn.ensemble import RandomForestClassifier    # Random Forest model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)                                                        # Evaluation metrics
import pickle                          # Save trained model to disk
import io                               # For capturing df.info() output

warnings.filterwarnings('ignore')      # Ignore warnings for cleaner output

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150

print("=" * 70)
print("  EMPLOYEE ATTRITION ANALYSIS USING MACHINE LEARNING")
print("=" * 70)

# =============================================================================
# PHASE 2: LOAD DATASET
# =============================================================================

print("\n[PHASE 2] Loading Dataset...")
print("-" * 40)

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('data/employee_attrition.csv')

print(f"Dataset loaded successfully!")
print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# =============================================================================
# PHASE 3: DATASET EXPLORATION
# =============================================================================

print("\n[PHASE 3] Dataset Exploration...")
print("-" * 40)

# Display the first 5 rows to understand the data structure
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display dataset information (column names, data types, non-null counts)
print("\nDataset Info:")
buffer = io.StringIO()
df.info(buf=buffer)
info_text = buffer.getvalue()
print(info_text)

# Check data types of each column
print("\nData Types:")
print(df.dtypes)

# Statistical summary of numerical columns
print("\nStatistical Summary (Numerical Columns):")
print(df.describe())

# Statistical summary of categorical columns
print("\nStatistical Summary (Categorical Columns):")
print(df.describe(include='object'))

# =============================================================================
# PHASE 4: MISSING VALUE ANALYSIS
# =============================================================================

print("\n[PHASE 4] Missing Value Analysis...")
print("-" * 40)

# Count missing values in each column
missing_values = df.isnull().sum()
print("\nMissing Values per Column:")
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found in the dataset!")

# Calculate percentage of missing values
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("\nMissing Value Percentages:")
print(missing_percentage[missing_percentage > 0] if missing_percentage.sum() > 0 else "All columns have 0% missing values.")

# =============================================================================
# PHASE 5: DATA CLEANING
# =============================================================================

print("\n[PHASE 5] Data Cleaning...")
print("-" * 40)

# Introduce some artificial missing values for demonstration purposes
# (Since the original synthetic data has no missing values, we simulate the cleaning process)
print("\nCreating a copy of dataset for cleaning demonstration...")
df_clean = df.copy()

# For demonstration, we show how missing values WOULD be handled:
# - Numerical columns: Impute with median (robust to outliers)
# - Categorical columns: Impute with mode (most frequent value)

# Identify numerical and categorical columns
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical Columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical Columns ({len(categorical_cols)}): {categorical_cols}")

# Since our data has no missing values, we report it clean
print("\nData Cleaning Complete - No missing values detected in the dataset.")
print("If missing values were present, the following approach would be used:")
print("  - Numerical columns: Median imputation")
print("  - Categorical columns: Mode imputation")

# =============================================================================
# PHASE 6: LABEL ENCODING
# =============================================================================

print("\n[PHASE 6] Label Encoding...")
print("-" * 40)

# Initialize LabelEncoder
label_encoders = {}
df_encoded = df_clean.copy()

# Encode each categorical column using LabelEncoder
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le  # Store encoder for later use (e.g., inverse transform)
    print(f"  Encoded '{col}' -> Unique values: {len(le.classes_)} | Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print(f"\nAll {len(categorical_cols)} categorical columns encoded successfully.")

# =============================================================================
# PHASE 7: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATIONS
# =============================================================================

print("\n[PHASE 7] Exploratory Data Analysis & Visualizations...")
print("-" * 40)

# --- Visualization 1: Employee Attrition Distribution ---
plt.figure(figsize=(8, 6))
attrition_counts = df['Attrition'].value_counts()
colors = ['#4CAF50', '#F44336']
bars = plt.bar(attrition_counts.index, attrition_counts.values, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bar, count in zip(bars, attrition_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.title('Employee Attrition Distribution', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Attrition', fontsize=14)
plt.ylabel('Number of Employees', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('screenshots/attrition_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: screenshots/attrition_distribution.png")

# --- Visualization 2: Correlation Heatmap ---
plt.figure(figsize=(20, 16))
correlation_matrix = df_encoded.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=20, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('screenshots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: screenshots/correlation_heatmap.png")

# --- Visualization 3: Overtime vs Attrition ---
plt.figure(figsize=(8, 6))
overtime_ct = pd.crosstab(df['OverTime'], df['Attrition'])
overtime_ct.plot(kind='bar', stacked=True, color=['#4CAF50', '#F44336'], edgecolor='black', linewidth=1.2)
plt.title('Impact of Overtime on Employee Attrition', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Overtime', fontsize=14)
plt.ylabel('Number of Employees', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Attrition', title_fontsize=12, fontsize=11)
plt.tight_layout()
plt.savefig('screenshots/overtime_vs_attrition.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: screenshots/overtime_vs_attrition.png")

# --- Visualization 4: Job Role vs Attrition ---
plt.figure(figsize=(14, 7))
jobrole_ct = pd.crosstab(df['JobRole'], df['Attrition'])
jobrole_ct.plot(kind='bar', stacked=True, color=['#4CAF50', '#F44336'], edgecolor='black', linewidth=1.2)
plt.title('Employee Attrition Across Different Job Roles', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Job Role', fontsize=14)
plt.ylabel('Number of Employees', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=12)
plt.legend(title='Attrition', title_fontsize=12, fontsize=11)
plt.tight_layout()
plt.savefig('screenshots/jobrole_vs_attrition.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: screenshots/jobrole_vs_attrition.png")

# =============================================================================
# PHASE 8: FEATURE ENGINEERING
# =============================================================================

print("\n[PHASE 8] Feature Engineering...")
print("-" * 40)

# Separate features (X) and target (y)
# Drop columns that are identifiers or constants (not useful for prediction)
columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
X = df_encoded.drop(columns=['Attrition'] + columns_to_drop, errors='ignore')
y = df_encoded['Attrition']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nTarget distribution:\n{y.value_counts()}")
print(f"\nFeatures used for training: {X.columns.tolist()}")

# =============================================================================
# PHASE 9: TRAIN-TEST SPLIT
# =============================================================================

print("\n[PHASE 9] Train-Test Split...")
print("-" * 40)

# Split dataset into training (80%) and testing (20%) sets
# stratify=y ensures balanced class distribution in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Training attrition distribution:\n{y_train.value_counts()}")
print(f"Testing attrition distribution:\n{y_test.value_counts()}")

# =============================================================================
# PHASE 10: MODEL TRAINING
# =============================================================================

print("\n[PHASE 10] Model Training...")
print("-" * 40)

# Initialize models with class_weight='balanced' to handle imbalanced attrition classes
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
}

trained_models = {}

# Train each model
for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"  {name} training completed!")

# =============================================================================
# PHASE 11: PREDICTION
# =============================================================================

print("\n[PHASE 11] Making Predictions...")
print("-" * 40)

predictions = {}

for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    print(f"  {name} predictions completed. Sample: {y_pred[:10]}")

# =============================================================================
# PHASE 12: MODEL EVALUATION
# =============================================================================

print("\n[PHASE 12] Model Evaluation...")
print("-" * 40)

# Store results for comparison
results = []

print("\n" + "=" * 70)
print("  MODEL PERFORMANCE COMPARISON")
print("=" * 70)

for name, y_pred in predictions.items():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

    print(f"\n  {name}")
    print(f"  {'=' * 40}")
    print(f"  Accuracy   : {accuracy:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"\n  Classification Report:")
    print(f"  {classification_report(y_test, y_pred)}")

# Create comparison DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)

print("\n" + "=" * 70)
print("  MODEL COMPARISON SUMMARY (Sorted by F1 Score)")
print("=" * 70)
print(results_df.to_string(index=False))

# Identify the best model
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
print(f"\n  Best Performing Model: {best_model_name}")
print(f"  F1 Score: {results_df.iloc[0]['F1 Score']:.4f}")
print(f"  Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")

# --- Confusion Matrix for Best Model ---
best_predictions = predictions[best_model_name]
cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Stayed (No)', 'Left (Yes)'],
            yticklabels=['Stayed (No)', 'Left (Yes)'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.tight_layout()
plt.savefig('screenshots/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: screenshots/confusion_matrix.png")

# =============================================================================
# PHASE 13: FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\n[PHASE 13] Feature Importance Analysis...")
print("-" * 40)

# Random Forest is used for feature importance since it's tree-based
rf_model = trained_models['Random Forest']
feature_importances = rf_model.feature_importances_

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print(f"\nTop 10 Most Important Features (Random Forest):")
print(f"{'-' * 60}")
print(f"{'Rank':<5} {'Feature':<35} {'Importance':<15}")
print(f"{'-' * 60}")
for i, row in feature_importance_df.head(10).iterrows():
    print(f"{i+1:<5} {row['Feature']:<35} {row['Importance']:.6f}".format(i+1, row['Feature'], row['Importance']))

# Plot Feature Importance (Top 10)
plt.figure(figsize=(12, 7))
top_10 = feature_importance_df.head(10)
bars = plt.barh(range(len(top_10)), top_10['Importance'].values, color='teal', edgecolor='black', linewidth=1.2)
plt.yticks(range(len(top_10)), top_10['Feature'].values, fontsize=11)
plt.xlabel('Importance Score', fontsize=14)
plt.title('Top 10 Features Influencing Employee Attrition', fontsize=16, fontweight='bold', pad=20)
plt.gca().invert_yaxis()

# Add value labels
for bar, val in zip(bars, top_10['Importance'].values):
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('screenshots/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: screenshots/feature_importance.png")

# =============================================================================
# PHASE 14: MODEL SAVING
# =============================================================================

print("\n[PHASE 14] Saving the Best Model...")
print("-" * 40)

# Save the best performing model using pickle
model_filename = 'models/employee_attrition_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)

# Also save the label encoders for deployment (needed to transform new data)
encoders_filename = 'models/label_encoders.pkl'
with open(encoders_filename, 'wb') as f:
    pickle.dump(label_encoders, f)

print(f"\n  Best Model ({best_model_name}) saved to: {model_filename}")
print(f"  Label Encoders saved to: {encoders_filename}")
print(f"\n  {'=' * 40}")
print(f"  Employee Attrition Model Saved Successfully!")
print(f"  {'=' * 40}")

# =============================================================================
# PHASE 15: FINAL SUMMARY & OUTPUT FILES
# =============================================================================

print("\n[PHASE 15] Generating Final Summary...")
print("-" * 40)

# --- Save Dataset Summary ---
with open('outputs/dataset_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("  DATASET SUMMARY - EMPLOYEE ATTRITION ANALYSIS\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")

    f.write("Column Names and Data Types:\n")
    f.write("-" * 50 + "\n")
    for col in df.columns:
        f.write(f"  {col:<35} {df[col].dtype}\n")

    f.write("\nNumerical Columns Statistical Summary:\n")
    f.write("-" * 50 + "\n")
    f.write(df.describe().to_string())
    f.write("\n\nCategorical Columns Statistical Summary:\n")
    f.write("-" * 50 + "\n")
    f.write(df.describe(include='object').to_string())

    f.write("\n\nMissing Value Analysis:\n")
    f.write("-" * 50 + "\n")
    f.write(f"Total Missing Values: {df.isnull().sum().sum()}\n")
    f.write(df.isnull().sum().to_string())

    f.write("\n\nTarget Variable (Attrition) Distribution:\n")
    f.write("-" * 50 + "\n")
    f.write(df['Attrition'].value_counts().to_string())
    f.write("\n")
    f.write(df['Attrition'].value_counts(normalize=True).mul(100).round(2).to_string())
    f.write("\n")

print("  Saved: outputs/dataset_summary.txt")

# --- Save Model Metrics ---
with open('outputs/model_metrics.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("  MODEL EVALUATION METRICS\n")
    f.write("=" * 70 + "\n\n")

    for name, y_pred in predictions.items():
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        cm = confusion_matrix(y_test, y_pred)

        f.write(f"Model: {name}\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Accuracy             : {accuracy:.4f}\n")
        f.write(f"  Precision            : {precision:.4f}\n")
        f.write(f"  Recall               : {recall:.4f}\n")
        f.write(f"  F1 Score             : {f1:.4f}\n\n")
        f.write(f"  Confusion Matrix:\n")
        f.write(f"  {cm}\n\n")
        f.write(f"  Classification Report:\n")
        f.write(f"{classification_report(y_test, y_pred)}\n")
        f.write("=" * 50 + "\n\n")

    f.write("\nModel Comparison (Sorted by F1 Score):\n")
    f.write("-" * 50 + "\n")
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nBest Performing Model: {best_model_name}\n")
    f.write(f"F1 Score: {results_df.iloc[0]['F1 Score']:.4f}\n")
    f.write(f"Accuracy: {results_df.iloc[0]['Accuracy']:.4f}\n")

print("  Saved: outputs/model_metrics.txt")

# --- Save Feature Importance ---
with open('outputs/feature_importance.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("  FEATURE IMPORTANCE ANALYSIS (Random Forest)\n")
    f.write("=" * 70 + "\n\n")

    f.write("Top 10 Most Important Features:\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Rank':<5} {'Feature':<35} {'Importance':<15}\n")
    f.write("-" * 60 + "\n")
    for i, row in feature_importance_df.head(10).iterrows():
        f.write(f"{i+1:<5} {row['Feature']:<35} {row['Importance']:.6f}\n")

    f.write("\n\nAll Features Sorted by Importance:\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Rank':<5} {'Feature':<35} {'Importance':<15}\n")
    f.write("-" * 60 + "\n")
    for i, row in feature_importance_df.iterrows():
        f.write(f"{i+1:<5} {row['Feature']:<35} {row['Importance']:.6f}\n")

print("  Saved: outputs/feature_importance.txt")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("  PROJECT EXECUTION SUMMARY")
print("=" * 70)
print(f"""
  Dataset              : employee_attrition.csv ({df.shape[0]} records, {df.shape[1]} features)
  Target Variable      : Attrition (Yes/No)
  Models Used          : Logistic Regression, Decision Tree, Random Forest
  Best Model           : {best_model_name}
  Best F1 Score        : {results_df.iloc[0]['F1 Score']:.4f}
  Best Accuracy        : {results_df.iloc[0]['Accuracy']:.4f}
  Top Feature          : {feature_importance_df.iloc[0]['Feature']}

  OUTPUTS GENERATED:
  - data/employee_attrition.csv
  - outputs/dataset_summary.txt
  - outputs/model_metrics.txt
  - outputs/feature_importance.txt
  - screenshots/attrition_distribution.png
  - screenshots/correlation_heatmap.png
  - screenshots/overtime_vs_attrition.png
  - screenshots/jobrole_vs_attrition.png
  - screenshots/confusion_matrix.png
  - screenshots/feature_importance.png
  - models/employee_attrition_model.pkl
  - models/label_encoders.pkl
""")
print("=" * 70)
print("  PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 70)
