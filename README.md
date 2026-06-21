# Employee Attrition Analysis Using Machine Learning

---

## Project Overview

Employee attrition, also known as employee turnover, is one of the most critical challenges faced by organizations worldwide. When talented employees leave, companies lose valuable human capital, face reduced productivity, and incur significant costs in hiring and training replacements. This project leverages the power of **Machine Learning** to predict which employees are at risk of leaving, enabling HR teams to take proactive retention measures.

This is a complete, production-ready, end-to-end machine learning project built with **Python** and **Scikit-Learn**. It handles everything from data loading and cleaning to model training, evaluation, and saving. The code is beginner-friendly, fully commented, and follows industry-standard practices.

---

## Business Problem

**Why is employee attrition costly?**

According to industry research, replacing a single employee can cost anywhere from **50% to 200%** of their annual salary. These costs include:

- **Recruitment expenses** (job postings, agency fees, interview time)
- **Onboarding and training costs** for new hires
- **Lost productivity** during the transition period
- **Loss of institutional knowledge** and client relationships
- **Decreased team morale** and increased workload on remaining staff

By predicting attrition risk early, organizations can:
- Identify at-risk employees before they resign
- Implement targeted retention strategies
- Reduce turnover costs significantly
- Improve overall employee satisfaction and engagement

---

## Project Objectives

- Perform comprehensive Exploratory Data Analysis (EDA) to understand attrition patterns
- Build and compare multiple classification models to predict employee attrition
- Identify the most important factors that contribute to employee turnover
- Generate professional visualizations and metrics for reporting
- Save the best-performing model for deployment
- Provide a fully documented, beginner-friendly codebase

---

## Dataset Information

The dataset used in this project is inspired by the **IBM HR Analytics Employee Attrition Dataset**. It contains **1,470 employee records** with **35 features** covering demographics, job roles, satisfaction levels, and work history.

**Dataset Size:** 1,470 rows x 35 columns  
**Target Variable:** Attrition (Yes = Employee Left, No = Employee Stayed)  
**Attrition Rate:** ~16% (indicative of a realistic imbalanced classification problem)

---

## Feature Descriptions

| Feature | Description | Type |
|---------|-------------|------|
| Age | Employee age in years | Numerical |
| Attrition | Whether employee left the company (Yes/No) | Categorical (Target) |
| BusinessTravel | Frequency of business travel | Categorical |
| DailyRate | Daily salary rate | Numerical |
| Department | Employee's department | Categorical |
| DistanceFromHome | Distance from home to work (miles) | Numerical |
| Education | Education level (1-Below College to 5-Doctor) | Ordinal |
| EducationField | Field of education | Categorical |
| EmployeeCount | Count (always 1, constant) | Numerical |
| EmployeeNumber | Employee ID (identifier) | Numerical |
| EnvironmentSatisfaction | Satisfaction with work environment (1-4) | Ordinal |
| Gender | Employee gender | Categorical |
| HourlyRate | Hourly salary rate | Numerical |
| JobInvolvement | Job involvement level (1-4) | Ordinal |
| JobLevel | Job level (1-5) | Ordinal |
| JobRole | Employee's job role | Categorical |
| JobSatisfaction | Job satisfaction level (1-4) | Ordinal |
| MaritalStatus | Marital status | Categorical |
| MonthlyIncome | Monthly salary | Numerical |
| MonthlyRate | Monthly salary rate | Numerical |
| NumCompaniesWorked | Number of companies worked at | Numerical |
| Over18 | Whether over 18 (constant 'Y') | Categorical |
| OverTime | Whether employee works overtime | Categorical |
| PercentSalaryHike | Percentage salary increase | Numerical |
| PerformanceRating | Performance rating (3-4) | Ordinal |
| RelationshipSatisfaction | Relationship satisfaction (1-4) | Ordinal |
| StandardHours | Standard working hours (constant 80) | Numerical |
| StockOptionLevel | Stock option level (0-3) | Ordinal |
| TotalWorkingYears | Total years of work experience | Numerical |
| TrainingTimesLastYear | Training sessions attended last year | Numerical |
| WorkLifeBalance | Work-life balance rating (1-4) | Ordinal |
| YearsAtCompany | Years at current company | Numerical |
| YearsInCurrentRole | Years in current job role | Numerical |
| YearsSinceLastPromotion | Years since last promotion | Numerical |
| YearsWithCurrManager | Years with current manager | Numerical |

---

## Technology Stack

| Technology | Purpose |
|------------|---------|
| Python 3.x | Primary programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Seaborn | Statistical data visualization |
| Scikit-Learn | Machine learning models and evaluation |
| Pickle | Model serialization and saving |

---

## Machine Learning Workflow

### Phase 1: Import Libraries
Import all required Python libraries for data analysis, visualization, and machine learning.

### Phase 2: Load Dataset
Load the employee attrition CSV file into a Pandas DataFrame for analysis.

### Phase 3: Dataset Exploration
Examine the dataset shape, column information, data types, and statistical summaries to understand the data structure.

### Phase 4: Missing Value Analysis
Check for missing or null values in each column to determine if data cleaning is required.

### Phase 5: Data Cleaning
Handle missing values using appropriate imputation techniques:
- **Numerical columns:** Median imputation (robust against outliers)
- **Categorical columns:** Mode imputation (most frequent value)

### Phase 6: Label Encoding
Convert all categorical columns into numerical format using Scikit-Learn's LabelEncoder, making them suitable for machine learning algorithms.

### Phase 7: Exploratory Data Analysis (EDA)
Generate visualizations to uncover patterns and relationships in the data, including attrition distribution, correlation analysis, and feature relationships.

### Phase 8: Feature Engineering
Separate features (X) and target (y), drop irrelevant columns (identifiers, constants), and prepare data for modeling.

### Phase 9: Train-Test Split
Split the dataset into training (80%) and testing (20%) sets to evaluate model performance on unseen data. Stratified splitting ensures balanced class distribution.

### Phase 10: Model Training
Train three classification algorithms on the training data to learn patterns associated with employee attrition.

### Phase 11: Prediction
Use trained models to predict attrition on the test dataset.

### Phase 12: Evaluation
Evaluate model performance using accuracy, precision, recall, F1 score, classification report, and confusion matrix.

### Phase 13: Feature Importance Analysis
Analyze which features most strongly influence attrition predictions using Random Forest feature importance.

### Phase 14: Model Saving
Save the best-performing model using Pickle for future deployment and inference.

### Phase 15: Final Summary
Generate comprehensive output files summarizing the dataset, model metrics, and feature importance.

---

## Algorithms Used

### Logistic Regression
Logistic Regression is a statistical model that predicts the probability of a binary outcome (Yes/No). It works by fitting a logistic function (S-shaped curve) to the data. Despite its name, it is used for **classification**, not regression. It is simple, interpretable, and serves as an excellent baseline model. It tells us how each feature contributes to the probability of attrition in a linear manner.

### Decision Tree Classifier
A Decision Tree is a tree-like model where each internal node represents a decision based on a feature, each branch represents the outcome of that decision, and each leaf represents a class label (Yes/No). It is intuitive and easy to visualize. Decision Trees can capture non-linear relationships in the data and are very interpretable, but they can overfit if not properly constrained.

### Random Forest Classifier
Random Forest is an **ensemble learning** method that builds multiple Decision Trees (a "forest") and combines their predictions. Each tree is trained on a random subset of the data and features. By averaging the predictions of many trees, Random Forest reduces overfitting and generally achieves higher accuracy than a single Decision Tree. It also provides feature importance scores, making it valuable for understanding which factors drive attrition.

---

## Feature Importance Analysis

The Random Forest model identifies the most influential factors driving employee attrition. Understanding these factors helps HR teams design targeted retention strategies:

1. **Monthly Income** - Lower-paid employees are more likely to leave
2. **OverTime** - Employees working overtime are at higher attrition risk
3. **Age** - Younger employees tend to have higher turnover rates
4. **Job Role** - Certain roles have inherently higher attrition
5. **Years at Company** - Newer employees and those at tenure inflection points are vulnerable
6. **Job Satisfaction** - Dissatisfied employees are more likely to leave
7. **Distance from Home** - Longer commute distances increase attrition risk
8. **Stock Option Level** - Lower stock ownership correlates with higher attrition
9. **Years Since Last Promotion** - Lack of career growth drives turnover
10. **Work-Life Balance** - Poor work-life balance is a key attrition driver

---

## Project Structure

```
CodeTech_Employee_Attrition_Analysis/
│
├── data/
│   └── employee_attrition.csv          # Employee attrition dataset
│
├── screenshots/
│   ├── attrition_distribution.png      # Target variable distribution
│   ├── correlation_heatmap.png         # Feature correlation matrix
│   ├── overtime_vs_attrition.png       # Overtime impact analysis
│   ├── jobrole_vs_attrition.png        # Job role attrition rates
│   ├── confusion_matrix.png            # Best model confusion matrix
│   └── feature_importance.png          # Top feature importance chart
│
├── outputs/
│   ├── dataset_summary.txt             # Complete dataset exploration report
│   ├── model_metrics.txt               # All model evaluation metrics
│   └── feature_importance.txt          # Feature importance rankings
│
├── models/
│   ├── employee_attrition_model.pkl    # Best trained model (Pickle)
│   └── label_encoders.pkl              # Label encoders for deployment
│
├── employee_attrition.py              # Main Python script (entry point)
├── generate_data.py                   # Synthetic data generator
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── .gitignore                         # Git ignore rules
```

---

## Installation Guide

Follow these steps carefully. No technical background is required.

### Step 1: Install Python

1. Go to [python.org](https://www.python.org/downloads/)
2. Download the latest Python 3.x version (e.g., Python 3.11 or 3.12)
3. Run the installer
4. **IMPORTANT:** Check the box that says **"Add Python to PATH"** before clicking Install
5. Complete the installation

### Step 2: Download the Project

- **Option A (GitHub):** Clone the repository using Git:
  ```bash
  git clone https://github.com/your-username/CodeTech_Employee_Attrition_Analysis.git
  ```
- **Option B (Direct Download):** Download the ZIP file and extract it to a folder

### Step 3: Open VS Code (or any code editor)

1. Install [VS Code](https://code.visualstudio.com/) if you don't have it
2. Open VS Code
3. Click **File > Open Folder** and select the project folder

### Step 4: Install Dependencies

Open the terminal in VS Code:
- **Windows:** `Ctrl + `` (backtick)
- **Mac:** `Cmd + ``

Then type the following command and press Enter:

```bash
pip install -r requirements.txt
```

This will install all required Python libraries (Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn).

### Step 5: Run the Project

In the terminal, type the following command and press Enter:

```bash
python employee_attrition.py
```

The script will:
- Load and analyze the dataset
- Train machine learning models
- Generate visualizations and metrics
- Save the best model

All outputs will be saved in the `outputs/`, `screenshots/`, and `models/` folders.

---

## Outputs Generated

After running the project, the following outputs are generated:

| Output File | Location | Description |
|-------------|----------|-------------|
| Dataset Summary | `outputs/dataset_summary.txt` | Complete data exploration report including shape, types, statistics, and missing values |
| Model Metrics | `outputs/model_metrics.txt` | Performance metrics for all models including accuracy, precision, recall, F1, and confusion matrices |
| Feature Importance | `outputs/feature_importance.txt` | Ranking of all features by their importance in predicting attrition |
| Attrition Distribution | `screenshots/attrition_distribution.png` | Bar chart showing the distribution of the target variable |
| Correlation Heatmap | `screenshots/correlation_heatmap.png` | Heatmap showing correlations between all numerical features |
| Overtime vs Attrition | `screenshots/overtime_vs_attrition.png` | Stacked bar chart comparing attrition across overtime categories |
| Job Role vs Attrition | `screenshots/jobrole_vs_attrition.png` | Stacked bar chart showing attrition rates by job role |
| Confusion Matrix | `screenshots/confusion_matrix.png` | Confusion matrix for the best performing model |
| Feature Importance | `screenshots/feature_importance.png` | Horizontal bar chart of top 10 most important features |
| Trained Model | `models/employee_attrition_model.pkl` | Pickle file of the best model ready for deployment |

---

## Evaluation Metrics

The following metrics are used to evaluate model performance:

| Metric | Description | Formula |
|--------|-------------|---------|
| **Accuracy** | Proportion of correct predictions out of total predictions | (TP + TN) / (TP + TN + FP + FN) |
| **Precision** | Proportion of positive identifications that were actually correct | TP / (TP + FP) |
| **Recall** | Proportion of actual positives that were identified correctly | TP / (TP + FN) |
| **F1 Score** | Harmonic mean of Precision and Recall | 2 x (Precision x Recall) / (Precision + Recall) |

Where:
- **TP** = True Positives (correctly predicted attrition)
- **TN** = True Negatives (correctly predicted retention)
- **FP** = False Positives (incorrectly predicted attrition)
- **FN** = False Negatives (missed attrition cases)

**Note:** For imbalanced datasets (like employee attrition), **F1 Score** and **Recall** are more informative than Accuracy alone, since a model could achieve high accuracy by simply predicting "No" for everyone.

---

## Screenshots

### Employee Attrition Distribution
![Attrition Distribution](screenshots/attrition_distribution.png)

### Correlation Heatmap
![Correlation Heatmap](screenshots/correlation_heatmap.png)

### Overtime vs Attrition
![Overtime vs Attrition](screenshots/overtime_vs_attrition.png)

### Job Role vs Attrition
![Job Role vs Attrition](screenshots/jobrole_vs_attrition.png)

### Confusion Matrix (Best Model)
![Confusion Matrix](screenshots/confusion_matrix.png)

### Feature Importance
![Feature Importance](screenshots/feature_importance.png)

---

## Future Enhancements

- **XGBoost / LightGBM / CatBoost:** Implement gradient boosting algorithms for potentially higher accuracy
- **Hyperparameter Tuning:** Use GridSearchCV or RandomizedSearchCV to optimize model parameters
- **Streamlit Dashboard:** Build an interactive web application for HR teams to explore predictions
- **HR Analytics Dashboard:** Create a comprehensive dashboard with real-time attrition monitoring
- **Real-Time Employee Risk Prediction:** Deploy the model as an API for continuous risk assessment
- **Deep Learning:** Experiment with neural networks for complex pattern recognition
- **Cross-Validation:** Implement k-fold cross-validation for more robust evaluation
- **Feature Selection:** Apply automated feature selection techniques to reduce dimensionality
- **Explainable AI (XAI):** Use SHAP or LIME to provide interpretable explanations for individual predictions

---

## Learning Outcomes

By studying this project, you will learn:

- How to structure a complete machine learning project from scratch
- How to perform data exploration and visualization using Pandas, Matplotlib, and Seaborn
- How to handle categorical data using Label Encoding
- How to train, evaluate, and compare multiple classification algorithms
- How to interpret model performance using precision, recall, F1 score, and confusion matrix
- How to perform feature importance analysis to identify key drivers
- How to save and deploy machine learning models using Pickle
- How to document a project professionally with a comprehensive README
- How to follow PEP8 coding standards and maintain clean code
- How to work with imbalanced classification problems

---

## Author

**Vijayaragavan U**

B.E Computer Science and Engineering  
Saranathan College of Engineering  
Tiruchirappalli, Tamil Nadu, India

---

## License

This project is open source and available for educational and professional use.

---

## Acknowledgments

- IBM for the original HR Analytics dataset inspiration
- Scikit-Learn, Pandas, NumPy, Matplotlib, and Seaborn communities for excellent documentation
- CodeTech IT Solutions for the internship opportunity and project guidance

---

*Built with passion for data science, machine learning, and HR analytics.*
