# 👨‍💼 Employee Attrition Analysis Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Internship](https://img.shields.io/badge/CodeTech-Internship-blueviolet)

---

## Internship Information

| Field              | Details                         |
| ------------------ | ------------------------------- |
| Internship Program | Machine Learning Internship     |
| Organization       | CodeTech IT Solutions Pvt. Ltd. |
| Intern Name        | Vijayaragavan U                 |
| Intern ID          | CITS4915                        |
| Duration           | 4 Weeks                         |
| Internship Period  | 18 June 2026 – 16 July 2026     |
| Domain             | Machine Learning                |
| Project            | Employee Attrition Analysis     |
| Project Type       | HR Analytics                    |

---

## 📖 Project Overview

Employee Attrition Analysis is a Human Resource Analytics project that leverages Machine Learning techniques to predict whether an employee is likely to leave an organization. Employee turnover is a significant challenge for companies because it increases recruitment costs, impacts productivity, and affects overall organizational performance.

This project analyzes employee-related factors such as job role, salary, overtime, work environment, satisfaction levels, and years of experience to identify patterns associated with employee attrition. The developed Machine Learning models help organizations proactively identify at-risk employees and implement retention strategies.

---

## 🎯 Problem Statement

Employee attrition can lead to substantial financial and operational losses for organizations. Human Resource departments need data-driven insights to understand why employees leave and how attrition can be reduced.

The objective of this project is to:

* Predict employee attrition using Machine Learning.
* Identify the most influential factors contributing to attrition.
* Support HR teams in workforce planning.
* Improve employee retention strategies through predictive analytics.

---

## 🚀 Project Objectives

* Perform comprehensive employee data analysis.
* Handle missing values and categorical data.
* Visualize workforce trends and attrition patterns.
* Train multiple Machine Learning classification models.
* Compare model performance using evaluation metrics.
* Identify key features influencing employee attrition.
* Generate HR-focused business insights.
* Save the best-performing model for future predictions.

---

## 📊 Dataset Information

### Dataset Used

IBM HR Analytics Employee Attrition Dataset

### Dataset Description

The dataset contains employee demographic information, job-related details, compensation information, satisfaction levels, and performance indicators.

### Key Features

| Feature                 | Description                         |
| ----------------------- | ----------------------------------- |
| Age                     | Employee Age                        |
| BusinessTravel          | Frequency of Business Travel        |
| Department              | Employee Department                 |
| DistanceFromHome        | Distance Between Home and Workplace |
| Education               | Education Level                     |
| EnvironmentSatisfaction | Workplace Environment Satisfaction  |
| Gender                  | Employee Gender                     |
| JobRole                 | Employee Position                   |
| JobSatisfaction         | Job Satisfaction Level              |
| MonthlyIncome           | Monthly Salary                      |
| OverTime                | Overtime Status                     |
| PercentSalaryHike       | Salary Increment Percentage         |
| TotalWorkingYears       | Total Professional Experience       |
| YearsAtCompany          | Years Worked at Current Company     |
| Attrition               | Employee Attrition Status           |

### Target Variable

| Value | Meaning                    |
| ----- | -------------------------- |
| Yes   | Employee Left Company      |
| No    | Employee Stayed in Company |

---

## 🛠️ Technologies Used

| Category             | Technology          |
| -------------------- | ------------------- |
| Programming Language | Python              |
| Data Processing      | Pandas, NumPy       |
| Data Visualization   | Matplotlib, Seaborn |
| Machine Learning     | Scikit-Learn        |
| Model Persistence    | Pickle              |
| IDE                  | Visual Studio Code  |
| Version Control      | Git & GitHub        |

---

## 🧠 Machine Learning Workflow

### 1. Data Collection

Load employee dataset into a Pandas DataFrame.

### 2. Data Exploration

Analyze:

* Dataset Shape
* Feature Information
* Statistical Summary
* Data Types
* Attrition Distribution

### 3. Data Cleaning

Handle missing values and data inconsistencies to improve model quality.

### 4. Label Encoding

Convert categorical attributes into numerical values for Machine Learning algorithms.

Examples:

* Department
* Gender
* JobRole
* BusinessTravel
* OverTime
* Attrition

### 5. Exploratory Data Analysis

Generate workforce insights through visualizations:

* Employee Attrition Distribution
* Correlation Heatmap
* Overtime vs Attrition
* Job Role vs Attrition

### 6. Feature Engineering

Prepare features for model training and evaluation.

### 7. Train-Test Split

Split the dataset into:

* 80% Training Data
* 20% Testing Data

### 8. Model Training

Train multiple Machine Learning models.

### 9. Model Evaluation

Evaluate performance using:

* Accuracy
* Precision
* Recall
* F1 Score

### 10. Feature Importance Analysis

Identify factors contributing most to employee attrition.

### 11. Model Saving

Save the best-performing model for future HR analytics applications.

---

## 🤖 Machine Learning Algorithms Used

### Logistic Regression

A statistical classification algorithm used for binary prediction problems.

### Decision Tree Classifier

A tree-based model that learns decision patterns from employee attributes.

### Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees to improve prediction performance.

---

## 📈 Model Performance

### Best Performing Model

**Logistic Regression**

### Evaluation Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 54.08% |
| Precision | 52.34% |
| Recall    | 43.37% |
| F1 Score  | 47.47% |

> Note: Employee attrition prediction is a complex HR analytics problem influenced by multiple behavioral and organizational factors. The project focuses on demonstrating a complete Machine Learning workflow and HR analytics methodology.

---

## 📊 Feature Importance Analysis

The project identifies the most influential factors contributing to employee attrition, including:

* Monthly Income
* Overtime Status
* Job Satisfaction
* Years at Company
* Total Working Years
* Environment Satisfaction
* Distance From Home
* Job Role

These insights can assist HR departments in improving employee retention programs.

---

## 📂 Project Structure

```text
CodeTech_Employee_Attrition_Analysis/
│
├── data/
│   └── employee_attrition.csv
│
├── screenshots/
│   ├── attrition_distribution.png
│   ├── correlation_heatmap.png
│   ├── overtime_vs_attrition.png
│   ├── jobrole_vs_attrition.png
│   └── confusion_matrix.png
│
├── outputs/
│   ├── dataset_summary.txt
│   ├── model_metrics.txt
│   └── feature_importance.txt
│
├── models/
│   └── employee_attrition_model.pkl
│
├── employee_attrition.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation Guide

### Step 1: Clone Repository

```bash
git clone https://github.com/vijayaragavan-dev/CodeTech_Employee_Attrition_Analysis.git
```

### Step 2: Navigate to Project Directory

```bash
cd CodeTech_Employee_Attrition_Analysis
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Project

```bash
python employee_attrition.py
```

---

## 📌 Outputs Generated

### Model File

```text
models/employee_attrition_model.pkl
```

### Reports

```text
outputs/dataset_summary.txt
outputs/model_metrics.txt
outputs/feature_importance.txt
```

### Visualizations

```text
screenshots/attrition_distribution.png
screenshots/correlation_heatmap.png
screenshots/overtime_vs_attrition.png
screenshots/jobrole_vs_attrition.png
screenshots/confusion_matrix.png
```

---

## 📷 Project Visualizations

### Employee Attrition Distribution

![Attrition Distribution](screenshots/attrition_distribution.png)

### Correlation Heatmap

![Heatmap](screenshots/correlation_heatmap.png)

### Overtime vs Attrition

![Overtime Analysis](screenshots/overtime_vs_attrition.png)

### Job Role vs Attrition

![Job Role Analysis](screenshots/jobrole_vs_attrition.png)

---

## 💼 Business Impact

This project demonstrates how Machine Learning can support Human Resource departments by:

* Identifying employees at risk of leaving.
* Improving employee retention strategies.
* Reducing recruitment and training costs.
* Supporting workforce planning decisions.
* Enhancing employee satisfaction initiatives.

---

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

* HR Analytics
* Employee Data Analysis
* Classification Algorithms
* Data Visualization
* Feature Importance Analysis
* Workforce Analytics
* Model Evaluation
* Business Insight Generation
* Technical Documentation

---

## 🔮 Future Enhancements

* XGBoost Implementation
* Hyperparameter Optimization
* Advanced Feature Engineering
* Employee Retention Dashboard
* Streamlit Web Application
* Real-Time Attrition Risk Prediction

---

## 👨‍💻 Author

### Vijayaragavan U

Bachelor of Engineering (B.E.) – Computer Science and Engineering

Saranathan College of Engineering

Tiruchirappalli, Tamil Nadu, India

### Internship Details

* Organization: CodeTech IT Solutions Pvt. Ltd.
* Internship Domain: Machine Learning
* Intern ID: CITS4915
* Duration: 4 Weeks

### Connect With Me

* GitHub: https://github.com/vijayaragavan-dev
* LinkedIn: https://www.linkedin.com/in/vijaya-ragavan-ki10052007
* Portfolio: https://vijayaragavan.vercel.app

---

### ⭐ If you found this project useful, consider giving it a star on GitHub.

**Submitted as part of the Machine Learning Internship at CodeTech IT Solutions Pvt. Ltd. (Intern ID: CITS4915).**
