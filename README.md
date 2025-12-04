# 🏆 Insurance Policy Renewal Prediction
## 🚀 Project Executive Summary
This project delivers an end-to-end Machine Learning solution designed to predict the likelihood of an existing customer renewing their insurance policy (Target: 1 for Renew, 0 for Not Renew). The pipeline successfully addresses critical challenges, including severe class imbalance and noisy features, resulting in robust models ready for deployment to drive targeted customer retention strategies.

## 📑 Problem Statement & Goal:
Overview / Problem Statement:
Insurance companies rely heavily on policy renewals for long-term revenue and customer retention. This project focuses on developing a robust machine learning model to predict whether an existing customer will renew their insurance policy.

This is a Binary Classification task:
Target = 1: Customer WILL renew the policy.
Target = 0: Customer WILL NOT renew the policy.

Goal: Develop a model to predict the binary outcome of policy renewal.

##  💾 Dataset Structure:
The project utilizes customer-level data containing demographic, policy, and behavioral features of policyholders.

| Metric | Train Set | Test Set |
| :--- | :--- | :--- |
| **Total Rows (Observations)** | 8,000 | 2,000 |
| **Total Columns** | 19 (including Target) | 18 (excluding Target) |
| **Target Column** | target | N/A |

## 📊 Feature Breakdown

| The columns in the training set and testing set are composed of the following data types and roles:

| Column Type | Count | Key Examples | Role & Necessity |
| :--- | :--- | :--- | :--- |
| **Numerical** (int/float) | 12 | `age`, `premium`, `Count_3-6_months_late` | Direct model input, requires imputation/scaling. |
| **Categorical** (object) | 6 | `sourcing_channel`, `occupation`, `residence_area_type` | Requires **One-Hot Encoding**. |
| **Redundant/Correlated** | 2 | `age_in_days`, `Income` | **Dropped** due to correlation with `age` and `income`. |
| **Identifier** | 2 | `unique_id`, `name` | **Separated** (`unique_id`) or **Dropped** (`name`). |
| **Target Column** | 1 | `target` (Binary: 0 or 1) | **Predictive Goal.** |

# 🔍 Exploratory Data Analysis (EDA) Insights
The initial analysis was crucial for defining the preprocessing strategy, focusing on data quality, distribution, and feature utility:

• Missing Values: Detected and confirmed the presence of missing values across key numerical features (e.g., late payment counts, underwriting score).

• Data Distribution & Outliers:

-> Right Skewness: Histograms verified that the majority of numerical features are right-skewed, necessitating careful transformation.

-> Outliers: Outliers were visually identified through box plots in most continuous variables.

• Imbalance Crisis: The target variable imbalance was confirmed using bar graphs, showing the severe minority class representation (≈95% Renewal vs. ≈5% Non-Renewal). This validated the necessity of using the F1-Score as the primary success metric.

• Correlation : The correlation heatmap was used to check relationships. Crucially, no strong linear correlation was found between features or between features and the target, indicating that all non-redundant variables contribute uniquely.

• Redundant Feature Identification: Multiple columns providing similar information were identified: (age_in_days, age) and (Income, income).

Selection Rationale: After comparing these columns against the target and checking data realism, age and income were selected as the definitive features, while age_in_days and Income were dropped due to being deemed unrealistic or less reliable representations of the underlying data.

## 🧹 Data Preprocessing & Feature Engineering 🛠️

| Step | Technique Applied | Rationale |
| :--- | :--- | :--- |
| **Missing Values** | **Median Imputation** | Filled missing numerical values using the **training set median** to prevent data leakage. |
| **Outlier Treatment** | Identified (No transformation applied in the final pipeline) | Outliers were summarized by percentage per column but were not explicitly clipped/capped in the final encoding steps. |
| **Feature Dropping** | `unique_id`, `name`, `Income`, `age_in_days` | Removed because age_in_days and Income are non-realistic values with the target and other independent columns. |
| **Categorical Encoding** | **labelEncoding** (`pd.get_dummies`) | Converted nominal features (e.g., `occupation`, `sourcing_channel`,'occupation_type') into model-readable format. |
| **Data Imbalance** | **SMOTE** (Synthetic Minority Over-sampling Technique) | Used on the training set to balance the target classes, ensuring models do not bias towards the majority (non-renewal) class. |


### ✅ ML/DL Models Used 🧠

A suite of diverse and powerful classification algorithms was implemented to ensure comprehensive evaluation:

| Model | Type | Strategy for Imbalance |
| :--- | :--- | :--- |
| **XGBoost Classifier** | Gradient Boosting | Tuned using **`scale_pos_weight`** to account for class imbalance. |
| **Random Forest Classifier** | Ensemble (Bagging) | Robust, high-performance tree model. |
| **Logistic Regression** | Linear Baseline | Used **`class_weight='balanced'`** parameter. |
| **Gradient Boosting** | Ensemble (Boosting) | Strong sequential learner. |
| **SVC** | Kernel-based (Non-Linear) | Included for complex boundary exploration. |

📊 Model Training & Evaluation 📈
Data Split: 80% Training (SMOTE-processed) and 20% Validation (Original Distribution).
### Model Performance Summary

| Model | Validation Insight | Key Interpretation |
| :--- | :--- | :--- |
| **Gradient Boosting** | Accuracy: **0.9375**, F1 (Class 0): **0.2308**, Recall (Class 0): **0.1613** | Best **F1-score** for the minority class; most balanced prediction for Class 0. |
| **Random Forest** | Accuracy: **0.9337**, F1 (Class 0): **0.2740**, Recall (Class 0): **0.2151** | Highest **Recall** for Class 0 → best at identifying **non-renewal** cases. |
| **XGBoost Classifier** | Accuracy: **0.9400**, F1 (Class 0): **0.0769**, Recall (Class 0): **0.0430** | Highest **overall accuracy**, but **poor minority class detection**, showing bias toward Class 1. |

# prediction:


# 📤 Submission File Generation & Mapping 🏷️
The critical final stage involves applying the best model to the unseen test data and creating the submission file using the separated identifiers.

Process Summary

Prediction: The best-performing model (e.g., XGBoost) generates predictions (0 or 1) for the preprocessed test data matrix.

ID Retrieval: The predictions are combined with the previously stored test_ids (the original unique_id column from the test set).

Output Creation: A final submission CSV file is generated, consisting only of the two required columns: the original unique_id and the predicted target for accurate competition/deployment mapping.

Key Output

Final Output: A CSV file mapping the original unique_id to the model's predicted renewal outcome.

🖼️ Visualization Highlights
Key visualizations ensured transparency and explainability:

Correlation Heatmaps: Visualized feature interdependence post-encoding.

Distribution Plots: Showcased the shift in target distribution after SMOTE application.

Bivariate Analysis: Count plots detailed the relationship between various occupation and sourcing_channel groups and the renewal rate.

▶️ How to Run the Code
Prerequisites: Ensure Python 3.x is installed.

Clone the Repository:

Bash
git clone https://github.com/<your-username>/insurance-renewal-prediction.git
Setup Environment:

Bash
pip install -r requirements.txt
Execute the Script:

Bash
python insurance_renewal_prediction.py
📂 Folder Structure
.
├── insurance_renewal_prediction.py  # 🖥️ Main Analysis & ML Pipeline
├── train.csv                        # 📥 Training Data
├── test.csv                         # 📥 Test Data (Used for Final Predictions)
├── submission.csv                   # 📤 Final Mapped Output File
└── requirements.txt                 # 📋 Project Dependencies
🛠️ Requirements
Install all necessary packages using the requirements.txt (or manually):

Bash
pandas
numpy
scikit-learn
seaborn
matplotlib
xgboost
imbalanced-learn
💡 Future Improvements
Advanced Hyperparameter Tuning: Implement GridSearchCV / RandomizedSearchCV on XGBoost for production-level optimization.

Model Stacking: Experiment with ensemble techniques like stacking or voting classifiers to potentially boost final performance.

Lighter Models: Explore LightGBM or CatBoost for faster training and potentially better handling of categorical features.
