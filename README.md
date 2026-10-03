# Diabetes Identifier

## 📌 Overview
This project builds a Machine Learning binary classification model to predict the likelihood of diabetes onset based on diagnostic medical features. It was developed as part of a Machine Learning Internship at 1Stop.ai, focusing on handling imbalanced medical data and optimizing evaluation metrics for clinical relevance.

## 📊 Dataset
*   **Source:** [Insert Dataset Source, e.g., Kaggle Pima Indians Diabetes Database or 1Stop]
*   **Features:** The dataset includes diagnostic measurements such as Glucose level, Blood Pressure, BMI, Insulin, Age, and Diabetes Pedigree Function.
*   **Target:** Binary classification (0 = No Diabetes, 1 = Diabetes).

## ⚙️ Tech Stack
*   Python
*   Pandas & NumPy (Data Manipulation)
*   Scikit-Learn (Modeling & Evaluation)
*   Imbalanced-Learn (SMOTE for class imbalance)
*   Matplotlib & Seaborn (Data Visualization)

## 🧠 Methodology
1.  **Data Preprocessing:** Handled missing/zero values in medical features and standardized numerical data to improve model convergence.
2.  **Class Imbalance Handling:** Applied Synthetic Minority Over-sampling Technique (SMOTE) to ensure the model learns equally from both diabetic and non-diabetic cases.
3.  **Modeling:** Trained and compared multiple classifiers including Logistic Regression, Random Forest, and Gradient Boosting.
4.  **Evaluation:** Rather than relying solely on accuracy, the model was optimized using Precision, Recall, and ROC-AUC curves to minimize false negatives in a medical context.

## 📈 Results
*   Achieved an Accuracy of 85%.
*   Achieved an ROC-AUC Score of 0.89.
*   Successfully reduced false negatives through threshold tuning and ROC-AUC optimization.

## 🚀 How to Run
1. Clone this repository:
   `git clone https://github.com/dlalitha0127-maker/diabetes-identifier.git`
2. Install the required libraries:
   `pip install -r requirements.txt`
3. Run the Python script:
   `python diabetes_identifier.py`
