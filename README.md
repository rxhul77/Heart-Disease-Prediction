# ❤️ Heart Disease Prediction Using Machine Learning

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">

  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit-learn">

  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-purple?style=for-the-badge&logo=pandas" alt="Pandas">

  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" alt="Project Status">

</p>

<p align="center">
  <b>A machine learning classification project for predicting the presence of heart disease from clinical and demographic features.</b>
</p>

---

## 📌 Project Overview

Heart disease prediction is a binary classification problem where machine learning models can be used to identify patterns associated with the presence or absence of heart disease.

This project uses the **Heart Disease UCI dataset** and implements an end-to-end machine learning workflow:

**Data Loading → Data Cleaning → Missing Value Handling → Feature Scaling → Categorical Encoding → Model Training → Evaluation → Visualization**

Two machine learning algorithms were implemented:

- **Logistic Regression**
- **Decision Tree**

The models were evaluated using:

- Accuracy
- Precision
- Recall
- ROC-AUC
- Confusion Matrix
- ROC Curve

---

## 🎯 Project Objectives

The main objectives of this project are:

- Perform data cleaning and preprocessing.
- Handle missing values appropriately.
- Convert the original target into a binary classification problem.
- Scale numerical features.
- Encode categorical features.
- Train Logistic Regression and Decision Tree models.
- Evaluate classification performance using multiple metrics.
- Generate confusion matrix visualizations.
- Generate ROC curves.
- Save model predictions and evaluation results as CSV files.

---

## 📊 Dataset

The project uses the **Heart Disease UCI dataset**, containing **920 patient records and 16 original columns**.

### Dataset Summary

| Property | Details |
|---|---|
| Records | 920 |
| Original Columns | 16 |
| Target Column | `num` |
| Final Target | `target` |
| Problem Type | Binary Classification |
| Training Samples | 736 |
| Testing Samples | 184 |

### Features Used

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex |
| `dataset` | Dataset/source group |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting electrocardiographic results |
| `thalch` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression |
| `slope` | Slope of the ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia-related feature |

The `id` column was removed because it does not represent a useful predictive feature.

---

## 🔄 Target Transformation

The original dataset contains the `num` target with values from **0 to 4**.

For binary classification, it was transformed as follows:

| Original `num` | New `target` | Meaning |
|---:|---:|---|
| 0 | 0 | No Heart Disease |
| 1–4 | 1 | Heart Disease |

### Target Distribution

| Class | Samples |
|---|---:|
| No Heart Disease | 411 |
| Heart Disease | 509 |

---

## 🧹 Data Preprocessing

### 1. Missing Value Handling

Missing numerical values were replaced using the **median**.

Missing categorical values were replaced using the **most frequent value**.

This approach allows the models to work with incomplete records without simply deleting large portions of the dataset.

### 2. Numerical Feature Scaling

Numerical features were standardized using:

```text
StandardScaler