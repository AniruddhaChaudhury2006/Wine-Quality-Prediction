# Wine Quality Prediction

This notebook demonstrates a machine learning workflow to predict the quality of wine based on its physiochemical properties. The dataset used is the 'Wine Quality' dataset, commonly used for classification and regression tasks in machine learning.

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset Overview](#dataset-overview)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Prediction System](#prediction-system)

## 1. Problem Statement
The goal is to classify red wine into 'good quality' (quality score >= 7) or 'bad quality' (quality score < 7) based on various physiochemical input features. This is a binary classification problem.

## 2. Dataset Overview
The dataset `winequality-red.csv` contains various attributes of red wine, such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, and quality.

- **Loading the data:** The dataset is loaded using pandas from `/content/winequality-red.csv`.
- **Initial inspection:** `wine_dataset.shape`, `wine_dataset.head()`, and `wine_dataset.isnull().sum()` are used to inspect the dimensions, first few rows, and check for missing values, respectively.
- **Statistical summary:** `wine_dataset.describe()` provides a statistical overview of the dataset.

## 3. Exploratory Data Analysis (EDA)
EDA helps understand the data distribution and relationships between features.

- **Quality distribution:** `sns.catplot(x='quality', data=wine_dataset, kind='count')` visualizes the distribution of wine quality ratings.
- **Feature vs. Quality:** Bar plots are used to visualize the relationship between individual features (e.g., 'volatile acidity', 'citric acid') and wine quality.
- **Correlation Heatmap:** A heatmap `sns.heatmap(correlation, ...)` is generated to visualize the correlation matrix between all features, helping identify highly correlated features.

## 4. Data Preprocessing

- **Feature and Target Separation:**
  - Features (X) are created by dropping the 'quality' column: `X = wine_dataset.drop("quality",axis=1)`.
  - The target variable (Y) is binarized: `Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)`, where 1 represents 'good quality' and 0 represents 'bad quality'.
- **Train-Test Split:** The dataset is split into training and testing sets using `train_test_split` with a `test_size` of 0.2 and `random_state` of 2.

## 5. Model Training

- **Model Selection:** A `RandomForestClassifier` is chosen for this classification task.
- **Training:** The model is trained on the preprocessed training data: `model.fit(X_train, Y_train)`.

## 6. Model Evaluation

- **Prediction:** The trained model predicts the quality on the test set: `X_test_prediction = model.predict(X_test)`.
- **Accuracy:** The accuracy of the model on the test data is calculated using `accuracy_score`.

## 7. Prediction System

- **Input Data:** A sample input `input_data` (a tuple of physiochemical properties) is created.
- **Preprocessing Input:** The input data is converted to a NumPy array, reshaped to match the model's expected input format.
- **Prediction:** The model makes a prediction on the single input data point.
- **Output:** The prediction (Good quality wine or Bad quality wine) is printed.
