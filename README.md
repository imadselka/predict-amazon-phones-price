﻿# Amazon Phone Price Prediction

## Overview

This project predicts the prices of Amazon phones based on several features such as discounts, ratings, brand, operating system, RAM, CPU, storage, screen size, and more. The prediction model uses Linear Regression, along with data preprocessing techniques such as scaling numerical features and one-hot encoding categorical features. The goal is to provide accurate price predictions using the phone’s specifications.

## Dataset

The dataset contains information about cell phones available on Amazon, including:

- **Price (Dollar)**: The actual price of the phone (target variable).
- **Discount Percentage**: The percentage discount on the phone.
- **Price Before Discount**: The original price before any discounts.
- **Ratings**: The user rating out of 5 stars.
- **Number of Ratings**: The number of user ratings.
- **Brand**: The phone's brand.
- **Operating System**: The phone's operating system (e.g., Android, iOS).
- **RAM (GB)**: RAM size in gigabytes.
- **CPU**: The phone’s CPU model.
- **Storage (GB)**: Storage size in gigabytes.
- **Screen Size (Inches)**: Size of the phone screen in inches.
- **Cellular Technology**: Type of cellular network technology (e.g., 4G, 5G).
- **Available Colors**: The colors in which the phone is available.

## Approach

1. **Data Preprocessing**:

   - **Handling Missing Data**: Rows with missing values are dropped.
   - **Categorical and Numerical Features**:
     - Categorical features such as `brand`, `operating_system`, and `CPU` are one-hot encoded.
     - Numerical features such as `discount_percentage` and `price_before_discount` are standardized using `StandardScaler`.

2. **Pipeline Setup**:

   - A pipeline is used to streamline the preprocessing and modeling steps. The pipeline includes:
     - **ColumnTransformer**: Applies the appropriate transformations (scaling or encoding) to numerical and categorical data.
     - **Linear Regression Model**: The model used to predict prices.

3. **Hyperparameter Tuning**:

   - **GridSearchCV**: Used to perform grid search over different hyperparameters for Linear Regression, optimizing for Mean Squared Error (MSE).

4. **Model Training and Evaluation**:

   - The dataset is split into training and testing sets (80% training, 20% testing).
   - The model is trained on the training data, and predictions are made on the test set.
   - Model evaluation is performed using:
     - **Mean Absolute Error (MAE)**
     - **Root Mean Squared Error (RMSE)**
     - **R-squared (R²) Score**

5. **Visualization**:
   - A scatter plot is generated to compare actual vs predicted prices, giving insights into the model’s performance.

## Key Python Libraries

- **pandas**: For data manipulation and analysis.
- **scikit-learn**: For building and evaluating the machine learning model.
  - `LinearRegression`: Linear regression model for price prediction.
  - `StandardScaler`: To scale numerical features.
  - `OneHotEncoder`: To encode categorical variables.
  - `Pipeline`: To streamline preprocessing and model training.
  - `GridSearchCV`: For hyperparameter tuning.
- **matplotlib**: For visualizing the actual vs predicted prices.
- **numpy**: For numerical operations.

_The script will train the model, evaluate its performance, and generate a scatter plot comparing actual vs predicted phone prices._

# Conclusion

 *This project showcases how to approach a machine learning task, from data preprocessing and model building to hyperparameter tuning and evaluation. The linear regression model provides an initial baseline for predicting phone prices, but the model can be further refined by experimenting with other algorithms or more advanced feature engineering*
