import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset (Replace with your actual file path)
df = pd.read_csv('Amazon_Cell_Phones.csv')

# Selecting relevant features for price prediction
df = df[['Price (Dollar)', 'discount_percentage', 'price_before_discount', 'rating_out_of_5',
         'number_of_ratings', 'brand', 'operating_system', 'RAM (GB)', 'CPU', 
         'Storage (GB)', 'screen_size (Inches)', 'cellular_technology', 
         'cpu_model', 'available_colors']]

# Drop rows with missing values
df = df.dropna()

# Separate features (X) and target (y)
X = df.drop('Price (Dollar)', axis=1)  # Features
y = df['Price (Dollar)']               # Target

# Define the categorical and numerical features
categorical_features = ['brand', 'operating_system', 'CPU', 'cellular_technology', 'cpu_model', 'available_colors']
numerical_features = ['discount_percentage', 'price_before_discount', 'rating_out_of_5',
                       'number_of_ratings', 'RAM (GB)', 'Storage (GB)', 'screen_size (Inches)']

# Preprocessing for numerical data: standard scaling
numerical_transformer = StandardScaler()

# Preprocessing for categorical data: one-hot encoding
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine preprocessing steps for both numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline that first preprocesses the data and then applies linear regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Define hyperparameters to tune
param_grid = {
    'model__n_jobs': [-1, 1, 2, 3, 4],
    'preprocessor__num__with_mean': [True, False],
    'preprocessor__num__with_std': [True, False]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Get the best model's parameters and estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Print the best model's parameters and estimator
print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the best model
best_estimator.fit(X_train, y_train)

# Make predictions on the test set
predictions = best_estimator.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Visualizing actual vs predicted prices with a scatter plot
plt.scatter(y_test, predictions, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)  # Perfect prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (Linear Regression)')
plt.show()