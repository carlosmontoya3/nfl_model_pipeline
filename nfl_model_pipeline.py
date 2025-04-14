# nfl_model_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import os

# Load the dataset
file_path = "/Users/carlosmontoya3/Documents/Capstone/nfl_model_pipeline/nfl_offensive_stats.csv"
df = pd.read_csv(file_path)

# --- Basic Cleaning (adjust as needed) ---
df = df.dropna()  # Drop missing values for simplicity

# --- Feature and Target Selection (example features) ---
features = ['pass_att', 'pass_cmp', 'pass_int'] 
X = df[features]
y = df['pass_yds']  # Update target if needed

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Model Training ---
linear_model = LinearRegression().fit(X_train, y_train)
gb_model = GradientBoostingRegressor().fit(X_train, y_train)

# --- Predictions ---
linear_preds = linear_model.predict(X_test)
gb_preds = gb_model.predict(X_test)

# --- Evaluation ---
linear_r2 = r2_score(y_test, linear_preds)
gb_r2 = r2_score(y_test, gb_preds)
rmse = mean_squared_error(y_test, gb_preds, squared=False)

print(f"Linear Regression R^2: {linear_r2:.2f}")
print(f"Gradient Boosting R^2: {gb_r2:.2f}")
print(f"Gradient Boosting RMSE: {rmse:.2f}")

# --- Visualization ---
models = ['Linear Regression', 'Gradient Boosting']
scores = [linear_r2, gb_r2]

plt.bar(models, scores)
plt.title('Model Comparison (R² Score)')
plt.ylabel('R² Score')
plt.savefig('model_comparison.png')
plt.show()

# --- Optional: Save results ---
with open('model_results.txt', 'w') as f:
    f.write(f"Linear Regression R^2: {linear_r2:.2f}\n")
    f.write(f"Gradient Boosting R^2: {gb_r2:.2f}\n")
    f.write(f"Gradient Boosting RMSE: {rmse:.2f}\n")
