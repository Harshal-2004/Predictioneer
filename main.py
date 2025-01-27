# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import xgboost as xgb

# Step 1: Load the Train and Test Datasets
# -----------------------------------------
# Read train and test datasets
train_data = pd.read_csv('train_data.csv')  # Replace with your train dataset path
test_data = pd.read_csv('test_data.csv')   # Replace with your test dataset path

# Step 2: Data Preprocessing
# ---------------------------
# a) Remove missing values from the training and testing  dataset
train_data = train_data.dropna()
test_data = test_data.dropna()

# b) Normalize the Latitude and Longitude in both train and test datasets
train_data['Lat_norm'] = (train_data['Lat'] - train_data['Lat'].mean()) / train_data['Lat'].std()
train_data['Long_norm'] = (train_data['Long_'] - train_data['Long_'].mean()) / train_data['Long_'].std()

test_data['Lat_norm'] = (test_data['Lat'] - train_data['Lat'].mean()) / train_data['Lat'].std()
test_data['Long_norm'] = (test_data['Long_'] - train_data['Long_'].mean()) / train_data['Long_'].std()

# Step 3: Split Features and Targets
# -----------------------------------
# Define features (X) and target variables (y) for training
X_train = train_data[['Lat', 'Long_', 'Lat_norm', 'Long_norm']]
y_train_deaths = train_data['Deaths']
y_train_cfr = train_data['Case_Fatality_Ratio']

# Step 4: Train Machine Learning Models
# --------------------------------------
# a) Model to predict Deaths
xgb_deaths = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42)
xgb_deaths.fit(X_train, y_train_deaths)

# b) Model to predict Case Fatality Ratio (CFR)
xgb_cfr = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42)
xgb_cfr.fit(X_train, y_train_cfr)

# Step 5: Evaluate Model Performance
# -----------------------------------
# a) Evaluate the Deaths model
y_train_pred_deaths = xgb_deaths.predict(X_train)
accuracy_deaths = r2_score(y_train_deaths, y_train_pred_deaths) * 100
print(f"Model Accuracy for Deaths Prediction: {accuracy_deaths:.2f}%")

# b) Evaluate the CFR model
y_train_pred_cfr = xgb_cfr.predict(X_train)
accuracy_cfr = r2_score(y_train_cfr, y_train_pred_cfr) * 100
print(f"Model Accuracy for CFR Prediction: {accuracy_cfr:.2f}%")

# Step 6: Make Predictions on Test Data
# --------------------------------------
# Prepare test features
X_test = test_data[['Lat', 'Long_', 'Lat_norm', 'Long_norm']]

# Predict Deaths and CFR using the test data
test_data['Predicted_Deaths'] = xgb_deaths.predict(X_test)
test_data['Predicted_CFR'] = xgb_cfr.predict(X_test)

# Step 7: Calculate Predicted Cases
# ----------------------------------
# Using the formula: Predicted Cases = Predicted Deaths / (Predicted CFR / 100)
test_data['Predicted_Cases'] = test_data['Predicted_Deaths'] / (test_data['Predicted_CFR'] / 100)

# Step 8: Save the Predictions to a CSV File
# -------------------------------------------
# Save relevant columns to 'predictions.csv'
test_data[['Lat', 'Long_', 'Predicted_Deaths', 'Predicted_CFR', 'Predicted_Cases']].to_csv('predictions.csv', index=False)
print("\nPredictions saved to 'predictions.csv'")

# Step 9: Display Debugging Outputs (Optional)
# ---------------------------------------------
print("\nTest Data with Predictions:")
print(test_data[['Lat', 'Long_', 'Predicted_Deaths', 'Predicted_CFR', 'Predicted_Cases']].head())
