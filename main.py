import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import xgboost as xgb

# Step 1: Load Train and Test Datasets
train_data = pd.read_csv('train_data.csv')  # Replace with your train dataset path
test_data = pd.read_csv('test_data.csv')   # Replace with your test dataset path

# Step 2: Preprocess Train Data
train_data = train_data.dropna()  # Drop rows with missing values

# Normalize Latitude and Longitude for train and test data
train_data['Lat_norm'] = (train_data['Lat'] - train_data['Lat'].mean()) / train_data['Lat'].std()
train_data['Long_norm'] = (train_data['Long_'] - train_data['Long_'].mean()) / train_data['Long_'].std()

test_data['Lat_norm'] = (test_data['Lat'] - train_data['Lat'].mean()) / train_data['Lat'].std()
test_data['Long_norm'] = (test_data['Long_'] - train_data['Long_'].mean()) / train_data['Long_'].std()

# Step 3: Split Train Data into Features and Targets
X_train = train_data[['Lat', 'Long_', 'Lat_norm', 'Long_norm']]
y_train_deaths = train_data['Deaths']
y_train_cfr = train_data['Case_Fatality_Ratio']

# Step 4: Train Models
# Model for Deaths Prediction
xgb_deaths = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42)
xgb_deaths.fit(X_train, y_train_deaths)

# Model for Case Fatality Ratio Prediction
xgb_cfr = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42)
xgb_cfr.fit(X_train, y_train_cfr)

# Step 5: Evaluate Model Accuracy on Training Data
y_train_pred_deaths = xgb_deaths.predict(X_train)
accuracy_deaths = r2_score(y_train_deaths, y_train_pred_deaths) * 100
print(f"Model Accuracy for Deaths Prediction: {accuracy_deaths:.2f}%")

y_train_pred_cfr = xgb_cfr.predict(X_train)
accuracy_cfr = r2_score(y_train_cfr, y_train_pred_cfr) * 100
print(f"Model Accuracy for CFR Prediction: {accuracy_cfr:.2f}%")

# Step 6: Predict on Test Data
X_test = test_data[['Lat', 'Long_', 'Lat_norm', 'Long_norm']]

# Predict Deaths and CFR on the test data
test_data['Predicted_Deaths'] = xgb_deaths.predict(X_test)
test_data['Predicted_CFR'] = xgb_cfr.predict(X_test)

# Step 7: Calculate Predicted Cases on the test data
test_data['Predicted_Cases'] = test_data['Predicted_Deaths'] / (test_data['Predicted_CFR'] / 100)

# Step 8: Save Predictions to CSV
test_data[['Lat', 'Long_', 'Predicted_Deaths', 'Predicted_CFR', 'Predicted_Cases']].to_csv('predictions.csv', index=False)
print("\nPredictions saved to 'predictions.csv'")

# Step 9: Debugging Outputs
print("\nTest Data with Predictions:")
print(test_data[['Lat', 'Long_', 'Predicted_Deaths', 'Predicted_CFR', 'Predicted_Cases']].head())
