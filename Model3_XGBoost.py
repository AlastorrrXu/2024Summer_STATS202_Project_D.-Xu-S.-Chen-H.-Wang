import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

# Load the training and test datasets from CSV files
search_mid = pd.read_csv("D:/Users/Alastor/Desktop/STAT202/Project/training.csv")
test_mid = pd.read_csv("D:/Users/Alastor/Desktop/STAT202/Project/test.csv")

# Define the features to be used for training, excluding non-feature columns
allvars = search_mid.columns.drop(['query_id', 'url_id', 'relevance', 'id', 'is_homepage'])

# Separate the data into is_homepage == 0 and is_homepage == 1 groups
test_0 = test_mid[test_mid['is_homepage'] == 0]
search_0 = search_mid[search_mid['is_homepage'] == 0]

# Define the features and labels for the is_homepage == 0 group
X_0 = search_0[allvars]
y_0 = search_0['relevance']

# Split the data into training and validation sets
X_train_0, X_valid_0, y_train_0, y_valid_0 = train_test_split(X_0, y_0, test_size=0.2, random_state=42)

# Initialize the XGBoost model for is_homepage == 0 group
xgb_model_0 = xgb.XGBClassifier(random_state=42)

# Define the parameter grid for GridSearchCV
param_grid_0 = {
    'n_estimators': [60, 70, 80],
    'learning_rate': [0.1, 0.15, 0.2],
    'max_depth': [4, 5],
    'subsample': [0.8],
    'colsample_bytree': [1.0]
}

# Perform grid search with 5-fold cross-validation using f1_macro as the scoring metric
grid_search_0 = GridSearchCV(
    estimator=xgb_model_0, param_grid=param_grid_0, 
    scoring='f1_macro', cv=5, verbose=1
)
grid_search_0.fit(X_train_0, y_train_0)

# Print the best parameters found for is_homepage == 0
print(f'Best parameters for is_homepage == 0: {grid_search_0.best_params_}')

# Train the best model on the full dataset for is_homepage == 0
best_model_0 = grid_search_0.best_estimator_
best_model_0.fit(X_0, y_0)

# Predict the test data for is_homepage == 0
X_test_0 = test_0[allvars]
test_0['relevance'] = best_model_0.predict(X_test_0)

# Create the output dataframe for is_homepage == 0 predictions
output_0 = test_0[['id', 'relevance']]

# Repeat the process for is_homepage == 1 group
test_1 = test_mid[test_mid['is_homepage'] == 1]
search_1 = search_mid[search_mid['is_homepage'] == 1]

X_1 = search_1[allvars]
y_1 = search_1['relevance']

X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(X_1, y_1, test_size=0.1, random_state=42)

xgb_model_1 = xgb.XGBClassifier(random_state=42)

param_grid_1 = {
    'n_estimators': [60, 70, 80],
    'learning_rate': [0.25],
    'max_depth': [3],
    'subsample': [1.0],
    'colsample_bytree': [1.0]
}

grid_search_1 = GridSearchCV(
    estimator=xgb_model_1, param_grid=param_grid_1, 
    scoring='accuracy', cv=3, verbose=1
)
grid_search_1.fit(X_train_1, y_train_1)

# Print the best parameters found for is_homepage == 1
print(f'Best parameters for is_homepage == 1: {grid_search_1.best_params_}')

# Train the best model on the full dataset for is_homepage == 1
best_model_1 = grid_search_1.best_estimator_
best_model_1.fit(X_1, y_1)

# Predict the test data for is_homepage == 1
X_test_1 = test_1[allvars]
test_1['relevance'] = best_model_1.predict(X_test_1)

# Create the output dataframe for is_homepage == 1 predictions
output_1 = test_1[['id', 'relevance']]

# Combine the predictions from both is_homepage groups
final_output = pd.concat([output_0, output_1])

# Save the final predictions to a CSV file
final_output.to_csv('D:/Users/Alastor/Desktop/STAT202/Project/Report/model3_xgboost_results.csv', index=False)
print("Final predictions saved to D:/Users/Alastor/Desktop/STAT202/Project/Report/model3_xgboost_results.csv")
