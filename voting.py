import pandas as pd
import sys
sys.path.append('D:/Users/Alastor/Desktop/STAT202/Project/Report/')
# Importing the necessary models (assuming these files contain functions or classes to be used)
import Model1_CNN
import Model2_NN
import Model3_XGBoost

# Load prediction results from three different models
df_prediction1 = pd.read_csv('D:/Users/Alastor/Desktop/STAT202/Project/Report/model1_CNN_results.csv')
df_prediction2 = pd.read_csv('D:/Users/Alastor/Desktop/STAT202/Project/Report/model2_nn_results.csv')  # Corrected file path
df_prediction3 = pd.read_csv('D:/Users/Alastor/Desktop/STAT202/Project/Report/model3_xgboost_results.csv')

# Create a copy of the first predictions dataframe to store final results
result = df_prediction1.copy()

# Iterate through each ID in the result dataframe
for i in result.id:
    # Find the corresponding records in all three prediction dataframes based on ID
    record1 = df_prediction1.iloc[df_prediction1[df_prediction1.id == i].index[0]]
    record2 = df_prediction2.iloc[df_prediction2[df_prediction2.id == i].index[0]]
    record3 = df_prediction3.iloc[df_prediction3[df_prediction3.id == i].index[0]]

    # Count how many times the relevance is predicted as 0
    count = 0
    if record1['relevance'] == 0:
        count += 1
    if record2['relevance'] == 0:
        count += 1
    if record3['relevance'] == 0:
        count += 1

    # If two or more models predict relevance as 0, set final prediction to 0, otherwise set to 1
    if count >= 2:
        result.loc[df_prediction1[df_prediction1.id == i].index[0], 'relevance'] = 0
    else:
        result.loc[df_prediction1[df_prediction1.id == i].index[0], 'relevance'] = 1

# Save the final predictions to a CSV file
result.to_csv('D:/Users/Alastor/Desktop/STAT202/Project/Report/predictions_selected.csv', index=False)

