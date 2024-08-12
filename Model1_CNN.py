import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
import os

# Enable multi-threading to fully utilize CPU
os.environ["OMP_NUM_THREADS"] = "20"  # Adjust to the number of threads your CPU can handle
os.environ["TF_NUM_INTRAOP_THREADS"] = "20"
os.environ["TF_NUM_INTEROP_THREADS"] = "20"
tf.config.threading.set_intra_op_parallelism_threads(20)
tf.config.threading.set_inter_op_parallelism_threads(20)

# Load training and test data
data = pd.read_csv('D:/Users/Alastor/Desktop/STAT202/Project/training.csv')
test_data = pd.read_csv('D:/Users/Alastor/Desktop/STAT202/Project/test.csv')

# Select features and labels for training
features = [
    'query_length', 'is_homepage', 'sig1', 'sig2', 'sig3',
    'sig4', 'sig5', 'sig6', 'sig7', 'sig8'
]
X = data[features]
y = data['relevance']

# Create interaction terms (only for original features)
def create_interaction_terms(X):
    """
    Generate interaction terms between all pairs of features.
    """
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            X[f'{features[i]}_x_{features[j]}'] = X[features[i]] * X[features[j]]
    return X

# Create local statistical features
def create_local_stat_features(X, window_size=3):
    """
    Create local statistical features (mean and standard deviation)
    using a rolling window.
    """
    for feature in X.columns:
        X[f'{feature}_mean'] = X[feature].rolling(window=window_size, min_periods=1).mean()
        X[f'{feature}_std'] = X[feature].rolling(window=window_size, min_periods=1).std().fillna(0)
    return X

# Create multi-scale features
def create_multi_scale_features(X):
    """
    Create multi-scale features by applying square, square root, 
    and logarithmic transformations to the original features.
    """
    for feature in X.columns:
        X[f'{feature}_squared'] = X[feature] ** 2
        X[f'{feature}_sqrt'] = np.sqrt(X[feature])
        X[f'{feature}_log'] = np.log1p(X[feature])
    return X

# Feature engineering for is_homepage = 0
def feature_engineering_homepage0(X):
    """
    Apply feature engineering to the data where is_homepage = 0.
    Includes creating local statistical features, multi-scale features,
    and interaction terms.
    """
    X = create_local_stat_features(X)
    X = create_multi_scale_features(X)
    interaction_terms = create_interaction_terms(X[features])
    X = pd.concat([X, interaction_terms], axis=1)
    return X

# Feature engineering for is_homepage = 1
def feature_engineering_homepage1(X):
    """
    Apply feature engineering to the data where is_homepage = 1.
    Includes creating local statistical features and interaction terms.
    """
    X = create_local_stat_features(X)
    interaction_terms = create_interaction_terms(X[features])
    X = pd.concat([X, interaction_terms], axis=1)
    return X

# Efficient data input pipeline
def create_dataset(X, y, batch_size=64):
    """
    Create a TensorFlow dataset for efficient input pipeline, 
    including shuffling, batching, and prefetching.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# Train CNN model using stratified sampling
def process_data_with_cnn(X, y, epochs, feature_engineering_fn):
    """
    Process the data and train a CNN model using stratified K-fold cross-validation.
    """
    # Apply feature engineering
    X = feature_engineering_fn(X)
    
    # Standardize features
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    
    # Reshape data for CNN input (samples, timesteps, features)
    X_cnn = X_standardized.reshape((X_standardized.shape[0], X_standardized.shape[1], 1))
    
    # Stratified K-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_index, val_index in kf.split(X_cnn, y):
        X_train, X_val = X_cnn[train_index], X_cnn[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Build CNN model
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
            BatchNormalization(),
            Dropout(0.5),
            Conv1D(128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Create datasets for training and validation
        train_dataset = create_dataset(X_train, y_train)
        val_dataset = create_dataset(X_val, y_val)
        
        # Train the model
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=2)
        
        # Evaluate model performance
        y_pred = (model.predict(X_val) > 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_pred)
        cv_scores.append(accuracy)
        print(f"Fold Accuracy: {accuracy}")
    
    mean_accuracy = np.mean(cv_scores)
    print(f"Mean Cross-Validation Accuracy: {mean_accuracy}")
    
    # Train final model on the entire training set
    final_model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_cnn.shape[1], 1)),
        BatchNormalization(),
        Dropout(0.5),
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    final_optimizer = Adam(learning_rate=0.0001)
    final_model.compile(loss='binary_crossentropy', optimizer=final_optimizer, metrics=['accuracy'])
    
    final_model.fit(create_dataset(X_cnn, y), epochs=epochs, verbose=2)
    
    # Save the final model
    final_model.save('D:/Users/Alastor/Desktop/STAT202/Project/cnn_model.h5')
    
    return final_model, scaler

# Train the model with data where is_homepage = 0
data_homepage_0 = data[data['is_homepage'] == 0]
model_homepage_0, scaler_homepage_0 = process_data_with_cnn(
    data_homepage_0[features], data_homepage_0['relevance'], epochs=50, 
    feature_engineering_fn=feature_engineering_homepage0
)

# Train the model with data where is_homepage = 1
data_homepage_1 = data[data['is_homepage'] == 1]
model_homepage_1, scaler_homepage_1 = process_data_with_cnn(
    data_homepage_1[features], data_homepage_1['relevance'], epochs=50, 
    feature_engineering_fn=feature_engineering_homepage1
)

# --- Prediction Section ---

# Apply feature engineering to test data
test_X_homepage_0 = feature_engineering_homepage0(
    test_data[features][test_data['is_homepage'] == 0]
)
test_X_homepage_1 = feature_engineering_homepage1(
    test_data[features][test_data['is_homepage'] == 1]
)

# Standardize the test data
test_X_standardized_0 = scaler_homepage_0.transform(test_X_homepage_0).reshape(
    (test_X_homepage_0.shape[0], test_X_homepage_0.shape[1], 1)
)
test_X_standardized_1 = scaler_homepage_1.transform(test_X_homepage_1).reshape(
    (test_X_homepage_1.shape[0], test_X_homepage_1.shape[1], 1)
)

# Generate predictions using the models
test_predictions_0 = (model_homepage_0.predict(test_X_standardized_0) > 0.5).astype(int)
test_predictions_1 = (model_homepage_1.predict(test_X_standardized_1) > 0.5).astype(int)

# Combine predictions
test_predictions = np.zeros(len(test_data), dtype=int)
test_data_homepage_0_indices = test_X_homepage_0.index
test_data_homepage_1_indices = test_X_homepage_1.index

for index, row in test_data.iterrows():
    if row['is_homepage'] == 0:
        test_predictions[index] = test_predictions_0[
            np.where(test_data_homepage_0_indices == index)[0][0]
        ]
    else:
        test_predictions[index] = test_predictions_1[
            np.where(test_data_homepage_1_indices == index)[0][0]
        ]

# Prepare submission file
sample_submission_path = 'D:/Users/Alastor/Desktop/STAT202/Project/sample_submission.csv'
sample_submission = pd.read_csv(sample_submission_path)
sample_submission['relevance'] = test_predictions

# Save the final predictions to a CSV file
output_file_path = 'D:/Users/Alastor/Desktop/STAT202/Project/Report/model1_CNN_results.csv'
sample_submission.to_csv(output_file_path, index=False)

print(f"Predictions have been saved to {output_file_path}")
