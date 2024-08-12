## Overview

This repository contains the project code and results for our 2024 Summer STATS202 course at Stanford University. Our project focuses on building and evaluating machine learning models to predict URL relevance based on a variety of features. We have implemented three different models and used a voting ensemble strategy to improve the precision of our predictions.

## Models Used

### Model 1: Convolutional Neural Network (CNN)
**Developed by D. Xu.**  
A deep learning model that processes and learns from input features using convolutional layers. The model was trained using stratified K-fold cross-validation and included interaction terms and statistical feature engineering.

### Model 2: Fully Connected Neural Network (NN)
**Developed by H. Wang.**  
A traditional neural network that utilizes fully connected layers. Interaction terms and local statistical features (mean and standard deviation) were applied during preprocessing.

### Model 3: XGBoost
**Developed by S. Chen.**  
An implementation of gradient boosting that incorporates regularization, parallel processing, and tree pruning. XGBoost helps to handle complex relationships between features and target variables more effectively.

## Voting Strategy

After generating predictions from each model, **S. Chen** implemented a voting strategy in the `voting_final.py` script to combine the outputs. For each test observation, the majority vote from the three models determined the final prediction. This ensemble approach led to a 1% improvement in prediction precision.

## Directory Structure

- `Model1_CNN.py`: Contains the code and results for the CNN model by D. Xu.
- `Model2_NN.py`: Contains the code and results for the Fully Connected Neural Network model by H. Wang.
- `Model3_XGBoost.py`: Contains the code and results for the XGBoost model by S. Chen.
- `voting_final.py`: The final script to run the voting ensemble strategy and generate the combined predictions.
- `Report/`: Contains the combined results and analysis, as well as the final prediction files.
- `README.md`: This file, providing an overview and instructions.

## How to Run

To reproduce the results, clone this repository and follow these steps:

1. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the final voting ensemble strategy script:
   ```bash
   python voting_final.py
   ```
3. Review the final prediction output in the `Report/` directory.

## Results

The final results are saved in the `Report/` directory, including the CSV files with the predicted relevance scores for the test dataset. The ensemble model demonstrated superior precision compared to individual models.

## Authors

- **D. Xu**: Project lead and developer of the Convolutional Neural Network (CNN) model.
- **S. Chen**: Developed the XGBoost model and implemented the voting strategy.
- **H. Wang**: Conducted data analysis and developed the Fully Connected Neural Network (NN) model.
