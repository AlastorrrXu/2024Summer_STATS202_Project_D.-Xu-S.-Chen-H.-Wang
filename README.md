## Overview

In this project, we implemented a voting ensemble strategy to improve the precision of predicting URL relevance. The ensemble method combines the predictions from three different models: a Convolutional Neural Network (CNN), a Fully Connected Neural Network (NN), and an XGBoost model. By using the majority voting technique, we achieved a 1% improvement in prediction precision compared to individual models.

## Voting Strategy

The final prediction for each observation is determined by a majority vote from the three models:

```
Prediction = mode(CNN_prediction, NN_prediction, XGBoost_prediction)
```

### Model 1: Convolutional Neural Network (CNN)

- **Developed by D. Xu.**
- The CNN processes input features through multiple convolutional layers followed by fully connected layers. The model is trained using stratified K-fold cross-validation and includes interaction terms and statistical feature engineering.

**Model Architecture:**

The CNN uses the following layers:
1. **Convolutional Layers**
   - Convolution: `Conv2D(W * X + b)`
   - Activation (ReLU): `ReLU(X) = max(0, X)`
2. **Fully Connected Layers**
   - Fully Connected: `FC(X) = W^T * X + b`
3. **Output Layer with Sigmoid Activation for binary classification**
   - `sigma(X) = 1 / (1 + exp(-X))`

### Model 2: Fully Connected Neural Network (NN)

- **Developed by H. Wang.**
- This model is a traditional fully connected neural network that applies interaction terms and local statistical features (mean and standard deviation) during preprocessing.

**Model Architecture:**

The NN consists of several fully connected layers with ReLU activations:
1. **Hidden Layers**
   - Fully Connected: `FC(X) = W^T * X + b`
   - Activation (ReLU): `ReLU(X) = max(0, X)`
2. **Output Layer with Sigmoid Activation**
   - `sigma(X) = 1 / (1 + exp(-X))`

### Model 3: XGBoost

- **Developed by S. Chen.**
- XGBoost is a gradient boosting technique that incorporates regularization, parallel processing, and tree pruning to handle complex relationships between features and target variables.

**Objective Function:**

The objective function for XGBoost includes a regularization term to prevent overfitting:

```
Objective(theta) = Sum(Loss(y_i, y_hat_i)) + Sum(Regularization(f_j))
```
where `Loss(y_i, y_hat_i)` represents the loss function (such as logistic loss for binary classification) and `Regularization(f_j)` is the term added to prevent overfitting by penalizing model complexity, particularly in the trees used by XGBoost.

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
Thanks for your grading!



