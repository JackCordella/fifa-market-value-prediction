"""
Full training and prediction pipeline.

Written as top-level script code so that the run reads top to bottom in the order the steps
actually happen: load the data, pre-process it, select features, split and scale, compare
the candidate models, cross-validate the winner, then retrain it on everything and predict
on the test set.

Run:

    python main.py
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: plt.show() must not block a script

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import seaborn as sns
from tqdm import tqdm

from config import DATA_DIR, IMG_DIR, F_CSV_TRAIN, F_CSV_TEST, F_CSV_SUB, SEED
from src.data import (
    load_data,
    features_pre_selection,
    features_embedding,
    data_evaluation,
    data_pre_processing,
)
from src.features import (
    get_target_correlated_features,
    get_highly_correlated_features,
    heatmap_plot,
    features_selection,
)
from src.model import SimpleNeuralNetwork


# ============================================================================
# DATA LOADING, PRE-PROCESSING, FEATURE SELECTION, SPLIT AND SCALING
# ============================================================================

# Fixing seed for replicability
torch.manual_seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.mps.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


verbose_train = True
verbose_test = True 


# 1. DATA LOADING
df_train = load_data(F_CSV_TRAIN, verbose_train)
df_test = load_data(F_CSV_TEST, verbose_test)    


# 2. - DATASET PRE-PROCESSING
df_train = data_pre_processing(df_train, verbose=verbose_train)
df_test = data_pre_processing(df_test, verbose=verbose_test)

# 3. FEATURES SELECTION
target = 'value_eur'
df_train = features_selection(df_train, target, verbose=verbose_train)

# 4. DATASET DEFINITION
# 4.1 For the training dataset
y = df_train[target].values                     # target variable as numpy array
X = df_train.drop(columns=[target]).values      # features  as numpy array

# Only for the training dataset
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2, random_state=SEED, shuffle=False)

# 4.2 For the test dataset (without target variable)
lst_features = df_train.columns.tolist()
lst_features.remove(target)
X_test = df_test[lst_features].values                       
if verbose_test:
    print(f"TEST DATASET - Number of features: {len(lst_features)}")
    print(f"TEST DATASET - Number of samples: {X_test.shape[0]}")


# 5. DATA NORMALIZATION
# We use two different scalers: one for the features and one for the target variable
# to avoid the error due to the different dimensions of the dataset
scaler_X = MinMaxScaler()  # For the features
scaler_y = MinMaxScaler()  # For the target variable

# Use the scaler to normalize the data 
# Use fit_transform for features and transform for target
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))  

# "transform" (not "fit_transform") the validation set and the test set
# using the parameters calculated on the training set
# This is important for two reasons:
# 1. Avoid data leakage:
#    If we fit the scaler on the validation or test data, we would be using information from those sets to scale the training data.
#    This could lead to overfitting and an unrealistic evaluation of the model's performance.
# 2. Consistency:
#    We want to ensure that the model sees the data in the same scaled space.
#    By using the same scaler fitted on the training data, we ensure that the validation and test data are scaled in the same way.

X_valid_scaled = scaler_X.transform(X_valid)
y_valid_scaled = scaler_y.transform(y_valid.reshape(-1, 1))

X_test_scaled = scaler_X.transform(X_test)


# ============================================================================
# MODEL ENGINEERING: TRAIN AND COMPARE THE CANDIDATE MODELS
# ============================================================================

 
# 6. MODEL ENGINEERING
MODELS = {
    "LINEAR REGRESSION": LinearRegression(),
    "RIDGE REGRESSION": Ridge(alpha=0.1),  # L2 regularization
    "LASSO REGRESSION": Lasso(alpha=0.1),  # L1 regularization
    "ELASTIC NET": ElasticNet(alpha=0.1, l1_ratio=0.5),  # Combination of L1 and L2
    "ADA BOOST": AdaBoostRegressor(
                    estimator=DecisionTreeRegressor(max_depth=10),  # Weak learner
                    n_estimators=100,                             # Number of weak learners
                    learning_rate=0.5,                             # Learning rate
                    random_state=SEED                              # For reproducibility
                ),
    "RANDOM FOREST": RandomForestRegressor(
                        n_estimators=350,       # Number of trees in the forest
                        min_samples_leaf= 1,    # Minimum number of samples required to be at a leaf node
                        min_samples_split=2,    # Minimum number of samples required to split an internal node
                        max_depth=100,           # Maximum depth of the trees
                        random_state=SEED,      # For reproducibility
                        n_jobs=-1               # Use all available CPU cores
                    ),  
    "GRADIENT BOOSTING": GradientBoostingRegressor(
                    n_estimators=1900,       # Number of boosting stages
                    learning_rate=0.1,      # Learning rate shrinks the contribution of each tree
                    max_depth=3,            # Maximum depth of the individual regression estimators
                    min_samples_split=4,    # Minimum number of samples required to split an internal node
                    min_samples_leaf=2,     # Minimum number of samples required to be at a leaf node
                    random_state=SEED       # For reproducibility
                ),        
    "SIMPLE NEURAL NETWORK": SimpleNeuralNetwork(
                        input_size=X_train.shape[1], # Number of features
                        hidden_size=256,             # Number of neurons in the hidden layer
                        output_size=1,               # Number of output neurons (1 for regression)   
                        batch_size=32,               # Batch size for training 
                        learning_rate=0.0001,        # Learning rate for the optimizer
                        epochs=1000)                 # Number of epochs for training       
}


best_rmse = float('inf')  # Initialize to infinity
best_model = None
best_model_name = None
best_model_rmse = None

# Main loop for training and evaluating the models
for name, model in MODELS.items():

    print(f"\n\n=== Training {name} model ===")    

    # Train the model on the training set
    if name == "SIMPLE NEURAL NETWORK":
        model.fit(X_train_scaled, y_train_scaled, X_valid_scaled, y_valid_scaled, verbose=True)
    else:
        model.fit(X_train_scaled, y_train_scaled.ravel())

    # Predict on the validation set
    y_valid_pred = model.predict(X_valid_scaled)

    # Inverse transform the predictions to the original scale
    real_valid_pred = scaler_y.inverse_transform(y_valid_pred.reshape(-1, 1))

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_valid, real_valid_pred))
    print(f"{name} - RMSE: {rmse:.1f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_model_name = name
        print(f"Now the best model is {best_model_name} with RMSE: {best_rmse:.1f}")

print(f"\n\n****************************************************************")    
print(f"The BEST MODEL is {best_model_name} with RMSE: {best_rmse:.1f}")
print(f"****************************************************************\n\n")


# ============================================================================
# CROSS-VALIDATION OF THE BEST MODEL
# ============================================================================

# Set up the folds for cross-validation
folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)

print(f"Cross-validation with {folds} folds using the best model: {best_model_name}")

# Create a list to store the RMSE for each fold
rmse_list = []

X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

# Loop through each fold
for fold, (train_index, val_index) in enumerate(kf.split(X_tensor)):
    
    print(f"\nFold {fold + 1} of {folds}")

    # Reinitialize the best model
    best_model = MODELS[best_model_name]

    # Split the data into training and validation sets
    X_train_fold, X_val_fold = X_tensor[train_index], X_tensor[val_index]
    y_train_fold, y_val_fold = y_tensor[train_index], y_tensor[val_index]

    # We use two different scalers: one for the features and one for the target variable
    # to avoid the error due to the different dimensions of the dataset
    scaler_X = MinMaxScaler()  # For the features
    scaler_y = MinMaxScaler()  # For the target variable

    # Use the scaler to normalize the data 
    # Use fit_transform for features and transform for target
    X_train_scaled = scaler_X.fit_transform(X_train_fold)
    y_train_scaled = scaler_y.fit_transform(y_train_fold.reshape(-1, 1)) 

    X_valid_scaled = scaler_X.transform(X_val_fold)
    y_valid_scaled = scaler_y.transform(y_val_fold.reshape(-1, 1))

    # Train the model on the training set
    if best_model_name == "SIMPLE NEURAL NETWORK":
        best_model.fit(X_train_scaled, y_train_scaled, X_valid_scaled, y_valid_scaled, verbose=False)
    else:
        best_model.fit(X_train_scaled, y_train_scaled.ravel())


    # Predict on the validation set
    y_val_pred = best_model.predict(X_valid_scaled)

    # Inverse transform the predictions to the original scale
    real_val_pred = scaler_y.inverse_transform(y_val_pred.reshape(-1, 1))

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val_fold, real_val_pred))
    print(f"RMSE for fold {fold + 1}: {rmse:.1f}")
    
    # Append the RMSE to the list
    rmse_list.append(rmse)


# Calculate the average RMSE across all folds
avg_rmse = np.mean(rmse_list)
print(f"\n\n****************************************************************")    
print(f"       Average RMSE across {folds} folds: {avg_rmse:.1f}")
print(f"****************************************************************\n\n")


# ============================================================================
# RETRAIN THE BEST MODEL AND PREDICT ON THE TEST SET
# ============================================================================

# Retrain the best model on the entire training set
print(f"\n\n=== Retraining the best model {best_model_name} on the entire training set ===")

# Reinitialize the best model
best_model = MODELS[best_model_name]

# We use two different scalers: one for the features and one for the target variable
# to avoid the error due to the different dimensions of the dataset
scaler_X = MinMaxScaler()  # For the features
scaler_y = MinMaxScaler()  # For the target variable

# Use the scaler to normalize the data
# Use fit_transform for features and transform for target
X_train_scaled = scaler_X.fit_transform(X)
y_train_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Train the model on the entire training set
if best_model_name == "SIMPLE NEURAL NETWORK":
    best_model.fit(X_train_scaled, y_train_scaled, X_train_scaled, y_train_scaled, verbose=False)
else:
    best_model.fit(X_train_scaled, y_train_scaled.ravel())


# Predict on the TEST set
y_test_pred = best_model.predict(X_test_scaled)

# Inverse transform the predictions to the original scale
real_test_pred = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1))

print(f"\nTEST DATASET - Prediction {real_test_pred}")

# Save the predictions to a CSV file
df_predictions = pd.DataFrame(real_test_pred, columns=["value_eur"])
df_predictions.to_csv(F_CSV_SUB, index=False)
print(f"\nSUBMISSION FILE {F_CSV_SUB} saved!\n\n")


