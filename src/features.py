"""
Correlation analysis and feature selection.

`features_selection` is the entry point: it drops features that correlate strongly with each
other (keeping the redundant one closest to the target), then those that correlate weakly
with the target, plotting the correlation heatmap before and after.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import data_evaluation


def get_target_correlated_features(dataset, target, threshold=0.25, verbose=True):
    """
    This function is used to get the features that are correlated with the target variable
    @param dataset: the dataset to be processed
    @param feature: the main feature to be correlated with
    @param threshold: the threshold for the correlation
    @return: list of the features that are low correlated with the main feature
    """

    # Correlation matrix with absolute values
    corr_matrix = dataset.corr().abs()  

    # Selecting the correlation matrix with the main feature
    df_corr_data_value_eur = corr_matrix[target].abs()

    lst_features_to_drop = []
    # Loop through the correlation matrix and select the features that are low correlated with the main feature
    for column in df_corr_data_value_eur.index:
        if column == 'value_eur':
            continue
        if df_corr_data_value_eur[column] <= threshold:
            lst_features_to_drop.append(column)
        elif df_corr_data_value_eur[column] > 0.80:
            if verbose:
                print(f"get_target_correlated_features - Feature {column} is highly correlated with the target variable with a correlation of {df_corr_data_value_eur[column]}")

    if verbose:
        print(f'get_target_correlated_features - List of features to drop: {lst_features_to_drop}')
    return lst_features_to_drop



def get_highly_correlated_features(dataset, threshold=0.90, verbose=True):
    """ 
    This function is used to get the features that are highly correlated with each other
    @param dataset: the dataset to be processed
    @param threshold: the threshold for the correlation
    @return: list of the features that are highly correlated with each other
    """

    # Correlation matrix with absolute values
    corr_matrix = dataset.corr().abs()  

    # Upper triangle of the correlation matrix
    # to avoid duplicate pairs and self-correlation
    # np.triu returns the upper triangle of an array
    # where k=1 means we want to exclude the diagonal and the lower triangle
    # np.ones(corr_matrix.shape) creates a matrix of ones with the same shape as corr_matrix
    # astype(bool) converts the matrix to boolean, where True indicates the upper triangle
    # and False indicates the lower triangle
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # Triangolo superiore

    # Trova le features con correlazione superiore alla soglia
    lst_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    lst_features_to_drop = list(set(lst_features))  # Remove duplicates

    if 'release_clause_eur' in lst_features_to_drop:
        # Remove this feature because it is highly correlated to target variable
        # and we want to keep it for the model
        lst_features_to_drop.remove('release_clause_eur')  # Remove this feature because it is higly correlated to target variable

    if 'wage_eur' in lst_features_to_drop:
        # Remove this feature because it is highly correlated to target variable
        # and we want to keep it for the model
        lst_features_to_drop.remove('wage_eur')  # Remove this feature because it is higly correlated to target variable

    if verbose:
        print(f'get_highly_correlated_features - List of features to drop: {lst_features_to_drop}')
    
    return lst_features_to_drop


def heatmap_plot(dataset, fase, verbose=False):
    """
    This function is used to plot the heatmap of the correlation matrix
    and if verbose is True save it as a png file
    @param dataset: the dataset to be processed
    @param fase: the phase of the processing (BEFORE or AFTER)
    @param verbose: if True save the heatmap as a png file
    @return: None
    """

    corr_matrix = dataset.corr().round(2)

    # Set parameters based on the phase
    if fase == "BEFORE":
        filename = "Heatmap_Before_Processing.png"
        fig_width = 35
        fig_height = 30
        title = "Correlation Heatmap Before Feature Selection"
    else:
        filename = "Heatmap_After_Processing.png"
        fig_width = 10
        fig_height = 8
        title = "Correlation Heatmap After Feature Selection"

    # Set the theme for the heatmap
    sns.set_theme(style="whitegrid")


    # Create the heatmap
    plt.figure(figsize=(fig_width, fig_height))
    heatmap = sns.heatmap(
        data=corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        cbar=True,
        linewidths=0.5,  # Add gridlines for better readability
        annot_kws={"size": 10}  # Adjust annotation font size
    )
    heatmap.set_title(title, fontdict={'fontsize': 16}, pad=16)

    if verbose:
        # Save the heatmap as a PNG file
        os.makedirs("img", exist_ok=True)  # Ensure the directory exists
        file_path = os.path.join("img", filename)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved as '{file_path}'")        
    plt.show()



def features_selection(dataset, target, verbose=True):
    """ 
    This function is used to select the features that are useful for the model.
    This function is diveded into three parts:
        1. Identify and remove the features with high correlation, used for removing the redundant features
        2. Identify and remove the features with low correlation with the target variable, used for removing the features that are not informative
        3. Evaluating the dataset for the quality of the data (only if verbose is True)        
    @param dataset: the dataset to be processed
    @return: the dataset with the selected features
    """
    
    # Plot heatmap BEFORE feature selection
    heatmap_plot(dataset, "BEFORE", verbose=False)

    # 1. Identifying the features with high correlation
    threshold_high_correlation = 0.85
    lst_features_to_drop = get_highly_correlated_features(dataset, threshold_high_correlation, verbose=verbose)
    dataset = dataset.drop(columns=lst_features_to_drop)


    # 2. Identifying the features with low correlation with the main feature
    threshold_low_correlation = 0.50
    lst_features_to_drop = get_target_correlated_features(dataset, target, threshold_low_correlation, verbose=verbose)
    dataset = dataset.drop(columns=lst_features_to_drop)

    if verbose:
        print(f"DATASET FEATURES SELECTED - Number of features: {dataset.shape[1]}")
        print(f"DATASET FEATURES SELECTED - Number of samples: {dataset.shape[0]}")


    # 3. Evaluating the dataset for the quality of the data
    if verbose:
        data_evaluation(dataset)

    # Plot heatmap AFTER feature selection
    heatmap_plot(dataset, "AFTER", verbose=False)

    return dataset
