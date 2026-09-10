"""
Dataset loading and pre-processing.

Extracted verbatim from `ML_Project_notebook.ipynb` (cells 12, 14, 16, 18, 20).
The function bodies are unchanged; only the import header below was added.
"""

import pandas as pd


def load_data(file_path, verbose=True):
    """
    This function is used to load the data from a csv file
    @param file_path: the path of the csv file
    @return: the dataset loaded
    """
    dataset = pd.read_csv(file_path) # Load the dataset

    if verbose:
        print(f"DATASET LOADED - Number of features: {dataset.shape[1]}")
        print(f"DATASET LOADED - Number of samples: {dataset.shape[0]}")

    return dataset


def features_pre_selection (dataset):
    """
    This function is used to eliminate the features that are useless
    @param dataset: the dataset to be preprocessed
    @return: the dataset without the useless features
    """

    # Select the feature that are useless (without any useful information)
    features_useless = ['Unnamed: 0','id','short_name','long_name','dob','club_name',
                'league_name', 'club_jersey_number', 'club_joined','nationality_name',
                'nation_jersey_number', 'club_team_id', 'nationality_id','preferred_foot', 
                'work_rate', 'body_type', 'player_tags', 'player_traits']
    
    # Eliminating the features that are useless
    df_pre_selected = dataset.drop(features_useless, axis=1)

    return df_pre_selected


def features_embedding(dataset):
    """
    This function is used to preprocess the dataset for the features embedding
    @param dataset: the dataset to be processed
    @return: the dataset with the processed features
    """

    # "Target Encoding": transform 'player_positions' into embedded features
    # Every category is replaced by the mean of release_clause_eur for that category.
    if 'player_positions' in dataset.columns:
        mean_target = dataset.groupby('player_positions')['release_clause_eur'].mean()
        dataset['player_positions'] = dataset['player_positions'].map(mean_target)
        dataset = dataset.drop(['player_positions'], axis=1)

    # Selecting if player plays in the national team
    dataset['nation_position'] = dataset['nation_position'].apply(lambda x: 0 if pd.isna(x) else 1)

    # Selecting if the player is loaned
    dataset['club_loaned_from'] = dataset['club_loaned_from'].apply(lambda x: 0 if pd.isna(x) else 1)

    # Selecting the playing players
    dataset['club_position'] = dataset['club_position'].apply(lambda x: 'PLAY' if x != 'SUB' and x != 'RES' else x)
    dummy_play  = pd.get_dummies(dataset['club_position'])
    dataset = pd.concat([dataset, dummy_play], axis=1)
    dataset = dataset.drop(['club_position'], axis=1)

    # Selecting data end contract
    if dataset['club_contract_valid_until'].notnull().any():
        # Sobistituting the NaN values with the minimum value of the column
        min_value = dataset['club_contract_valid_until'].min()    
        dataset['club_contract_valid_until'] = dataset['club_contract_valid_until'].fillna(min_value)

    constract_duration = dataset['club_contract_valid_until'] - dataset['club_contract_valid_until'].min()
    dataset = pd.concat([dataset, constract_duration], axis=1)
    dataset = dataset.drop(['club_contract_valid_until'], axis=1)    

    # For all the other features, fill the NaN values with 0
    dataset = dataset.fillna(0)

    return dataset


def data_evaluation(dataset, verbose=True):
    """
    This function is used to evaluate the quality if the dataset in terms of NaN values.
    Print the features with NaN values and the number of NaN values for each feature.
    @param dataset: the dataset to be evaluated
    @return: None
    """

    # Evaluate the dataset, managing the NaN values
    features_with_nan = dataset.columns[dataset.isnull().any()].tolist()
    if len(features_with_nan) == 0:
        if verbose:
            print("DATASET EVALUATION - No features with NaN value.")
    else:
        if verbose:
            print("DATASET EVALUATION - Features with NaN value:", features_with_nan)
        df_whit_nan_counts = dataset.isnull().sum()

        for col in features_with_nan:
            print(f"DATASET EVALUATION - {col} contains {df_whit_nan_counts[col]} NaN values.")

    return None


def data_pre_processing(dataset, verbose=True):
    """
    This function is used to preprocess the dataset, not only the features, but also the target variable
    This function is divided into four parts:
        1. Feature pre-selection (eliminating the features that are useless)
        2. Eliminating the rows with NaN values in the 'value_eur' column
        3. Transforming the categorical features into numerical features
        4. Evaluating the dataset for the quality of the data (only if verbose is True)
    @param dataset: the raw dataset to be preprocessed
    @return: the dataset preprocessed features
    """
    
    # 1. Feature pre-selection (eliminating the features that are useless)   
    dataset = features_pre_selection(dataset)
    if verbose:
        print(f"DATASET PRE-SELECTED - Number of features: {dataset.shape[1]}")


    # 2. Eliminating the rows with NaN values in the 'value_eur' column
    if 'value_eur' in dataset.columns:
        # Eliminating the rows with NaN values in the 'value_eur' column
        dataset = dataset.dropna(subset=['value_eur'])
 
    if verbose:    
        print(f"DATASET PRE-SELECTED - Number of samples: {dataset.shape[0]}")

    # 3. Transforming the categorical features into numerical features
    dataset = features_embedding(dataset)

    # 4. Evaluating the dataset for the quality of the data
    if verbose:
        data_evaluation(dataset)


    return dataset
