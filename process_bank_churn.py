import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def split_data(
        raw_df: pd.DataFrame, target_col: str
) -> Dict[str, pd.DataFrame]:
    """
    Split the dataframe into training and validation sets.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.
        target_col (str): Target column.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing the train
            and validation dataframes.
    """
    train_df, val_df = train_test_split(
        raw_df, test_size=0.2, random_state=24, stratify=raw_df[target_col]
    )

    return {'train': train_df, 'val': val_df}


def create_inputs_targets(
        df_dict: Dict[str, pd.DataFrame],
        input_cols: list[str],
        target_col: str
) -> Dict[str, Any]:
    """
    Create inputs and targets for training and validation sets.

    Args:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing the train
            and validation dataframes.
        input_cols (list): List of input columns.
        target_col (str): Target column.

    Returns:
        Dict[str, Any]: Dictionary containing inputs and targets
            for train and validation sets.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()

    return data


def impute_missing_values(
        data: Dict[str, Any], numeric_cols: list[str], strategy: str = 'median'
) -> None:
    """
    Imputes missing values in numeric columns using the specified strategy
        ('mean' or 'median').

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets
            for train and validation sets.
        numeric_cols (list): List of numerical columns.
        strategy (str): The imputation strategy. Default is 'median'.
    """
    imputer = SimpleImputer(strategy=strategy).fit(
        data['train_inputs'][numeric_cols]
    )
    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = imputer.transform(
            data[f'{split}_inputs'][numeric_cols]
        )

    return data


def scale_numeric_features(
        data: Dict[str, Any], numeric_cols: list[str]
) -> None:
    """
    Scales numeric features using MinMaxScaler.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets
            for train and validation sets.
        numeric_cols (list): List of numerical columns.
    """
    scaler = MinMaxScaler().fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = scaler.transform(
            data[f'{split}_inputs'][numeric_cols]
        )
    data['scaler'] = scaler


def encode_categorical_features(
        data: Dict[str, Any], categorical_cols: list[str]
) -> None:
    """
    One-hot encode categorical features.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets
            for train and validation sets.
        categorical_cols (list): List of categorical columns.
    """
    encoder = OneHotEncoder(
        drop='if_binary', sparse_output=False, handle_unknown='ignore'
    )
    encoder.fit(data['train_inputs'][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    for split in ['train', 'val']:
        encoded = encoder.transform(data[f'{split}_inputs'][categorical_cols])
        encoded_df = pd.DataFrame(
            encoded, columns=encoded_cols, index=data[f'{split}_inputs'].index
        )
        data[f'{split}_inputs'] = pd.concat(
            [data[f'{split}_inputs'].drop(columns=categorical_cols),
             encoded_df],
            axis=1
            )
    data['encoded_cols'] = encoded_cols
    data['encoder'] = encoder


def preprocess_data(
        raw_df: pd.DataFrame,
        target_col: str,
        scaler_numeric: bool = True,
        impute_strategy: str = 'median'
) -> Dict[str, Any]:
    """
    Preprocess the raw dataframe.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.
        target_col (str): Target column.
        scaler_numeric (bool): Whether to scale numeric features.
            Default is True.
        impute_strategy (str): Strategy for imputing missing values.
            Default is 'median'.

    Returns:
        Dict[str, Any]: Dictionary containing processed inputs and targets
            for train and validation sets.
    """
    split_dfs = split_data(raw_df, target_col)
    input_cols = list(raw_df.columns)[2:-1]
    data = create_inputs_targets(split_dfs, input_cols, target_col)

    numeric_cols = (data['train_inputs']
                    .select_dtypes(include=np.number)
                    .columns.tolist())
    categorical_cols = (data['train_inputs']
                        .select_dtypes('object')
                        .columns.tolist())

    impute_missing_values(data, numeric_cols, impute_strategy)
    if scaler_numeric:
        scale_numeric_features(data, numeric_cols)
    encode_categorical_features(data, categorical_cols)

    # Extract X_train, X_val
    X_train = data['train_inputs'][numeric_cols + data['encoded_cols']]
    X_val = data['val_inputs'][numeric_cols + data['encoded_cols']]

    return {
        'train_X': X_train,
        'train_y': data['train_targets'],
        'val_X': X_val,
        'val_y': data['val_targets'],
        'scaler': data['scaler'],
        'encoder': data['encoder'],
        'input_cols': input_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
    }


def preprocess_new_data(
    new_data: pd.DataFrame,
    input_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    encoder: OneHotEncoder,
    scaler: MinMaxScaler = None,
) -> pd.DataFrame:
    """
    Preprocesses new data using a previously trained scaler and encoder.
    """
    new_data = new_data[input_cols]
    if scaler:
        new_data.loc[:, numeric_cols] = scaler.transform(
            new_data[numeric_cols]
        )
    encoded = encoder.transform(new_data[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=new_data.index
    )
    new_data = pd.concat(
        [new_data.drop(columns=categorical_cols), encoded_df],
        axis=1
    )
    return new_data
