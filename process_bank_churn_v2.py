import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures


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
        data[f'{split}_inputs'].loc[:, numeric_cols] = scaler.transform(
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
    ).set_output(transform='pandas')
    encoder.fit(data['train_inputs'][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    for split in ['train', 'val']:
        data[f'{split}_inputs'].loc[:, encoded_cols] = encoder.transform(
            data[f'{split}_inputs'][categorical_cols]
        )
        data[f'{split}_inputs'] = (data[f'{split}_inputs']
                                   .drop(columns=categorical_cols))
    data['encoded_cols'] = encoded_cols
    data['encoder'] = encoder


def create_polynomial_features(
        data: Dict[str, Any], numeric_cols: list[str], degree: int
) -> None:
    """
    One-hot encode categorical features.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets
            for train and validation sets.
        numeric_cols (list): List of numeric columns.
        degree (int): Specifies the maximal degree of the polynomial features.
    """

    poly_transformer = PolynomialFeatures(degree, include_bias=False)
    poly_transformer.fit(data['train_inputs'][numeric_cols])
    poly_transformer.set_output(transform='pandas')

    poly_cols = list(poly_transformer.get_feature_names_out(numeric_cols))
    for split in ['train', 'val']:
        data[f'{split}_inputs'].loc[:, poly_cols] = poly_transformer.transform(
            data[f'{split}_inputs'][numeric_cols]
        )
    data['poly_cols'] = poly_cols
    data['poly_transformer'] = poly_transformer


def preprocess_data(
        raw_df: pd.DataFrame,
        target_col: str,
        scaler_numeric: bool = True,
        polynomial_features: bool = False,
        polynomial_degree: int = 2
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
        polynomial_features(bool): Whether to create polynomial features.
            Default is False.
        polynomial_degree (int): Specifies the maximal degree
            of the polynomial features. Default=2.

    Returns:
        Dict[str, Any]: Dictionary containing processed inputs and targets
            for train and validation sets.
    """
    split_dfs = split_data(raw_df, target_col)
    input_cols = list(raw_df.columns)[2:-1]
    data = create_inputs_targets(split_dfs, input_cols, target_col)

    numeric_cols = (data['train_inputs']
                    .select_dtypes(include='number')
                    .columns.tolist())
    categorical_cols = (data['train_inputs']
                        .select_dtypes('object')
                        .columns.tolist())

    if scaler_numeric:
        scale_numeric_features(data, numeric_cols)
    else:
        data['scaler'] = None

    encode_categorical_features(data, categorical_cols)

    if polynomial_features:
        create_polynomial_features(data, numeric_cols, polynomial_degree)
    else:
        data['poly_cols'] = []
        data['poly_transformer'] = None

    # Extract X_train, X_val
    if polynomial_features:
        X_train = data['train_inputs'][data['poly_cols']
                                       + data['encoded_cols']]
        X_val = data['val_inputs'][data['poly_cols']
                                   + data['encoded_cols']]
    else:
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
        'poly_cols': data['poly_cols'],
        'poly_transformer': data['poly_transformer'],
    }


def preprocess_new_data(
    new_data: pd.DataFrame,
    input_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    poly_cols: list[str],
    encoder: OneHotEncoder,
    scaler: MinMaxScaler = None,
    poly_transformer: PolynomialFeatures = None,
) -> pd.DataFrame:
    """
    Preprocesses new data using a previously trained scaler and encoder.
    """
    preprocessed_new_data = new_data[input_cols].copy()
    if scaler:
        preprocessed_new_data.loc[:, numeric_cols] = scaler.transform(
            preprocessed_new_data[numeric_cols]
        )

    if poly_transformer:
        preprocessed_new_data.loc[:, poly_cols] = poly_transformer.transform(
            preprocessed_new_data[numeric_cols]
        )

    encoded = encoder.transform(preprocessed_new_data[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=preprocessed_new_data.index
    )

    preprocessed_new_data = pd.concat(
        [preprocessed_new_data.drop(columns=categorical_cols), encoded_df],
        axis=1
    )
    return preprocessed_new_data


def compute_auroc(model: BaseEstimator,
                  data_dict: Dict[str, Any]) -> Tuple[float, float]:
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (AUROC)
    for both the training and validation sets.

    Args:
        model (BaseEstimator): A trained classification model that supports
            probability predictions via `predict_proba`.
        data_dict (Dict[str, Any]): A dictionary containing the training
            and validation datasets with the following keys:
            - 'train_X': Features for training set (pd.DataFrame).
            - 'train_y': Target labels for training set (pd.Series).
            - 'val_X': Features for validation set (pd.DataFrame).
            - 'val_y': Target labels for validation set (pd.Series).

    Returns:
        Tuple[float, float]: A tuple containing:
            - AUROC score for the training set.
            - AUROC score for the validation set.
    """
    predict_train_y = model.predict_proba(data_dict['train_X'])[:, 1]
    predict_val_y = model.predict_proba(data_dict['val_X'])[:, 1]

    train_auroc = roc_auc_score(data_dict['train_y'], predict_train_y)
    val_auroc = roc_auc_score(data_dict['val_y'], predict_val_y)

    return train_auroc, val_auroc
