import pandas as pd
import numpy as np


def prepare_input(dict_input, predictor):
    # define the columns
    numerical_columns = predictor['columns']['numerical']['all']
    categorical_columns = predictor['columns']['categorical']['all']
    encoded_columns = predictor['columns']['encoded_columns']
    # define the numerical imputer and categorical imputer
    numerical_imputer = predictor['imputer']['numerical']
    categorical_imputer = predictor['imputer']['categorical']
    # define columns results of feature selection
    retained_columns = predictor['feature_selection']['retained_columns']
    # define onehot encoder
    onehot_encoder = predictor['encoder']['encoder']
    # define the standardscaler for features scaling
    scaler = predictor['feature_scaling']['scaler']
    # define pca models
    pca_scaler = predictor['dimensionality_reduction']['pca_scaler_final']

    # anticipate for nan values
    for k, v in dict_input.items():
        if v == "nan" or v == "-":
            dict_input[k] = np.nan

    # define data
    input_df = pd.DataFrame(dict_input, index=[0])

    # imputer missing values
    input_df[numerical_columns] = numerical_imputer.transform(
        input_df[numerical_columns])
    input_df[categorical_columns] = categorical_imputer.transform(
        input_df[categorical_columns])

    # encoding
    input_df[encoded_columns] = pd.DataFrame(data=onehot_encoder.transform(
        input_df[categorical_columns]), columns=encoded_columns)

    # filter atribut numerikal dan hasil encoding
    input_df = input_df.loc[:,
                            numerical_columns + encoded_columns]

    # scaling the features
    input_df = scaler.transform(input_df)

    # dimensionality reduction
    X_inputs = pca_scaler.transform(input_df)

    return X_inputs
