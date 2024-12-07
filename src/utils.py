import os
import sys
import pandas as pd
import numpy as np
import dill

from src.exception import CustomException
from src.logger import get_logger

from sklearn.model_selection import train_test_split

logger = get_logger('utils')

def split_data_to_train_test(data):
    '''
    This function takes raw data and split it into train and test part.

    Parameters:
        data: Raw CSV data.

    Returns:
        train and test splitted data.
    '''

    try:
        logger.info('Splitting data into train and test')
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        return (
            train_data,
            test_data
        )
    
    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)


def fetch_train_test_data(train_path, test_path):
    '''
    This function fetches the train and test data from path.

    Parameters:
        train_path (str): Training data path.
        test_path (str): Testing data path.

    Returns:
        train and test data.
    '''

    try:
        logger.info('Fetching train and test data from path')

        logger.info('Loading training data')
        train_data = pd.read_csv(train_path)

        logger.info('Loading testing data')
        test_data = pd.read_csv(test_path)

        logger.info('Data Fetched Successfully')

        return (
            train_data, 
            test_data
        )

    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)

def data_to_dependent_and_independent_features(data):
    '''
    This function takes data and extract independent and dependent features from it.

    Parameters:
        data.

    Returns:
     X and Y: Independent and Dependent feature.
    '''

    try:
        logger.info('Extracting Dependent and Independent features from data.')
        X_COL_NAME = 'review'
        Y_COL_NAME = 'sentiment'

        X = data[X_COL_NAME].values  # Get the actual review text values
        y = data[Y_COL_NAME].map({'positive' : 1, 'negative' : 0}) # this will also encode the values

        return (
            X,
            y
        )

    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)

def save_model(file_path, model):
    '''
    This function will save the model to path.

    Parameters:
        file_path: file path where the model should save.
        model: model that have to save.

    Returns: 
        None.
    '''

    try:
        logger.info('Saving our model to path')

        with open(file_path, 'wb') as f:
            dill.dump(model, f)

        logger.info('Model saved')

    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)

def load_pkl_model(file_path):
    '''
    This function will load the model from specified path.

    Parameters:
        file_path: file path where the model will be loaded.

    Returns:
        loaded model
    '''

    try:
        logger.info('Loading our model')
        with open(file_path, 'rb') as f:
            model = dill.load(f)

        return model

    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)

def sentiment_class_returner(prediction):
    '''
    This function will take the prediction and returns whether it is positive negative or netural.

    Parameters:
        prediction: model prediction on text.

    Returns:
        output Positive or Negative or Neutral.
    '''
    logger.info(f"Raw prediction value: {prediction[0][0]}")

    if prediction[0][0] >= 0.6:
        logger.info("Classified as Positive")
        return 'positive'
    elif prediction[0][0] <= 0.4:
        logger.info("Classified as Negative")
        return 'negative'
    else:
        logger.info("Classified as Neutral")
        return 'neutral'