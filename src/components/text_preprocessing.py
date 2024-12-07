import os
import sys
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.exception import CustomException
from src.logger import get_logger
from src.utils import fetch_train_test_data, data_to_dependent_and_independent_features

logger = get_logger('text-preprocessing')

class TextPreprocessing:
    def __init__(self):
        pass

    def cleaning_text(self, data):
        '''
        This functions take data and cleans it then returns it.

        Parameters:
            data: Raw data.

        Returns:
            clean data.
        '''

        try:
            logger.info("Cleaning our data")

            lemmatizer = WordNetLemmatizer()

            COL_NAME = 'review'

            for i in range(data.shape[0]):
                text = data[COL_NAME][i]
                re_text = re.sub(r'[^a-zA-Z]', ' ', text)
                tokens = [word.lower() for word in re_text.split()]
                clean_text = ' '.join(tokens)
                data[COL_NAME][i] = clean_text

            logger.info('Data Cleaned')

            return data
        
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def initiate_preprocessing(self, train_path, test_path):
        '''
        This function fetches the train and test data and then start cleaning on the data.

        Parameters:
            train_path (str): Training data path.
            test_path (str): Testing data path.

        Returns:
            X_train, X_test, y_train, y_test.
        '''

        try:
            logger.info("Starting Text Preprocessing")

            train_data, test_data = fetch_train_test_data(train_path=train_path, test_path=test_path)

            logger.info('Cleaning our train data')
            clean_train_data = self.cleaning_text(train_data)

            logger.info('Cleaning our test data')
            clean_test_data = self.cleaning_text(test_data)

            logger.info('Creating X_train and y_train from clean_train_data')
            X_train, y_train = data_to_dependent_and_independent_features(clean_train_data)
            logger.info(f'Shape X_train: {X_train.shape}, y_train: {y_train.shape}')

            logger.info('Creating X_test and y_test from clean_test_data')
            X_test, y_test = data_to_dependent_and_independent_features(clean_test_data)
            logger.info(f'Shape X_test: {X_test.shape}, y_test:{y_test.shape}')

            logger.info('Text Preprocessing and Cleaning Completed Successfully')

            return (
                X_train, 
                X_test,
                y_train,
                y_test
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)