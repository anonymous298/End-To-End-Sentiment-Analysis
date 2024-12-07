import os
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

logger = get_logger('text-transformation')

@dataclass
class TextTransformationObjectPath:
    tokenizer_path: str = os.path.join('model', 'tokenizer.pkl')

class TextTransformation:
    def __init__(self):
        self.text_transformation_object_path = TextTransformationObjectPath()

    def train_transformation(self, train_data):
        '''
        This function transforms the training data.

        Parameters:
            train_data: training data.

        Returns:
            vocab_size, transformed data and also saved the tokenizer object
        '''

        try:

            logger.info('Training our tokenizer object for train data')
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(train_data)

            vocab_size = len(tokenizer.word_index) + 1

            logger.info('Converting train data to sequences')
            sequences = tokenizer.texts_to_sequences(train_data)

            logger.info('Equalling the size of all sequences')
            MAXLEN = 1000
            transformed_train_data = pad_sequences(sequences, maxlen=MAXLEN)

            logger.info('Saving our Tokenizer object')
            os.makedirs(os.path.dirname(self.text_transformation_object_path.tokenizer_path), exist_ok=True)

            save_model(
                file_path=self.text_transformation_object_path.tokenizer_path,
                model=tokenizer
            )

            logger.info('Training data transformed')

            return (
                vocab_size,
                transformed_train_data
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def test_transformation(self, test_data):
        '''
        This function transforms the testing data.

        Parameters:
            test_data: testing data.

        Returns:
            transformed data.
        '''

        try:
            
            logger.info('Training our tokenizer object for test data')
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(test_data)

            logger.info('Converting test data to sequences')
            sequences = tokenizer.texts_to_sequences(test_data)

            logger.info('Equalling the size of all sequences')
            MAXLEN = 1000
            transformed_test_data = pad_sequences(sequences, maxlen=MAXLEN)

            logger.info('Testing data transformed')

            return transformed_test_data

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def apply_transformation(self, X_train, X_test, y_train, y_test):
        '''
        This function converts the data into model ready to train data.

        Parameters:
            X_train: X_training data.
            X_test: X_testing data.
            y_train: y_training data.
            y_test: y_testing data.

        Returns:
            vocab_size, transformed_train_data, y_train, transformed_test_data, y_test.
        '''

        try:
            logger.info('Applying transformation on text data')

            logger.info('Transforming our train data')
            vocab_size, transformed_train_data = self.train_transformation(X_train)
            logger.info(f'Transformed train data shape: {transformed_train_data.shape}')

            logger.info('Transforming our test data')
            transformed_test_data = self.test_transformation(X_test)
            logger.info(f'Transformed test data shape: {transformed_test_data.shape}')

            logger.info('Text Transformation completed successfully')


            return (
                vocab_size,
                transformed_train_data,
                y_train,
                transformed_test_data,
                y_test
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)