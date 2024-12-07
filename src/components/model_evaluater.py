import os
import sys

from src.exception import CustomException
from src.logger import get_logger

from tensorflow.keras.models import load_model

logger = get_logger('model-evaluation')

class ModelEvaluation:
    def __init__(self):
        pass

    def evaluate(self, model_path, X_train, y_train, X_test, y_test):
        '''
        This function will evaluate our model on training and testing data.

        Parameters:
            model_path: our trained model path.
            X_train: X_train_data
            y_train: y training data
            X_test: X_testing data
            y_tests: y testing data.

        Returns: 
            train and test score
        '''

        try:
            logger.info('Loading our trained Sequential model')
            model = load_model(model_path)

            logger.info('Evaluating for training data')
            _, train_score = model.evaluate(X_train, y_train)

            logger.info('Evaluating for testing data')
            _, test_score = model.evaluate(X_test, y_test)

            logger.info(f'Model score on training data: {train_score}')
            logger.info(f'Model score on testing data: {test_score}')

            logger.info('Model Evaluation completed')

            return (
                train_score, 
                test_score
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)