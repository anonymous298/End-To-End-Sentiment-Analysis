import pandas
import os
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger
from src.utils import split_data_to_train_test

logger = get_logger('data-ingestion')

@dataclass
class DataPaths:
    load_data_path: str = os.path.join('Notebook', 'data', 'IMDB Dataset.csv')
    save_train_data_path: str = os.path.join('artifacts', 'train.csv')
    save_test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.model_paths = DataPaths()

    def load_data(self):
        '''
        This functions loads the data and split it save it and then return the path.

        Returns:
            returns: Train and Test data path after saving it.
        '''

        try:
            logger.info('Data Ingestion Intialized')

            logger.info('Loading data from path')
            data = pd.read_csv(self.model_paths.load_data_path)

            train_data, test_data = split_data_to_train_test(data=data)

            logger.info('Saving our train and test data to path')
            os.makedirs(os.path.dirname(self.model_paths.save_train_data_path), exist_ok=True)

            logger.info('Saving our train data')
            train_data.to_csv(self.model_paths.save_train_data_path, index=False)

            logger.info('Saving our test data')
            test_data.to_csv(self.model_paths.save_test_data_path, index=False)

            logger.info('Data Ingestion completed')

            return (
                self.model_paths.save_train_data_path,
                self.model_paths.save_test_data_path
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)