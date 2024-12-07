import sys

from src.exception import CustomException
from src.logger import get_logger

from src.components.data_ingestion import DataIngestion
from src.components.text_preprocessing import TextPreprocessing
from src.components.text_transformation import TextTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluater import ModelEvaluation

logger = get_logger('Prediction-Pipeline')

def main():
    '''
    This function will initiate the whole training pipeline
    '''
    logger.info('Training Pipeline Initialized')

    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.load_data() # getting the train and test path

    text_preprocessing = TextPreprocessing()
    X_train, X_test, y_train, y_test = text_preprocessing.initiate_preprocessing(train_path, test_path) # getting the X_train, X_test, y_train, y_test

    text_transformation = TextTransformation()
    vocab_size, transformed_train_data, y_train, transformed_test_data, y_test = text_transformation.apply_transformation(X_train, X_test, y_train, y_test)

    model_trainer = ModelTrainer()
    model_path = model_trainer.start_training(transformed_train_data, y_train, transformed_test_data, y_test, vocab_size)

    model_evaluation = ModelEvaluation()
    train_score, test_score = model_evaluation.evaluate(model_path, transformed_train_data, y_train, transformed_test_data, y_test)

    print(f'Model Training Score: {train_score}')
    print(f'Model Testing Score: {test_score}')

    logger.info('Training Pipeline Ended Successfully')

if __name__ == '__main__':
    main()