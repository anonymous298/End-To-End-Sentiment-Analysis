import os
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

logger = get_logger('model-trainer')

@dataclass
class ModelSavingPath:
    model_path: str = os.path.join('model', 'model.h5')

class ModelTrainer:
    def __init__(self):
        self.model_saving_path = ModelSavingPath()
    
    def model_building(self, vocab_size):
        '''
        This function will build the sequential model and returns it.

        Parameters:
            vocab_size: vocabulary size for the Embedding layer.

        Returns:
            fully build model.
        '''

        try:
            logger.info('Building Our Sequential model')

            model = Sequential()
            model.add(Embedding(vocab_size, 100, input_length=1000))
            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(Dropout(0.5))
            model.add(GRU(32))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))

            logger.info('Compiling our model')
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            logger.info('Model Build Successfully')

            return model

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def start_training(self, X_train, y_train, X_test, y_test, vocab_size):
        '''
        This function will initiate training of our model.

        Parameters:
            X_train: X_transformed_data.
            y_train: y_train_data.
            X_test: X_transformed_data.
            y_test: y_transformed_data.
            vocab_size: vocabulary size for Embedding layer

        Returns:
            Save model and returns model path.
        '''

        try:
            logger.info('Model Training Started')

            model = self.model_building(vocab_size=vocab_size)

            logger.info('Setting callbacks for our model')
            es_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            os.makedirs('NN_Logs', exist_ok=True)
            tb_callback = TensorBoard(log_dir='NN_Logs/training1', histogram_freq=1)

            logger.info('Fitting our data')
            model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=10,
                batch_size=32,
                class_weight={0: 1.0, 1: 1.0},  
                callbacks=[es_callback,tb_callback]
            )

            logger.info('Saving our Sequential Model')
            model.save(self.model_saving_path.model_path)

            logger.info('Model trained successfully')

            return self.model_saving_path.model_path

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)