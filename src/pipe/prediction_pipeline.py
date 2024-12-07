import os
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger
from src.utils import load_pkl_model, sentiment_class_returner

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

logger = get_logger('Prediction-Pipeline')

@dataclass
class PredictionPipelinePaths:
    tokenizer_path: str = os.path.join('model', 'tokenizer.pkl')
    model_path: str = os.path.join('model', 'model.h5')

class PredicitonPipeline:
    def __init__(self):
        self.prediction_pipeline_paths = PredictionPipelinePaths()
        
        # Check if model files exist
        if not os.path.exists(self.prediction_pipeline_paths.tokenizer_path):
            raise CustomException(f"Tokenizer file not found at {self.prediction_pipeline_paths.tokenizer_path}", sys)
        if not os.path.exists(self.prediction_pipeline_paths.model_path):
            raise CustomException(f"Model file not found at {self.prediction_pipeline_paths.model_path}", sys)

    def clean_text(self, text):
        '''
        This function will take the text and clean it.

        Parameters:
            text: uncleaned text given by user.

        Returns: 
            clean text.
        '''

        try:
            logger.info('Cleaing text')

            lemmatizer = WordNetLemmatizer()

            re_text = re.sub(r'[^a-zA-Z]', ' ', text)
            tokens = [lemmatizer.lemmatize(word.lower()) for word in re_text.split() if word not in set(stopwords.words('english'))]
            clean_text = ' '.join(tokens)
            
            logger.info('Text cleaned')

            return clean_text

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def transform_text(self, clean_text):
        '''
        This function will tranform the text and returns model ready to train input.

        Parameters:
            clean_text: clean text 

        Returns:
            model ready to train input
        '''

        try:
            logger.info('Transforming our text')
            
            logger.info('Loading our tokenizer object')

            tokenizer = load_pkl_model(file_path=self.prediction_pipeline_paths.tokenizer_path)
            
            text_seq = tokenizer.texts_to_sequences([clean_text])

            transformed_text = pad_sequences(text_seq, maxlen=1000)

            logger.info('Text is Transformed')

            return transformed_text

        except Exception as e:
            logger.error(f"Error in transform_text: {str(e)}")
            raise CustomException(e, sys)

    def predict_text(self, text):
        '''
        This function will take text and returns the prediction

        Parameters:
            text: text from user.

        Returns:
            prediction about our text.
        '''

        try:
            logger.info('starting our prediction')
            logger.info(f'Input text: {text}')

            clean_text = self.clean_text(text)
            logger.info(f'Cleaned text: {clean_text}')

            transformed_querie = self.transform_text(clean_text=clean_text)
            logger.info(f'Transformed shape: {transformed_querie.shape}')

            logger.info('Loading our Sequential model')
            model = load_model(self.prediction_pipeline_paths.model_path)

            logger.info('Predicting the output')
            prediction = model.predict(transformed_querie)
            logger.info(f'Raw model prediction: {prediction}')

            logger.info('Converting prediction into output category')
            output = sentiment_class_returner(prediction)
            logger.info(f'Final sentiment: {output}')

            logger.info('Prediction completed')

            return output

        except Exception as e:
            logger.error(f"Error in predict_text: {str(e)}")
            raise CustomException(e, sys)