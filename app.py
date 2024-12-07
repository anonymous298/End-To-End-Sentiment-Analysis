from flask import Flask, render_template, request
from src.pipe.prediction_pipeline import PredicitonPipeline, CustomException
from src.logger import get_logger
import traceback
import nltk

app = Flask(__name__)
logger = get_logger('Flask-App')

# Download required NLTK data
try:
    logger.info("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    logger.info("NLTK data downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK data: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get text from form
            text = request.form.get('text')
            logger.info(f"Received text for prediction: {text}")
            
            if not text:
                logger.error("No text received in form")
                return render_template('predict.html', error="Please enter some text to analyze")
            
            # Initialize prediction pipeline
            try:
                pipeline = PredicitonPipeline()
            except CustomException as e:
                logger.error(f"Failed to initialize prediction pipeline: {str(e)}")
                return render_template('predict.html', error="Model files not found. Please ensure the model is properly set up.")
            
            # Get prediction
            logger.info("Starting prediction")
            prediction = pipeline.predict_text(text)
            logger.info(f"Prediction result: {prediction}")
            
            # Render template with prediction
            return render_template('predict.html', prediction=prediction, input_text=text)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(traceback.format_exc())
            error_message = "An error occurred during prediction. Please try again."
            return render_template('predict.html', error=error_message, input_text=text)
    
    # If GET request, just show the form
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)