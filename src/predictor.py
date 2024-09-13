# predictor.py
import os
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import constants
from utils import parse_string, download_image
from pathlib import Path

IMAGE_SIZE = (128, 128)

def preprocess_image(image_path):
    image = Image.open(image_path).resize(IMAGE_SIZE)
    return np.array(image) / 255.0

def load_cnn_model():
    return load_model('cnn_model.h5')

def predict_entity(model, image_path):
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return constants.allowed_units[predicted_class]  # Convert class index to unit

def predictor(image_link, category_id, entity_name):
    # Define the folder where images are stored
    image_folder = '../images/'
    
    # Ensure the image is downloaded
    download_image(image_link, save_folder=image_folder)
    
    # Load the downloaded image
    image_path = os.path.join(image_folder, Path(image_link).name)
    
    # Load model
    model = load_cnn_model()
    
    # Predict entity value
    predicted_unit = predict_entity(model, image_path)
    
    return f"{predicted_unit}" if predicted_unit else ""

if __name__ == "__main__":
    # Dataset paths
    DATASET_FOLDER = '../dataset/'
    
    # Read the test file
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # Download images for the test set
    image_links = test['image_link'].tolist()
    download_images(image_links, '../images/')
    
    # Generate predictions for each row
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    # Save the output in the required format
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)
