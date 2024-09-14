import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from utils import download_images
import constants

# Define dataset folder path
DATASET_FOLDER = '/content/Amazon-ML-Challenge/dataset'
IMAGE_SIZE = (100, 100)  # Resize images to a fixed size (can be tuned)

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=IMAGE_SIZE)
    image = img_to_array(image)
    image = image / 255.0  # Normalize the pixel values
    return image

# CNN model definition
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')  # For predicting continuous values like weight or dimensions
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train the model
def train_model(X_train, y_train, X_val, y_val):
    model = build_model()

    # Data augmentation to improve generalization
    datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)
    datagen.fit(X_train)

    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              validation_data=(X_val, y_val),
              epochs=10, verbose=2)
    return model

# Prediction function
def predictor(image_link, category_id, entity_name, model, image_folder):
    image_path = os.path.join(image_folder, os.path.basename(image_link))
    
    # Load and preprocess the image
    if os.path.exists(image_path):
        image = load_and_preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make prediction using the trained model
        prediction = model.predict(image)[0][0]  # Get the prediction
        unit = random.choice(list(constants.entity_unit_map[entity_name]))  # Placeholder for unit
        return f"{prediction:.2f} {unit}"
    else:
        return ""  # Return empty string if the image is not available

# Main function
if __name__ == "__main__":
    # Load training data
    train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    
    # Download images for training
    download_images(train['image_link'], download_folder=DATASET_FOLDER + 'train_images')

    # Load and preprocess images
    image_paths = [os.path.join(DATASET_FOLDER + 'train_images', os.path.basename(link)) for link in train['image_link']]
    images = np.array([load_and_preprocess_image(path) for path in image_paths if os.path.exists(path)])
    labels = np.array([float(value.split()[0]) for value in train['entity_value']])

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Train the CNN model
    model = train_model(X_train, y_train, X_val, y_val)

    # Load test data
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

    # Download images for testing
    download_images(test['image_link'], download_folder=DATASET_FOLDER + 'test_images')

    # Make predictions on test data
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name'], model, DATASET_FOLDER + 'test_images'), axis=1)

    # Output predictions to CSV
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)

    print(f"Predictions saved to {output_filename}")
