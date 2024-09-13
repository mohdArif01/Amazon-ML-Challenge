# cnn_model.py
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import re
import constants

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

def extract_number_unit(value):
    try:
        number, unit = utils.parse_string(value)
        return number, unit
    except ValueError:
        return None, None

def load_data(csv_file, image_folder):
    df = pd.read_csv(csv_file)
    images = []
    labels = []
    
    for _, row in df.iterrows():
        image_path = os.path.join(image_folder, os.path.basename(row['image_link']))
        try:
            image = Image.open(image_path).resize(IMAGE_SIZE)
            number, unit = extract_number_unit(row['entity_value'])
            if number is not None and unit is not None:
                images.append(np.array(image) / 255.0)
                labels.append((number, unit))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    return np.array(images), labels

def create_cnn_model(input_shape, num_units):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_units, activation='softmax')  # Number of units
    ])
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_images, train_labels, val_images, val_labels):
    num_units = len(constants.allowed_units)
    model = create_cnn_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_units)
    
    # Prepare labels for training
    train_labels = tf.keras.utils.to_categorical([constants.allowed_units.index(u) for _, u in train_labels], num_classes=num_units)
    val_labels = tf.keras.utils.to_categorical([constants.allowed_units.index(u) for _, u in val_labels], num_classes=num_units)
    
    datagen = ImageDataGenerator(validation_split=0.2)
    
    train_gen = datagen.flow(train_images, train_labels, subset='training', batch_size=BATCH_SIZE)
    val_gen = datagen.flow(val_images, val_labels, subset='validation', batch_size=BATCH_SIZE)
    
    model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)
    
    model.save('cnn_model.h5')

if __name__ == "__main__":
    IMAGE_FOLDER = '../images/'
    CSV_FILE = '../dataset/train.csv'
    
    images, labels = load_data(CSV_FILE, IMAGE_FOLDER)
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    train_model(train_images, train_labels, val_images, val_labels)
