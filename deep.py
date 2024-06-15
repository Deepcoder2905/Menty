import cv2
import numpy as np
import os
import sys
import warnings

# Suppress libpng warnings
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return None
    try:
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0
    except cv2.error as e:
        print(f"Error processing image {image_path}: {e}")
        return None
    return image

def load_dataset(directory):
    images = []
    labels = []
    label_map = {
        "acne face": 0,
        "pigmentation skin": 1,
        "redness face": 2,
        "sagging skin": 3,
        "hydration skin": 4,
        "skin translucency": 5,
        "skin uniformness": 6,
        "wrinkles skin": 7,
        "skin pores": 8,
        "oily skin": 9
    }
    
    for label, idx in label_map.items():
        folder_path = os.path.join(directory, label)
        if not os.path.exists(folder_path):
            print(f"Warning: Directory {folder_path} does not exist.")
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                file_path = os.path.join(folder_path, filename)
                image = preprocess_image(file_path)
                if image is not None:  # Only append if image is successfully processed
                    images.append(image)
                    labels.append(idx)
                else:
                    print(f"Warning: Failed to process image {file_path}")
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Set the terminal encoding to UTF-8 to handle Unicode characters properly
sys.stdout.reconfigure(encoding='utf-8')

# Load the dataset
dataset_directory = 'dataset'
X, y = load_dataset(dataset_directory)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# One-hot encode the labels
y = to_categorical(y, num_classes=10)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),  # Add dropout to reduce overfitting
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),  # Add dropout to reduce overfitting
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Add dropout to reduce overfitting
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
input_shape = (224, 224, 3)
num_classes = 10
model = create_cnn_model(input_shape, num_classes)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.3)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")

# Save the model
model.save('my_model.h5')






