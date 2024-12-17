import zipfile
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

# Unzip the provided file
def extract_file(file_path):
    with zipfile.ZipFile(file_path, 'r') as zipped:
        zipped.extractall('./')

# Extract the CIFAR-3 class data
extract_file('./cifar-3class-data.zip')

# Load image data and respective labels
def load_data(path):
    images = []
    labels = []
    for class_ in os.listdir(path):
        class_path = os.path.join(path, class_)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).resize((32, 32))
            img_array = np.array(img) / 255.0  # Normalize directly
            images.append(img_array)
            labels.append(int(class_[-1]))  # Class index based on folder name
    return np.array(images), np.array(labels)

# Set paths and load data
TRAIN_PATH = './cifar-3class-data/train/'
TEST_PATH = './cifar-3class-data/test/'
train, ytrain = load_data(TRAIN_PATH)
test, ytest = load_data(TEST_PATH)

# Plot any 4 images
plt.figure(figsize=(6, 6))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(train[i])
    plt.axis('off')
plt.suptitle("Sample Training Images")
plt.show()

# Split into train and validation sets
train, validation, ytrain, y_valid = train_test_split(train, ytrain, test_size=0.1, random_state=42)
print(f"Training set size: {len(train)}, Validation set size: {len(validation)}")

# Flatten images for FCNN
def flatten_images(images):
    return images.reshape(images.shape[0], -1)  # Shape: (num_samples, 3072)

# Prepare flattened data for FCNN
train_flattened = flatten_images(train)
valid_flattened = flatten_images(validation)
test_flattened = flatten_images(test)

# FCNN Model Definition
def build_fcnn(input_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # 3 output classes
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Training the FCNN model
fcnn_model = build_fcnn((32 * 32 * 3,))
fcnn_history = fcnn_model.fit(train_flattened, ytrain, validation_data=(valid_flattened, y_valid),
                              epochs=500, batch_size=200)

# Plot training & validation accuracy for FCNN
def plot_accuracy(history, epochs):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('FCNN Training and Validation Accuracy')
    plt.show()

plot_accuracy(fcnn_history, 500)

# FCNN Model Evaluation
fcnn_test_loss, fcnn_test_accuracy = fcnn_model.evaluate(test_flattened, ytest)
print(f"FCNN Test Loss: {fcnn_test_loss:.4f}, FCNN Test Accuracy: {fcnn_test_accuracy * 100:.2f}%")

# Predictions with FCNN
fcnn_predictions = np.argmax(fcnn_model.predict(test_flattened), axis=1)
print(f"FCNN Model Test Accuracy (Manual): {np.mean(fcnn_predictions == ytest) * 100:.2f}%")

# Confusion Matrix for FCNN
fcnn_cm = confusion_matrix(ytest, fcnn_predictions)
ConfusionMatrixDisplay(fcnn_cm, display_labels=['Plane', 'Car', 'Bird']).plot()
plt.title("FCNN Confusion Matrix")
plt.show()

# CNN Model Definition
def build_cnn(input_shape=(32, 32, 3)):
    model = Sequential([
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2), padding='valid'),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2), padding='valid'),

        Flatten(),
        Dense(512, activation='relu'),
        Dense(100, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Training the CNN model
cnn_model = build_cnn()
cnn_history = cnn_model.fit(train, ytrain, validation_data=(validation, y_valid), epochs=50, batch_size=200)

# Plot training & validation accuracy for CNN
plot_accuracy(cnn_history, 50)

# CNN Model Evaluation
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(test, ytest)
print(f"CNN Test Loss: {cnn_test_loss:.4f}, CNN Test Accuracy: {cnn_test_accuracy * 100:.2f}%")

# Predictions with CNN
cnn_predictions = np.argmax(cnn_model.predict(test), axis=1)

# Confusion Matrix for CNN
cnn_cm = confusion_matrix(ytest, cnn_predictions)
ConfusionMatrixDisplay(cnn_cm, display_labels=['Plane', 'Car', 'Bird']).plot()
plt.title("CNN Confusion Matrix")
plt.show()


