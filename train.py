import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model
from data_processing import load_data
from config import BATCH_SIZE, EPOCHS, NUM_CLASSES

def train_model():
    # Load and preprocess data
    images, labels = load_data()

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Update NUM_CLASSES in config
    NUM_CLASSES = len(label_encoder.classes_)

    # Normalize images
    X = np.array(images) / 255.0
    X = X.reshape(-1, X.shape[1], X.shape[2], 1)
    y = np.array(encoded_labels)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(X_train)

    # Create model
    model = create_model()

    # Train model
    history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        epochs=EPOCHS,
                        validation_data=(X_val, y_val))

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    # Save model and label encoder
    model.save('handwriting_model.h5')
    np.save('classes.npy', label_encoder.classes_)
