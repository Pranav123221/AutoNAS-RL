
import os
from pathlib import Path

# Store Keras dataset cache inside the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KERAS_DIR = DATA_DIR / ".keras"

KERAS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["KERAS_HOME"] = str(KERAS_DIR)

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def load_fashion_mnist(validation_size=0.10, random_state=42):
    """
    Load and preprocess Fashion-MNIST.

    Returns:
        Training, validation and test datasets
        in both CNN and flattened formats.
    """

    # Load original dataset
    (X_full, y_full), (X_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    # Normalize pixel values to [0, 1]
    X_full = X_full.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Add channel dimension for CNN
    X_full = X_full[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Stratified train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_full
    )

    # Flatten images for traditional ML models
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    return {
        "X_train_cnn": X_train,
        "X_val_cnn": X_val,
        "X_test_cnn": X_test,

        "X_train_flat": X_train_flat,
        "X_val_flat": X_val_flat,
        "X_test_flat": X_test_flat,

        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,

        "class_names": CLASS_NAMES
    }
