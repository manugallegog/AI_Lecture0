# Import the required libraries
import numpy as np

# Import the keras and pytorch necessities
import keras
import torch


# Data preparation for the Fashion MNIST dataset
def get_dataset():
    """
    Load and preprocess the Fashion MNIST dataset.

    Returns:
        train_dataset (torch.utils.data.TensorDataset): TensorDataset containing the training data.
        val_dataset (torch.utils.data.TensorDataset): TensorDataset containing the validation data.
    """

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()

    # Data preprocessing steps
    x_train = x_train.astype("float32")
    x_val = x_val.astype("float32")

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_val = np.expand_dims(x_val, -1)

    # Category labels for the dataset
    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)

    # Create a TensorDataset for the training and validation sets using torch utils
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_val), torch.from_numpy(y_val)
    )

    # Return the datasets for training and validation
    return train_dataset, val_dataset
