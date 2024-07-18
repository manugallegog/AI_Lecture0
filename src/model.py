# Import layers from keras to build our convnet model using the functional API
import keras
from keras import layers


# Create a simple convnet model using keras functional API
def get_model():
    """
    Creates and returns a simple convolutional neural network model.

    Returns:
        model (keras.Model): The created model.
    """
    
    # Make a simple convnet with batch normalization and dropout.
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.Conv2D(filters=12, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(scale=False, center=True)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters=24,
        kernel_size=6,
        use_bias=False,
        strides=2,
    )(x)
    x = layers.BatchNormalization(scale=False, center=True)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters=32,
        kernel_size=6,
        padding="same",
        strides=2,
        name="large_k",
    )(x)
    x = layers.BatchNormalization(scale=False, center=True)(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    return model
