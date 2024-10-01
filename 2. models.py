import tensorflow as tf
from tensorflow.keras import layers, models

def nested_unet(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Define the architecture (simplified example)
    conv1 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    conv1 = layers.ReLU()(conv1)
    
    # Decoder
    upsample1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(upsample1)
    
    model = models.Model(inputs, outputs)
    return model

def attention_unet(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Similar architecture with attention layers
    conv1 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    conv1 = layers.ReLU()(conv1)

    # Add attention mechanisms here...

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv1)
    
    model = models.Model(inputs, outputs)
    return model
