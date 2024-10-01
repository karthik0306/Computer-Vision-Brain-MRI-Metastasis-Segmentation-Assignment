import numpy as np
from data_preprocessing import load_data, preprocess_data
from models import nested_unet, attention_unet

image_dir = "path/to/images"
mask_dir = "path/to/masks"

images, masks = load_data(image_dir, mask_dir)
X_train, X_test, y_train, y_test = preprocess_data(images, masks)

model_nested = nested_unet((128, 128, 1))  # Adjust input shape as necessary
model_attention = attention_unet((128, 128, 1))

model_nested.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_attention.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_nested = model_nested.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)
history_attention = model_attention.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)

# Save models
model_nested.save("nested_unet_model.h5")
model_attention.save("attention_unet_model.h5")
