import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

# Function to load images from a folder
def load_images_from_folder(folder, target_size=(256, 256)):
    images = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if filename.startswith('.'):
            continue  # Skip hidden files like .DS_Store
        try:
            img = load_img(file_path, target_size=target_size)
            img = img_to_array(img)
            images.append(img)
        except UnidentifiedImageError:
            print(f"UnidentifiedImageError: Cannot identify image file {file_path}")
        except Exception as e:
            print(f"Error: {e} - for file {file_path}")
    return np.array(images)

# Define the DCE-Net model
def dce_net(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv2)
    conv4 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv3)
    conv5 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(24, (3, 3), padding='same', activation='relu')(conv5)
    conv7 = Conv2D(3, (3, 3), padding='same', activation='tanh')(conv6)
    
    enhanced_image = Add()([inputs, conv7])
    
    model = Model(inputs=inputs, outputs=enhanced_image)
    return model


# Function to calculate PSNR
def psnr(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * K.log(max_pixel / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)

# Load training and validation images
train_low_folder = 'train/low'
train_high_folder = 'train/high'

low_images = load_images_from_folder(train_low_folder)
high_images = load_images_from_folder(train_high_folder)

low_images = low_images / 255.0
high_images = high_images / 255.0

x_train, x_val, y_train, y_val = train_test_split(low_images, high_images, test_size=0.2, random_state=42)

# Compile and train the model
input_shape = (256, 256, 3)
model = dce_net(input_shape)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', psnr])

model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val))

# Save the trained model
model.save('dce_denoising_model.h5')

# Predict and save denoised images for the test set
test_low_folder = 'test/low'
prediction_folder = 'test/predicted'

os.makedirs(prediction_folder, exist_ok=True)

low_test_images = load_images_from_folder(test_low_folder)
low_test_images = low_test_images / 255.0

predictions = model.predict(low_test_images)

for i, prediction in enumerate(predictions):
    prediction = (prediction * 255).astype(np.uint8)
    prediction = Image.fromarray(prediction)
    prediction.save(os.path.join(prediction_folder, f'pred_{i}.png'))