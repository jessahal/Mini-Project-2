# %%
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import numpy as np
import matplotlib.pyplot as plt

# %%
images_dir = "training/"
labels = pd.read_csv("training_labels.csv")

# add the directory to the filename
labels['ID'] = labels['ID'].apply(lambda x: os.path.join(images_dir, x))

# Initialize the ImageDataGenerator
# You can change the size of the validation split (0.25 is 25% of data used as validation set)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.25,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create the training and validation generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=labels,
    directory=None,  
    x_col='ID',
    y_col='target',
    target_size=(224, 224), # You can change the size of the image
    batch_size=32, # You can change the batch_size
    class_mode='categorical',  
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=labels,
    directory=None,  
    x_col='ID',
    y_col='target',
    target_size=(224, 224), # Should match training size
    batch_size=32, # Should match training
    class_mode='categorical',  
    subset='validation'
)

# %%
## Plot a few of the images

# Fetch a batch of images and their labels
images, labels = next(train_generator)

# Number of images to show
num_images = 8

plt.figure(figsize=(20, 10))
for i in range(num_images):
    ax = plt.subplot(2, 4, i + 1)
    plt.imshow(images[i])
    # The label for current image
    label_index = labels[i].argmax()  # Convert one-hot encoding to index
    label = list(train_generator.class_indices.keys())[label_index]  # Get label name from index
    plt.title(label)
    plt.axis('off')
plt.show()


# %%
base_mod = ResNet50(weights='imagenet')

# Freeze the base model layers
for layer in base_mod.layers[-10:]:
    layer.trainable = True

# Create the model
model = Sequential([
    base_mod,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels['target'].unique()), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    validation_data=validation_generator, 
    epochs = 10
)

# %%
# Get the number of batches in the validation generator
num_batches = validation_generator.samples // validation_generator.batch_size

# Create a list to store predictions
predictions = []

# Loop through the batches in the validation generator
for i in range(num_batches):
    x_batch, _ = validation_generator.next()  # Get the next batch of images (x) and ignore labels (_)
    preds = model.predict(x_batch)  # Predict on the batch
    predictions.append(preds)

# Concatenate all predictions
predictions = np.concatenate(predictions, axis=0)


