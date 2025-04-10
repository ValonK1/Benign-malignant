''' Using the multi-task UNet model for segmentation and classification '''
import tensorflow as tf
import os
import random
from benign_malieng import unet_model  # import your base UNet model
from tensorflow.keras import layers, models

IMG_HEIGHT = 256
IMG_WIDTH = 256

# Define directories for benign and malignant data
benign_dir = "/Users/valonkrasniqi/Projects/Hassan/DataSet/benign"
malignant_dir = "/Users/valonkrasniqi/Projects/Hassan/DataSet/malignant"

# Function to get image-mask pairs with label (0 for benign, 1 for malignant)
def get_pairs_and_labels(data_dir, label):
    files = sorted([
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.endswith(".png") or fname.endswith(".jpg")
    ])
    pairs = []
    for file in files:
        base_name = os.path.basename(file)
        if "_mask" not in base_name:
            name, ext = os.path.splitext(base_name)
            mask_name = name + "_mask" + ext
            mask_path = os.path.join(data_dir, mask_name)
            if os.path.exists(mask_path):
                pairs.append((file, mask_path, label))
            else:
                print("Warning: No mask found for", file)
    return pairs

# Get pairs from benign (label 0) and malignant (label 1)
benign_pairs = get_pairs_and_labels(benign_dir, 0)
malignant_pairs = get_pairs_and_labels(malignant_dir, 1)

# Combine and shuffle all pairs
all_pairs = benign_pairs + malignant_pairs
random.shuffle(all_pairs)

# Split into training (70%) and testing (30%)
split_idx = int(0.7 * len(all_pairs))
train_pairs = all_pairs[:split_idx]
test_pairs = all_pairs[split_idx:]

# Function to load and preprocess image, mask, and label
def load_image_mask_label(image_path, mask_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH])
    mask = tf.cast(mask, tf.float32) / 255.0

    label = tf.cast(label, dtype=tf.float32)
    return image, (mask, label)

# Prepare training dataset
train_image_paths, train_mask_paths, train_labels = zip(*train_pairs)
train_dataset = tf.data.Dataset.from_tensor_slices((list(train_image_paths), list(train_mask_paths), list(train_labels)))
train_dataset = train_dataset.map(load_image_mask_label, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(8).prefetch(tf.data.AUTOTUNE)

# Prepare testing dataset
test_image_paths, test_mask_paths, test_labels = zip(*test_pairs)
test_dataset = tf.data.Dataset.from_tensor_slices((list(test_image_paths), list(test_mask_paths), list(test_labels)))
test_dataset = test_dataset.map(load_image_mask_label, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(8).prefetch(tf.data.AUTOTUNE)

# Build a multi-task model by adding a classification branch to the base UNet
base_model = unet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))

# Assume the base UNet's output is the segmentation mask.
# For the classification branch, we tap into an intermediate feature map.
# Here we use a GlobalAveragePooling2D on the second-to-last layer of the base model.
encoder_output = base_model.layers[-2].output

# Classification branch
x = layers.GlobalAveragePooling2D()(encoder_output)
x = layers.Dense(64, activation='relu')(x)
classification_output = layers.Dense(1, activation='sigmoid', name='class_output')(x)

# Create a new model with two outputs: segmentation and classification
multi_task_model = models.Model(inputs=base_model.input, outputs=[base_model.output, classification_output])

# Compile the model with losses for segmentation and classification
multi_task_model.compile(optimizer='adam',
                         loss=['binary_crossentropy', 'binary_crossentropy'],
                         loss_weights=[1.0, 0.5],
                         metrics=['accuracy', 'accuracy'])

# Train the model on the training dataset and validate on the testing dataset
multi_task_model.fit(train_dataset, epochs=1, validation_data=test_dataset)

# Test the model on a random sample from the test dataset and display results
import matplotlib.pyplot as plt
import numpy as np
import random

# Select a random test sample
idx = random.randint(0, len(test_image_paths) - 1)
sample_image_path = test_image_paths[idx]
sample_mask_path = test_mask_paths[idx]
sample_label = test_labels[idx]

# Load and preprocess the selected image
image = tf.io.read_file(sample_image_path)
image = tf.image.decode_png(image, channels=1)
image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
image = tf.cast(image, tf.float32) / 255.0
image_np = image.numpy()

# Load and preprocess the corresponding mask
mask = tf.io.read_file(sample_mask_path)
mask = tf.image.decode_png(mask, channels=1)
mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH])
mask = tf.cast(mask, tf.float32) / 255.0
mask_np = mask.numpy()

# Add a batch dimension to the image
image_batch = tf.expand_dims(image, axis=0)

# Get predictions from the model
pred_mask, pred_class = multi_task_model.predict(image_batch)
pred_mask = pred_mask[0]  # remove batch dimension
pred_class = pred_class[0]  # remove batch dimension

# Determine the classification result (using 0.5 threshold)
pred_class_label = 1 if pred_class >= 0.5 else 0

# Display the results
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image_np[:,:,0], cmap='gray')
plt.title("Input Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(mask_np[:,:,0], cmap='gray')
plt.title("True Segmentation Mask")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(pred_mask[:,:,0], cmap='gray')
plt.title(f"Predicted Segmentation Mask\nClassification: {pred_class_label}")
plt.axis('off')

plt.show()