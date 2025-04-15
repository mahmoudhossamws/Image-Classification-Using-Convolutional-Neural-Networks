import tensorflow as tf
import keras
import os
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Define the path to your dataset
dataset_dir = "train"
dataset_test_dir = "test"
# Initialize lists to store images and labels
images = []
labels = []
test_images = []
test_labels = []
# Assign a unique label to each folder
label_map = {"buildings": 0, "forest": 1, "glacier": 2, "mountain":3 , "sea":4 , "street":5}  # Example mapping
class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


# Load images and labels for training
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):  # Ensure it's a directory
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = tf.keras.utils.load_img(img_path, target_size=(32, 32))  # Resize images
            img_array = tf.keras.utils.img_to_array(img)  # Convert to NumPy array (shape: HxWx3)
            images.append(img_array)
            labels.append(label_map[class_name])  # Assign label

for class_name in os.listdir(dataset_test_dir):
    class_dir = os.path.join(dataset_test_dir, class_name)
    if os.path.isdir(class_dir):  # Ensure it's a directory
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = tf.keras.utils.load_img(img_path, target_size=(32, 32))  # Resize images
            img_array = tf.keras.utils.img_to_array(img)  # Convert to NumPy array (shape: HxWx3)
            test_images.append(img_array)
            test_labels.append(label_map[class_name])  # Assign label
# Convert to NumPy arrays
images = np.array(images)
labels = np.array(labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Normalize images (optional)
images = images / 255.0
test_images = test_images / 255.0

# Inspect the data
print("Images shape:", images.shape)  # Example: (num_images, 32, 32, 3) -> 3 channels for RGB
print("Labels shape:", labels.shape)  # Example: (num_images,)

# Generate a random permutation of indices
indices = np.arange(len(images))  # Create an array of indices [0, 1, 2, ..., N-1]
np.random.shuffle(indices)        # Shuffle the indices randomly

# Shuffle images and labels using the same indices
images = images[indices]
labels = labels[indices]

print("\nAfter shuffling:")
print("First 5 labels:", labels[:5])  # Verify shuffled labels

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6,activation='softmax'))

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy,
              metrics=['accuracy'])

history = model.fit(images, labels, epochs=7,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("test accuracy: ",test_acc)

predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

IMG_INDEX = 17  # change this to look at other images

plt.imshow(test_images[IMG_INDEX] ,cmap=plt.cm.binary)
plt.xlabel(class_names[predicted_classes[IMG_INDEX]])
plt.show()
print("expected:",class_names[test_labels[IMG_INDEX]],"\n got:",class_names[predicted_classes[IMG_INDEX]])