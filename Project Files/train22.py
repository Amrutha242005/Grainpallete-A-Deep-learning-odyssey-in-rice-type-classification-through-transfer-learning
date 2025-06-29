import pathlib
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import plotly.offline as iplot
import plotly.express as px
import pandas as pd
import tensorflow_hub as hub
import plotly
import matplotlib.pyplot as plt

data_dir = pathlib.Path('Rice_Image_Dataset')

# Limiting to 600 images per class for demonstration.
# For a more robust model, consider using the full dataset.
arborio = list(data_dir.glob('Arborio/*'))[:600]
basmati = list(data_dir.glob('Basmati/*'))[:600]
ipsala = list(data_dir.glob('Ipsala/*'))[:600]
jasmine = list(data_dir.glob('Jasmine/*'))[:600]
karacadag = list(data_dir.glob('Karacadag/*'))[:600]

# Display example images
fig, ax = plt.subplots(ncols=5, figsize=(20,5))
fig.suptitle('Rice Category Examples')

# Read and display the first image of each category
arborio_image = cv2.imread(str(arborio[0]))
basmati_image = cv2.imread(str(basmati[0]))
ipsala_image = cv2.imread(str(ipsala[0]))
jasmine_image = cv2.imread(str(jasmine[0]))
karacadag_image = cv2.imread(str(karacadag[0]))

ax[0].set_title('Arborio')
ax[1].set_title('Basmati')
ax[2].set_title('Ipsala')
ax[3].set_title('Jasmine')
ax[4].set_title('Karacadag')

# OpenCV reads images in BGR; Matplotlib expects RGB. Convert before displaying.
ax[0].imshow(cv2.cvtColor(arborio_image, cv2.COLOR_BGR2RGB))
ax[1].imshow(cv2.cvtColor(basmati_image, cv2.COLOR_BGR2RGB))
ax[2].imshow(cv2.cvtColor(ipsala_image, cv2.COLOR_BGR2RGB))
ax[3].imshow(cv2.cvtColor(jasmine_image, cv2.COLOR_BGR2RGB))
ax[4].imshow(cv2.cvtColor(karacadag_image, cv2.COLOR_BGR2RGB))

plt.show()

# Prepare image paths and labels
df_images = {
    'arborio': arborio,
    'basmati': basmati,
    'ipsala': ipsala,
    'jasmine': jasmine,
    'karacadag': karacadag
}

df_labels = {
    'arborio': 0,
    'basmati': 1,
    'ipsala': 2,
    'jasmine': 3,
    'karacadag': 4
}

x, y = [], []
for label, images in df_images.items():
    for image in images:
        img = cv2.imread(str(image))
        # Resize images to 224x224, as required by MobileNetV2
        resized_img = cv2.resize(img, (224, 224))
        x.append(resized_img)
        y.append(df_labels[label])

# Convert lists to numpy arrays
x = np.array(x)
# Normalize pixel values to be between 0 and 1
x = x / 255.0
y = np.array(y)

# Split data into training, validation, and test sets
# First split: train and (test+validation)
x_train, x_test_val, y_train, y_test_val = train_test_split(x, y, test_size=0.3, random_state=42) # 70% train, 30% test_val
# Second split: test and validation from the 30%
x_test, x_val, y_test, y_val = train_test_split(x_test_val, y_test_val, test_size=0.5, random_state=42) # 15% test, 15% val

print(f"Training data shape: {x_train.shape}")
print(f"Validation data shape: {x_val.shape}")
print(f"Test data shape: {x_test.shape}")


# Load the MobileNetV2 feature extractor from TensorFlow Hub
mobile_net_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
mobile_net = hub.KerasLayer(
    mobile_net_url,
    input_shape=(224, 224, 3), # Input shape for MobileNetV2
    trainable=False # Freeze the pre-trained layers
)

# Define the number of output classes
num_label = 5

# Build the Keras Sequential model
model = tf.keras.Sequential([
    mobile_net,
    # Add a Dense layer for classification with softmax activation
    # Softmax ensures the output is a probability distribution over the classes
    tf.keras.layers.Dense(num_label, activation='softmax')
])

# Compile the model
# Use Adam optimizer for efficient training
# Use SparseCategoricalCrossentropy as loss function because labels are integers (not one-hot encoded)
# Monitor accuracy during training
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print a summary of the model architecture
model.summary()

# Train the model
print("Starting model training...")
history = model.fit(
    x_train,
    y_train,
    epochs=10, # Number of epochs to train for
    validation_data=(x_val, y_val) # Data to evaluate the model on after each epoch
)
print("Model training finished.")

# Save the trained model
model.save("rice.h5")
print("Model saved as rice.h5")

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


