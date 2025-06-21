import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(" GPU found. Using:", physical_devices[0])
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print(" GPU not found. Training will use CPU.")

# Paths
image_dir = './imagesOrg/1024x768'
metadata_file = './metadata.csv'

# Load metadata with correct dtypes
df = pd.read_csv(metadata_file, dtype={'image_id': str})

X = []
y = []

# Debug metadata
print("Sample metadata:\n", df.head())

# Preprocess images
for idx, row in df.iterrows():
    image_id = row['image_id'].strip()  # Ensure no extra whitespace
    possible_filenames = [
        f"{image_id}.jpg",
        f"{image_id}.jpeg",
        f"{image_id}.png"
    ]

    img_path = None
    for fname in possible_filenames:
        full_path = os.path.join(image_dir, fname)
        if os.path.exists(full_path):
            img_path = full_path
            break

    if img_path:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        X.append(img)
        y.append(row['MOS'])
    else:
        print(f"[Missing] No image found for ID: {image_id}")

# Convert to arrays
X = np.array(X)
y = np.array(y)

print(f" Total images loaded: {len(X)}")

# Stop if nothing was loaded
if len(X) == 0:
    raise ValueError("No images were loaded. Please check the image directory and metadata.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate
loss, mae = model.evaluate(X_test, y_test)
print("Test MAE:", mae)

# Plot predictions
y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True MOS")
plt.ylabel("Predicted MOS")
plt.title("CNN Predicted vs True MOS")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.grid()
plt.show()

# Save model
model.save("model.h5")
