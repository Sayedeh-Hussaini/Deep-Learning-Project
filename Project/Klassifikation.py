import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("CUDA version: ", tf.__version__)
print(tf.config.list_physical_devices('GPU'))
# --- 1. Data Preparation ---
image_size = (299, 299)  # Keeping original size
num_classes = 4  # Example: 4 classes

# Paths to images & labels
train_image_dir = "datasets/train/images"
train_label_dir = "datasets/train/labels"

valid_image_dir = "datasets/valid/images"
valid_label_dir = "datasets/valid/labels"

# Get image file paths
train_image_files = sorted([os.path.join(train_image_dir, f) for f in os.listdir(train_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
valid_image_files = sorted([os.path.join(valid_image_dir, f) for f in os.listdir(valid_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

def load_class_id(txt_file):
    if not os.path.exists(txt_file) or os.stat(txt_file).st_size == 0:
        return None  # No class label if the file doesn't exist or is empty
    
    labels = np.loadtxt(txt_file)
    if labels.size == 0:
        return None  # Return None if the file is empty
    
    labels = labels.reshape(-1, 5)  # YOLO format: (class_id, x_center, y_center, width, height)
    class_id = int(labels[0, 0])  # Taking the first object's class
    return class_id

# Load images and labels
train_dataset = []
for img_path in train_image_files:
    txt_path = os.path.join(train_label_dir, os.path.basename(img_path).replace(".jpg", ".txt").replace(".png", ".txt"))
    
    # Load and preprocess image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize pixels (0-1)
    
    # Load class ID
    class_id = load_class_id(txt_path)
    if class_id is not None:
        train_dataset.append((img, class_id))

valid_dataset = []
for img_path in valid_image_files:
    txt_path = os.path.join(valid_label_dir, os.path.basename(img_path).replace(".jpg", ".txt").replace(".png", ".txt"))
    
    # Load and preprocess image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize pixels (0-1)
    
    # Load class ID
    class_id = load_class_id(txt_path)
    if class_id is not None:
        valid_dataset.append((img, class_id))

# Convert to TensorFlow dataset
batch_size = 16#8#32

train_tf_dataset = tf.data.Dataset.from_generator(
    lambda: train_dataset,
    output_signature=(
        tf.TensorSpec(shape=(299, 299, 3), dtype=tf.float32),  # Image
        tf.TensorSpec(shape=(), dtype=tf.int32)  # Class label
    )
).shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

valid_tf_dataset = tf.data.Dataset.from_generator(
    lambda: valid_dataset,
    output_signature=(
        tf.TensorSpec(shape=(299, 299, 3), dtype=tf.float32),  # Image
        tf.TensorSpec(shape=(), dtype=tf.int32)  # Class label
    )
).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# --- 2. Define Classification Model ---
def build_classification_model(input_shape=(299, 299, 3), num_classes=4):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2,2)(x)
    Dropout(0.5) # Adding Dropout layer
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2,2)(x)
    Dropout(0.25) # Adding Dropout layer
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Flatten()(x)
    Dropout(0.5) # Adding Dropout layer
    x = layers.Dense(256, activation='relu')(x)
    Dropout(0.5) # Adding Dropout layer
    class_output = layers.Dense(num_classes, activation='softmax', name='class')(x)
    
    model = Model(inputs=inputs, outputs=class_output)
    return model

# --- 3. Compile Model ---
classification_model = build_classification_model(input_shape=(299, 299, 3), num_classes=num_classes)
classification_model.compile(
    optimizer='adam',
    loss="sparse_categorical_crossentropy",  # Classification loss
    metrics=["accuracy"]  # Monitor accuracy
)

# --- 4. Train the Model ---

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=10,
    verbose=1,
    min_lr=1e-6
)

epochs = 50  # Adjust based on your dataset size and resources
classification_model.fit(train_tf_dataset, validation_data=valid_tf_dataset, epochs=epochs, 
                         callbacks=[early_stopping, reduce_lr])

# --- 5. Summary of Model ---
classification_model.summary()


# Learning curve
history = classification_model.history.history
print(history)

path_save = 'datasets/results'


plt.plot(history['val_loss'], 'k--', label='validation loss')
plt.plot(history['loss'], label='training loss', color='k')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Curve')
plt.savefig(os.path.join(path_save, 'LearningCurve-Loss.png'))
plt.clf()

plt.plot(history['val_accuracy'], 'r--', label='validation accuracy')
plt.plot(history['accuracy'], label='training accuracy', color='r')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Learning Curve')
plt.savefig(os.path.join(path_save, 'LearningCurve-Accuracy.png'))