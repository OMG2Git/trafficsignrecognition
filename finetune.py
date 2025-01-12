from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import cv2
import os

# Paths to your dataset
train_dir = "Path_To_Dataset_Train_directory"
test_dir = "Path_To_Dataset_Test_directory"

# Parameters
batch_size = 32
img_height, img_width = 64, 64
num_classes = len(os.listdir(train_dir))

# Custom Augmentation Functions
def add_motion_blur(image):
    ksize = np.random.choice([3, 5, 7])
    kernel = np.zeros((ksize, ksize))
    kernel[(ksize - 1) // 2, :] = np.ones(ksize) / ksize
    return cv2.filter2D(image, -1, kernel)

def add_shadow(image):
    h, w, _ = image.shape
    top_x, bottom_x = np.random.randint(0, w, 2)
    top_y, bottom_y = 0, h
    overlay = image.copy()
    alpha = np.random.uniform(0.4, 0.6)
    cv2.rectangle(overlay, (top_x, top_y), (bottom_x, bottom_y), (0, 0, 0), -1)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def custom_preprocessing(image):
    image = add_motion_blur(image)
    image = add_shadow(image)
    return image

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    preprocessing_function=custom_preprocessing
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

# Load the pre-trained model
base_model = load_model("train.h5")

inputs = tf.keras.Input(shape=(img_height, img_width, 3), name='input_layer_finetune')
x = base_model(inputs)  # Get the output of the base model

x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

# Create the new model
model = Model(inputs=inputs, outputs=output)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ModelCheckpoint("finetuned_model_best.keras", monitor="val_accuracy", save_best_only=True),
    EarlyStopping(monitor="val_loss", patience=5, verbose=1)
]

# Re-train the model
epochs = 30
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
    callbacks=callbacks
)

# Save the final fine-tuned model
model.save("finetuned_train_v2.keras")
print("Fine-tuned model saved as 'finetuned_train_v2.keras'")

# Evaluate the fine-tuned model
loss, accuracy = model.evaluate(test_generator)
print(f"Fine-tuned Test Accuracy: {accuracy * 100:.2f}%")

converter = tf.lite.TFLiteConverter.from_saved_model("finetuned_train_v2.keras")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

print("Quantized model saved as 'model_quantized.tflite'")