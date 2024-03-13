import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt


# Function to load images from a folder
def load_images(folder_path, target_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, target_size)
                images.append(image)
                labels.append(1 if "parasitized" in folder_path.lower() else 0)  # Corrected label assignment
        except Exception as e:
            print(f"Error loading image: {img_path} - {e}")
    return images, labels


# Loding images from the infected and uninfected folders
infected_path = "C:/Users/MAK ROG/Downloads/cells/cell_images/Parasitized"
uninfected_path = "C:/Users/MAK ROG/Downloads/cells/cell_images/Uninfected"

infected_images, infected_labels = load_images(infected_path)
uninfected_images, uninfected_labels = load_images(uninfected_path)

infected_images = np.array(infected_images)
uninfected_images = np.array(uninfected_images)
infected_labels = np.array(infected_labels)
uninfected_labels = np.array(uninfected_labels)

# Concatenate infected and uninfected images and labels
X = np.concatenate((infected_images, uninfected_images), axis=0)
y = np.concatenate((infected_labels, uninfected_labels), axis=0)

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# pixel values to a range between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0


# Data Augmentation 
datagen = ImageDataGenerator(
    rotation_range=10,  
    width_shift_range=0.1,
    height_shift_range=0.1, 
    shear_range=0.1,  
    zoom_range=0.1,  
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Prepare the iterator
augmented_data_iterator = datagen.flow(X_train, y_train, batch_size=32)

 # Visualizing some preprocessed images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel("Infected" if y_train[i] == 1 else "Uninfected")
plt.show()

# Build the simple CNN model
input_shape = X_train.shape[1:]
simple_cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compiling the models 
simple_cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Build VGG16 model
base_vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
model_vgg = Sequential([
    base_vgg_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model_vgg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Build MobileNetV2 model
base_mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
model_mobilenet = Sequential([
    base_mobilenet_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model_mobilenet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Build ResNet50 model
base_resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
model_resnet = Sequential([
    base_resnet_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model_resnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training all models
history_simple_cnn = simple_cnn_model.fit(augmented_data_iterator, epochs=12, validation_data=(X_test, y_test))
print("History Simple CNN:", history_simple_cnn.history)
# history_vgg = model_vgg.fit(augmented_data_iterator, epochs=1, validation_data=(X_test, y_test))
# print("History VGG:", history_vgg.history)
# history_mobilenet = model_mobilenet.fit(augmented_data_iterator, epochs=1, validation_data=(X_test, y_test))
# print("History MobileNetV2:", history_mobilenet.history)
# # history_resnet = model_resnet.fit(augmented_data_iterator, epochs=3, validation_data=(X_test, y_test))

# Saving the best model based on validation accuracy
best_model = None
best_val_accuracy = 0.0

# saving the best model for Simple CNN
if history_simple_cnn.history['val_accuracy'][-1] > best_val_accuracy:
    best_model = simple_cnn_model
    best_val_accuracy = history_simple_cnn.history['val_accuracy'][-1]
    best_model.save("best_model_simple_cnn.h5")

# # Check and save the best model for VGG16
# if history_vgg.history['val_accuracy'][-1] > best_val_accuracy:
#     best_model = model_vgg
#     best_val_accuracy = history_vgg.history['val_accuracy'][-1]
#     best_model.save("best_model_vgg16.h5")

# # Check and save the best model for MobileNetV2
# if history_mobilenet.history['val_accuracy'][-1] > best_val_accuracy:
#     best_model = model_mobilenet
#     best_val_accuracy = history_mobilenet.history['val_accuracy'][-1]
#     best_model.save("best_model_mobilenetv2.h5")

# # Check and save the best model for ResNet50
# if history_resnet.history['val_accuracy'][-1] > best_val_accuracy:
#     best_model = model_resnet
#     best_val_accuracy = history_resnet.history['val_accuracy'][-1]
#     best_model.save("best_model_resnet50.h5")
import matplotlib.pyplot as plt
   
# Ploting the training and validation accuracy for all models
plt.plot(history_simple_cnn.history['accuracy'], label='Simple CNN Training Accuracy')
plt.plot(history_simple_cnn.history['val_accuracy'], label='Simple CNN Validation Accuracy')
# plt.plot(history_vgg.history['accuracy'], label='VGG16 Training Accuracy')
# plt.plot(history_vgg.history['val_accuracy'], label='VGG16 Validation Accuracy')
# plt.plot(history_mobilenet.history['accuracy'], label='MobileNetV2 Training Accuracy')
# plt.plot(history_mobilenet.history['val_accuracy'], label='MobileNetV2 Validation Accuracy')
# # plt.plot(history_resnet.history['accuracy'], label='ResNet50 Training Accuracy')
# # plt.plot(history_resnet.history['val_accuracy'], label='ResNet50 Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy Comparison')
plt.show()

# Plot the training and validation loss for all models
plt.plot(history_simple_cnn.history['loss'], label='Simple CNN Training Loss')
plt.plot(history_simple_cnn.history['val_loss'], label='Simple CNN Validation Loss')
# plt.plot(history_vgg.history['loss'], label='VGG16 Training Loss')
# plt.plot(history_vgg.history['val_loss'], label='VGG16 Validation Loss')
# plt.plot(history_mobilenet.history['loss'], label='MobileNetV2 Training Loss')
# plt.plot(history_mobilenet.history['val_loss'], label='MobileNetV2 Validation Loss')
# # plt.plot(history_resnet.history['loss'], label='ResNet50 Training Loss')
# # plt.plot(history_resnet.history['val_loss'], label='ResNet50 Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Comparison')
plt.show()

# import matplotlib.pyplot as plt

# # History of Simple CNN
# history_simple_cnn = {'loss': [0.418170303106308], 'accuracy': [0.8033657073974609], 
#                       'val_loss': [0.1819852739572525], 'val_accuracy': [0.9501088261604309]}

# # History of VGG16
# history_vgg = {'loss': [0.7197151184082031], 'accuracy': [0.5174181461334229], 
#                'val_loss': [0.6932933330535889], 'val_accuracy': [0.4925616979598999]}

# # History of MobileNetV2
# history_mobilenet = {'loss': [0.2800233066082001], 'accuracy': [0.906150758266449], 
#                      'val_loss': [3.2132296562194824], 'val_accuracy': [0.6088533997535706]}
