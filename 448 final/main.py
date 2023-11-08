import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras import layers, models
from keras import layers, models
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data by scaling the images to the range of [0, 1]
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Convert the labels to one-hot encoded vectors
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# Build the CNN model
model_cnn = models.Sequential()
model_cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_cnn.add(layers.MaxPooling2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(layers.MaxPooling2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(64, activation='relu'))
model_cnn.add(layers.Dense(10, activation='softmax'))

# Compile the model
model_cnn.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model


# Evaluate the model
test_loss, test_acc = model_cnn.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

train_images_rnn = train_images.reshape((60000, 28, 28))
test_images_rnn = test_images.reshape((10000, 28, 28))

# Build the RNN model
model_rnn = models.Sequential()
model_rnn.add(layers.LSTM(128, input_shape=(28, 28), activation='relu', return_sequences=True))
model_rnn.add(layers.LSTM(128, activation='relu'))
model_rnn.add(layers.Dense(10, activation='softmax'))

# Compile the model
model_rnn.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model


# Evaluate the model
test_loss, test_acc = model_rnn.evaluate(test_images_rnn, test_labels)
print(f"Test accuracy: {test_acc}")



history_cnn = model_cnn.fit(train_images, train_labels, epochs=40, batch_size=64, validation_data=(test_images, test_labels))
predictions_cnn = model_cnn.predict(test_images)
predictions_cnn_classes = np.argmax(predictions_cnn, axis=1)
true_classes_cnn = np.argmax(test_labels, axis=1)
cm_cnn = confusion_matrix(true_classes_cnn, predictions_cnn_classes)

history_rnn = model_rnn.fit(train_images_rnn, train_labels, epochs=40, batch_size=64, validation_data=(test_images_rnn, test_labels))
predictions_rnn = model_rnn.predict(test_images_rnn)
predictions_rnn_classes = np.argmax(predictions_rnn, axis=1)
true_classes_rnn = np.argmax(test_labels, axis=1)
cm_rnn = confusion_matrix(true_classes_rnn, predictions_rnn_classes)


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()


# Function to plot the loss and accuracy charts
def plot_history(history, title):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title(title + ' Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title(title + ' Accuracy')
    plt.legend()

    plt.show()


# Plot the results for CNN
plot_confusion_matrix(cm_cnn, 'CNN Confusion Matrix')
plot_history(history_cnn, 'CNN')

# Plot the results for RNN
plot_confusion_matrix(cm_rnn, 'RNN Confusion Matrix')
plot_history(history_rnn, 'RNN')


predictions = model_cnn.predict(test_images)

# Convert predictions classes to one hot vectors
predicted_classes = np.argmax(predictions, axis=1)
# Convert test images and labels to one hot vectors
true_classes = np.argmax(test_labels, axis=1)

# Select a few images to display
images_to_display = test_images[:30]  # Displaying the first 16 images
true_labels = true_classes[:16]
predicted_labels = predicted_classes[:16]

# Reshape the images from (28, 28, 1) to (28, 28) for displaying
images_to_display = images_to_display.reshape(images_to_display.shape[0], 28, 28)

# Plot the images and labels using matplotlib
plt.figure(figsize=(10, 10))
for i in range(len(images_to_display)):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images_to_display[i], cmap=plt.cm.binary)
    plt.xlabel(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}")
plt.show()
