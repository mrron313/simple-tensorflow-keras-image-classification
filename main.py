

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

def pr(str):
    print(str)

fashion_mnist_dataset = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist_dataset.load_data()

pr("train images shape")
pr(train_images.shape)

pr("test images shape")
pr(test_images.shape)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

prediction = model.predict(test_images)

pr("predicted output class label of first image: ")
pr(np.argmax(prediction[0]))

pr("actual class label of first image: ")
pr(test_labels[0])