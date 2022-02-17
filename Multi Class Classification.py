import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

print(f"training sample:\n {train_data[0].shape}")  

# Data Visualizing
# import random
# plt.figure(figsize=(7,7))
# for i in range(4):
#     ax = plt.subplot(2,2,i+1)
#     rand_index = random.choice(range(len(train_data)))
#     plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
#     plt.axis(False)
# plt.show()


# Model building

tf.random.set_seed(42)

train_data = train_data /255
test_data = test_data /255

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(10, activation="sigmoid")
])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

history = model.fit(train_data, tf.one_hot(train_labels, depth=10), epochs=10, validation_data=(test_data, tf.one_hot(test_labels, depth=10)))