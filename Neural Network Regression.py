from numpy.core.fromnumeric import ndim
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

plt.scatter(x, y)
#plt.show()

print(y == x +10)

#Check Shapes
# Create features (using tensors)
X = tf.constant(x)

# Create labels (using tensors)
y = tf.constant(y)

#Steps in modeling with Tensorflow

#1 Creating a model
tf.random.set_seed(40)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

model.fit(tf.expand_dims(X, axis=-1), y, epochs=100)

print(model.predict([17.0]))

