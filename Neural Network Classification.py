from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

n = 1000

X, y = make_circles(n_samples=n, noise=0.03, random_state=42)

circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label" : y})

print(circles.head())

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1)
    ])

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["accuracy"])

model.fit(X, y, epochs=200)
