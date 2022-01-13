from sklearn.datasets import make_circles
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import itertools


n = 1000

X, y = make_circles(n_samples=n, noise=0.03, random_state=42)

# circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label" : y})

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

X_train = X[:800]
X_test = X[800:]
y_train = y[:800]
y_test = y[800:]

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:,0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:,1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    x_in = np.c_[xx.ravel(), yy.ravel()]
    
    y_pred = model.predict(x_in)
    
    if len(y_pred[0]) > 1:
        print("Doing Multiclass Classification")
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("Doing Binary Classification")
        y_pred = np.round(y_pred).reshape(xx.shape)
    
    plt.contourf(xx,yy,y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:,0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
    ])

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=["accuracy"],)

history = model.fit(X_train, y_train, epochs=40, callbacks=[lr_scheduler])

model.evaluate(X_test, y_test)

# lrs = 1e-4 * (10 ** (tf.range(40)/20))
# plt.figure(figsize=(10,7))
# plt.semilogx(lrs, history.history["loss"])
# plt.xlabel("Learning Rate")
# plt.ylabel("Loss")

y_preds =  model.predict(X_test)

print(confusion_matrix(y_test, tf.round(y_preds)))

figsize=(10, 10)

cm = confusion_matrix(y_test, tf.round(y_preds))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
n_classes = cm.shape[0]

fig, ax = plt.subplots(figsize=figsize)

cax = ax.matshow(cm, cmap=plt.cm.Blues)
fig.colorbar(cax)

classes = False

if classes:
    labels = classes
else:
    labels = np.arange(cm.shape[0])

ax.set(title="Confusion Matrix",
       xlabel="Predicted Label",
       ylabel="True Label",
       xticks=np.arange(n_classes),
       yticks=np.arange(n_classes),
       xticklabels=labels,
       yticklabels=labels)

threshhold = (cm.max() + cm.min()) /2

for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,i, f"{cm[i,j]} ({cm_norm[i,j]*100:.1f})",
             horizontalalignment="center",
             color="white" if cm[i,j] > threshhold else "black",
             size=15)
# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(model, X=X_train, y=y_train)
# plt.subplot(1,2,2)
# plot_decision_boundary(model, X=X_test, y=y_test)
# pd.DataFrame(history.history).plot()
plt.show()