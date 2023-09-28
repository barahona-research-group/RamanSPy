"""
Integrative analysis: Neural Network (NN) classification
=========================================================================

In this example, we will showcase `RamanSPy's` integrability by integrating a Neural Network (NN)
model for the identification of different bacteria species.

To build the model, we will use the `tensorflow <https://tensorflow.org/>`_ Python framework, but similar integrative
analyses are possible with the rest of the Python machine learning and deep learning ecosystem.

The data we will use is the :ref:`Bacteria data`, which is integrated into `RamanSPy`.

"""

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 2
# sphinx_gallery_end_ignore

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

import ramanspy

# %%
# First, we will use `RamanSPy` to load the validation and testing bacteria datasets.
dir_ = r"../../../../data/bacteria_data"

X_train, y_train = ramanspy.datasets.bacteria("val", folder=dir_)
X_test, y_test = ramanspy.datasets.bacteria("test", folder=dir_)

# %%
# Shuffling the dataset we will use to train the model.
X_train, y_train = shuffle(X_train.flat.spectral_data, y_train)


# %%
# Then, we construct the CNN model.
class NN(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.nn = tf.keras.models.Sequential()
        self.nn.add(tf.keras.Input(shape=(input_dim,)))
        self.nn.add(tf.keras.layers.Dense(output_dim, activation='softmax'))

    def call(self, x):
        return self.nn(x)


# %%
# Initialising the model instance
learning_rate = 0.001
batch_size = 32
epochs = 15
input_dim = X_train.shape[-1]
output_dim = len(np.unique(y_train))

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model = NN(input_dim, output_dim)
model.compile(opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
# Training the MLP model on the training dataset.
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)


# %%
# Testing the trained model on the unseen testing dataset.
y_pred = model.predict(X_test.flat.spectral_data)
y_pred = np.argmax(y_pred, axis=1)

print(f"The accuracy of the NN model is: {accuracy_score(y_pred, y_test)}")


# %%
# Confusion matrix:
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True)
plt.show()

# %%
# Accuracy profile:
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# %%
# Loss profile:
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
