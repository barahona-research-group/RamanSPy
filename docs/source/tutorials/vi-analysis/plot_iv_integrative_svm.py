"""
Integrative analysis: Support Vector Machine (SVM) classification
===================================================================

In this example, we will showcase `RamanSPy's` integrability by integrating a Support Vector Machine (SVM) machine learning
model for the identification of different bacteria species.

To build the model, we will use the `scikit-learn <https://scikit-learn.org/stable/index.html>`_ Python framework.

The data we will use is the :ref:`Bacteria data`, which is integrated into `RamanSPy`.

"""
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
import seaborn as sns

import ramanspy

# %%
# First, we will use `RamanSPy` to load the validation and testing bacteria datasets.
dir_ = r"../../../../data/bacteria_data"

X_train, y_train = ramanspy.datasets.bacteria("val", folder=dir_)
X_test, y_test = ramanspy.datasets.bacteria("test", folder=dir_)


# %%
# To guide the training, it is important to shuffle the training dataset, which is originally ordered by bacteria species.
X_train, y_train = shuffle(X_train.flat.spectral_data, y_train)


# %%
# Then, we can simply use `scikit-learn's` implementation of SVMs.
svc = svm.SVC()  # initialisation


# %%
# Training the SVM model on the training dataset.
_ = svc.fit(X_train, y_train)


# %%
# Testing the trained model on the unseen testing dataset.
y_pred = svc.predict(X_test.flat.spectral_data)
print(f"The accuracy of the SVM model is: {accuracy_score(y_pred, y_test)}")


# %%
# Confusion matrix:
cf_matrix = confusion_matrix(y_test, y_pred)
_ = sns.heatmap(cf_matrix, annot=True)
