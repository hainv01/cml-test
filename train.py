import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")


# fit a model
depth = 2
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(acc)
with open("metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy: {acc}")

# plot it ok
predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
plt.savefig("confusion_matrix.png")
