from id3 import Id3Estimator
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from extractData import X, y
from statistics import mean

skf = KFold(n_splits=4)
scores = []
est = Id3Estimator()
conf_matrix = np.zeros((2, 2), dtype='int64')
for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    conf_matrix += confusion_matrix(y_test, y_pred)

print(mean(scores))
print(conf_matrix)
