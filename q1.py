from id3 import Id3Estimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
from extractData import X, y

predicted = cross_val_predict(Id3Estimator(), X, y, cv=4)

print(accuracy_score(y, predicted))
print(confusion_matrix(y, predicted))
