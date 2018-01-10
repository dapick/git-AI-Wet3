from id3 import Id3Estimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from extractData import X, y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Overfitting
estimator = Id3Estimator(min_samples_split=1)
estimator.fit(X_train, y_train)
# print("Accuracy on train:", accuracy_score(y_train, estimator.predict(X_train)))
# print("Accuracy on test:", accuracy_score(y_test, estimator.predict(X_test)))
print(accuracy_score(y_train, estimator.predict(X_train)))

# Underfitting
estimator = Id3Estimator(max_depth=2)
estimator.fit(X_train, y_train)
# print("Accuracy on train:", accuracy_score(y_train, estimator.predict(X_train)))
# print("Accuracy on test:", accuracy_score(y_test, estimator.predict(X_test)))
print(accuracy_score(y_train, estimator.predict(X_train)))
