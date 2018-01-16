from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from id3 import Id3Estimator
from extractData import X, y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)

# Without pruning
no_prone_accuracy = cross_val_score(Id3Estimator(), X_train, y_train, cv=4, scoring='accuracy').mean()
print("Accuracy without pruning on cross validation is:", no_prone_accuracy)

# With pruning
estimator = Id3Estimator(min_samples_split=20)
no_prone_accuracy = cross_val_score(estimator, X_train, y_train, cv=4, scoring='accuracy').mean()
print("Accuracy with pruning on cross validation is:", no_prone_accuracy)

# estimator.fit(X_train, y_train)
# print("Accuracy with pruning on test is:", accuracy_score(y_test, estimator.predict(X_test)))
