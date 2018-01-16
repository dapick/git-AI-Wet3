import sfs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from id3 import Id3Estimator
from statistics import mean

from extractData import X, y


def scoreSFS(clf, x, y):
    return cross_val_score(clf, x, y, cv=4).mean()


results = {"KNN_with": [], "KNN_without": [],
           "Tree_with": [], "Tree_without": []}
for _ in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Question 7
    # Without choosing parameters
    es = KNeighborsClassifier()
    es.fit(X_train, y_train)
    results["KNN_without"].append(accuracy_score(y_test, es.predict(X_test)))

    # With choosing parameters
    number_of_indexes = sfs.sfs(X_train, y_train, 8, KNeighborsClassifier(), scoreSFS)
    es.fit(sfs.subset_of_x(X_train, number_of_indexes), y_train)
    results["KNN_with"].append(accuracy_score(y_test, es.predict(sfs.subset_of_x(X_test, number_of_indexes))))

    # Question 8
    # Without pruning
    es = Id3Estimator()
    es.fit(X_train, y_train)
    results["Tree_without"].append(accuracy_score(y_test, es.predict(X_test)))

    # With pruning
    es = Id3Estimator(min_samples_split=20)
    es.fit(X_train, y_train)
    results["Tree_with"].append(accuracy_score(y_test, es.predict(X_test)))

print(mean(results["KNN_without"]))
print(mean(results["KNN_with"]))
print(mean(results["Tree_without"]))
print(mean(results["Tree_with"]))
