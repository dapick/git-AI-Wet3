import sfs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from copy import deepcopy


from extractData import X, y

def scoreSFS(clf,x,y):
    return cross_val_score(clf, x, y, cv=4).mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


es=KNeighborsClassifier()

es.fit(X_train,y_train)

#predicted = cross_val_predict(KNeighborsClassifier(), X, y)
#print(accuracy_score(y, predicted))
#print(sfs(X, y, 8, KNeighborsClassifier, scoree))
#predicted = cross_val_predict(KNeighborsClassifier(), sfs.subset_of_x(X,sfs(X, y, 8, KNeighborsClassifier, scoree)), y)
print(accuracy_score(y_test, es.predict(X_test)))

number_of_indexes=sfs.sfs(X_train, y_train, 8, KNeighborsClassifier(), scoreSFS)
es.fit(sfs.subset_of_x(X_train,number_of_indexes),y_train)
print(accuracy_score(y_test, es.predict(sfs.subset_of_x(X_test,number_of_indexes))))

# print("Accuracy on train:", accuracy_score(y_train, estimator.predict(X_train)))
# print("Accuracy on test:", accuracy_score(y_test, estimator.predict(X_test)))