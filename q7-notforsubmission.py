import sfs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from copy import deepcopy


from extractData import X, y

def scoreSFS(clf,x,y):
    return cross_val_score(clf, x, y, cv=4).mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=17)


es=KNeighborsClassifier()

es.fit(X_train,y_train)

print(accuracy_score(y_test, es.predict(X_test)))

number_of_indexes=sfs.sfs(X_train, y_train, 8, KNeighborsClassifier(), scoreSFS)
es.fit(sfs.subset_of_x(X_train,number_of_indexes),y_train)
print(accuracy_score(y_test, es.predict(sfs.subset_of_x(X_test,number_of_indexes))))
