from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
from copy import deepcopy

def sfs(x, y, k, clf, score):
    """
    :param x: feature set to be trained using clf. list of lists.
    :param y: labels corresponding to x. list.
    :param k: number of features to select. int
    :param clf: classifier to be trained on the feature subset.
    :param score: utility function for the algorithm, that receives clf, feature subset and labeles, returns a score. 
    :return: list of chosen feature indexes
    """
    set_of_indexes_to_return=[]
    set_of_indexes_havnt_used_yet=list(range(len(x[0])))
    while len(set_of_indexes_to_return)!= k and len(set_of_indexes_havnt_used_yet) != 0  :
        print(set_of_indexes_to_return)
        print(set_of_indexes_havnt_used_yet)
        print("yo")
        max_score=0
        max_index=-1
        for i in set_of_indexes_havnt_used_yet :
            indexes= deepcopy(set_of_indexes_to_return)
            indexes.append(i)
            if max_index==-1:
                max_index=i
                max_score = score(clf, subset_of_x(x, indexes), y)
                continue
            cur_score=score(clf,subset_of_x(x, indexes),y)
            if cur_score > max_score:
                max_index=i
                max_score=cur_score
        set_of_indexes_havnt_used_yet.remove(max_index)
        set_of_indexes_to_return.append(max_index)
    return set_of_indexes_to_return


def subset_of_x(x,indexes):
    x_subset = []
    for ls in x:
        new_list = [ls[i] for i in indexes]
        x_subset.append(new_list)
    return x_subset

from extractData import X, y

def scoree(clf,x,y):
    predicted = cross_val_predict(clf(), x, y, cv=4)
    return accuracy_score(y, predicted)


predicted = cross_val_predict(KNeighborsClassifier(), X, y)
print(accuracy_score(y, predicted))
print(sfs(X, y, 8, KNeighborsClassifier, scoree))
predicted = cross_val_predict(KNeighborsClassifier(), subset_of_x(X,sfs(X, y, 8, KNeighborsClassifier, scoree)), y)
print(accuracy_score(y, predicted))