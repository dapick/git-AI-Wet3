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
    set_of_indexes_havnt_used_yet=range(len(x[0]))
    while len(set_of_indexes_to_return)!= k :
        max_score=0
        max_index=-1
        for i in set_of_indexes_havnt_used_yet :
            if max_index==-1:
                max_index=i
                max_score=score(clf,subset_of_x(x,i),y)
                continue
            indexes=set_of_indexes_to_return
            indexes.append(i)
            cur_score=score(clf,subset_of_x(x, indexes),y)
            if cur_score > max_score:
                max_index=i
                max_score=cur_score
        set_of_indexes_havnt_used_yet.remove(max_index)
        set_of_indexes_to_return.append(max_index)


    raise NotImplementedError

def choose_feature(x, y, k, clf, score):
    if k == 0:
        return []

def subset_of_x(x,i):
