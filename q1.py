import pandas as pd
from id3 import Id3Estimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score

features = "f1_0,f1_1,f1_2,f1_3,f1_4,f1_5,f2_0,f2_1,f2_2,f2_3,f2_4,f2_5,f3_0,f3_1,f3_2,f3_3,f4_1,f4_2,f5_1,f5_2,f5_3,f6_1,f6_2,f6_3,f7_1,f7_2,f8_1,f8_2,f9_1,f9_2,f10_1,f10_2".split(",")
data = pd.read_csv('flare.csv')
X = data[features].values
y = data["classification"].values
predicted = cross_val_predict(Id3Estimator(), X, y, cv=4)

print(accuracy_score(y, predicted))
print(confusion_matrix(y, predicted))
