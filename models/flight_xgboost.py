from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# get data
#######################
data = np.load('data/airline_final.npy')
X = data[:,:-1]
y = data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
data = None # free up memory

xgb = XGBClassifier(
    n_estimators=500,
    object='binary:logistic',
    random_state=42,
    n_jobs=-1,
)

# 1 for 0 class, 3 for 1 class
sample_weight = np.where(y_train == 0, 1, 3)
xgb.fit(X_train, y_train, sample_weight=sample_weight)

y_pred = xgb.predict(X_test)

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print('precision: ', precision[1])
print('recall: ', recall[1])
print('f1: ', f1[1])
print('accuracy', accuracy)