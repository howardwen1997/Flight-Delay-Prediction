from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# get data
#######################
data = np.load('data/airline_final.npy')
X = data[:,:-1]
y = data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
data = None # free up memory

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
#    max_depth=40,
    random_state=42,
    verbose=2,
    n_jobs=-1,
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print('feature importance', rf.feature_importances_)
print('precision: ', precision[1])
print('recall: ', recall[1])
print('f1: ', f1[1])
print('accuracy', accuracy)
