from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# get data
#######################
data = np.load('data/airline_final.npy')
X = data[:,:-1]
y = data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
data = None # free up memory

lr = LogisticRegression(
    random_state=42, 
    n_jobs=-1, 
    class_weight='balanced', 
)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test).round()

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print('precision: ', precision[1])
print('recall: ', recall[1])
print('f1: ', f1[1])
print('accuracy', accuracy)