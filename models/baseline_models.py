import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

rf = RandomForestClassifier(
    n_estimators=400,
    # class_weight='balanced',
    max_depth=10,
    min_samples_leaf=100,
    random_state=42,
    verbose=2,
    n_jobs=12,
)

lr = LogisticRegression(
    random_state=42, 
    n_jobs=-1, 
    # class_weight={0 : 1., 1 : 3.}, 
)

models = {
    'XGBoost' : xgb,
    'Logistic Regression' : lr,
    'Random Forest' : rf
}

columns=[
    'hidden layers', 
    'hidden nodes', 
    'lr', 
    'dropout',
    'accuracy',
    'precision',
    'recall',
    'f1',
    'auc'
]


# 1 for 0 class, 3 for 1 class
sample_weight = np.where(y_train == 0, 1, 3)

columns=[
    'hidden layers', 
    'hidden nodes', 
    'lr', 
    'accuracy',
    'precision',
    'recall',
    'f1',
    'auc'
]


results = pd.DataFrame(columns=columns)

for i, model in enumerate(models.keys()):
    
    models[model].fit(X_train, y_train, sample_weight=sample_weight)
    
    y_pred = models[model].predict(X_test)
    
    precision, recall, f1, support  = precision_recall_fscore_support(y_test, y_pred)
    precision = precision[1]
    recall = recall[1]
    f1 = f1[1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    d = {
        'accuracy' : accuracy,
        'precision' : precision,
        'recall' : recall,
        'f1' : f1,
        'accuracy' : accuracy, 
        'auc' : auc,
    }
    
    results.loc[i] = d

print(results.to_latex(index=False))