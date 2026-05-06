import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Airlines.csv')
df = df.drop('id', axis=1)
df_encoded = pd.get_dummies(df, columns=['Airline','AirportFrom','AirportTo'], drop_first=True)

X = df_encoded.drop('Delay', axis=1)
y = df_encoded['Delay']

selector = SelectKBest(score_func=f_classif, k=50)
selector.fit(X, y)
scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': selector.scores_
}).sort_values('Score', ascending=False)

top = scores.head(50)['Feature'].tolist()
X_sel = X[top]

X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, random_state=42)
X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb.fit(X_train2, y_train2)

calibrated = CalibratedClassifierCV(xgb, method='sigmoid')
calibrated.fit(X_val, y_val)

joblib.dump(calibrated, 'lr_deploy.pkl')
joblib.dump(top, 'lr_features.pkl')

import os
print('lr_deploy.pkl size:', os.path.getsize('lr_deploy.pkl'), 'bytes')
print('Done! Calibrated XGBoost saved.')