import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Airlines.csv')
df = df.drop('id', axis=1)
df_encoded = pd.get_dummies(df, columns=['Airline','AirportFrom','AirportTo'], drop_first=True)

X = df_encoded.drop('Delay', axis=1)
y = df_encoded['Delay']

selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(X, y)

import numpy as np
scores = pd.DataFrame({
    'Feature': X.columns, 
    'Score': selector.scores_
}).sort_values('Score', ascending=False)

top = scores.head(10)['Feature'].tolist()
X_sel = X[top]

X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'lr_deploy.pkl')
joblib.dump(top, 'lr_features.pkl')

import os
print('lr_deploy.pkl size:', os.path.getsize('lr_deploy.pkl'), 'bytes')
print('Done!')