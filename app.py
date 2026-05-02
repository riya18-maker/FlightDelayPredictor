from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

def train_and_save():
    print("Training models...")
    df = pd.read_csv('Airlines.csv')
    df = df.drop('id', axis=1)
    df_encoded = pd.get_dummies(df, columns=['Airline','AirportFrom','AirportTo'], drop_first=True)
    
    X = df_encoded.drop('Delay', axis=1)
    y = df_encoded['Delay']
    
    selector = SelectKBest(score_func=f_classif, k=10)
    selector.fit(X, y)
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)
    top_features = feature_scores.head(10)['Feature'].tolist()
    top_features_50 = feature_scores.head(50)['Feature'].tolist()
    
    X_selected_50 = X[top_features_50]
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected_50, y, test_size=0.2, random_state=42
    )
    
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    joblib.dump(xgb_model, 'xgb_model.pkl')
    joblib.dump(top_features_50, 'top_features_50.pkl')
    print("Models trained and saved!")
    return xgb_model, top_features_50

# Load or train models
if os.path.exists('xgb_model.pkl') and os.path.exists('top_features_50.pkl'):
    model = joblib.load('xgb_model.pkl')
    top_features = joblib.load('top_features_50.pkl')
else:
    model, top_features = train_and_save()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(request.form.get(f, 0)) for f in top_features]
        input_array = np.array(values).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]
        result = "✈️ DELAYED" if prediction == 1 else "✅ ON TIME"
        confidence = f"{probability*100:.1f}%"
        return render_template('index.html', prediction=result, confidence=confidence)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)