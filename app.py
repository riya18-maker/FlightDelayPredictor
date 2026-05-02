from flask import Flask, render_template, request
import numpy as np
import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Train simple fast model on startup
print("Loading data and training model...")
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
X_selected = X[top_features]

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("Model ready!")

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