from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

model = joblib.load('lr_deploy.pkl')
top_features = joblib.load('lr_features.pkl')

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