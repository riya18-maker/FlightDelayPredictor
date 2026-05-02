from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('xgb_model.pkl')
top_features = joblib.load('top_features_50.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all features, default to 0 if not in form
        values = []
        for f in top_features:
            val = request.form.get(f, 0)
            values.append(float(val))
        
        input_array = np.array(values).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]

        result = "✈️ DELAYED" if prediction == 1 else "✅ ON TIME"
        confidence = f"{probability*100:.1f}%"

        return render_template('index.html',
                             prediction=result,
                             confidence=confidence)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)