from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

model = joblib.load('lr_deploy.pkl')
top_features = joblib.load('lr_features.pkl')
kmeans = joblib.load('kmeans_model.pkl')
cluster_order = joblib.load('cluster_order.pkl')

CLUSTER_NAMES = {
    0: ("🟢 Low Risk", "Early departure flights with lowest delay probability — safest time to fly."),
    1: ("🟡 Moderate Risk", "Mid-day flights with moderate delay probability — average risk level."),
    2: ("🔴 High Risk", "Late departure flights with highest delay probability — most likely to be delayed.")
}

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

        time_val = float(request.form.get('Time', 0))
        cluster_input = np.array([[time_val, 130, 4, prediction]])
        cluster_id = kmeans.predict(cluster_input)[0]
        risk_level = cluster_order.index(cluster_id)
        cluster_name, cluster_desc = CLUSTER_NAMES[risk_level]
        print(f"Cluster ID: {cluster_id}, Risk Level: {risk_level}, Name: {cluster_name}")

        return render_template('index.html',
                             prediction=result,
                             confidence=confidence,
                             cluster_name=cluster_name,
                             cluster_desc=cluster_desc)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)