import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('airlines.csv')
df = df.drop('id', axis=1)
df_encoded = pd.get_dummies(df, columns=['Airline','AirportFrom','AirportTo'], drop_first=True)

X = df_encoded.drop('Delay', axis=1)
y = df_encoded['Delay']

# Forward Selection
selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(X, y)
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': selector.scores_
}).sort_values('Score', ascending=False)
top_features = feature_scores.head(10)['Feature'].tolist()
X_selected = X[top_features]

# Top 50 for RF and XGB
top_features_50 = feature_scores.head(50)['Feature'].tolist()
X_selected_50 = X[top_features_50]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(
    X_selected_50, y, test_size=0.2, random_state=42
)

# ---- LOGISTIC REGRESSION ----
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_acc = round(accuracy_score(y_test, lr_model.predict(X_test)) * 100, 2)
print(f"Logistic Regression: {lr_acc}%")

# ---- POLYNOMIAL REGRESSION ----
print("Training Polynomial Regression...")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = (np.clip(poly_model.predict(X_test_poly), 0, 1) >= 0.5).astype(int)
poly_acc = round(accuracy_score(y_test, y_pred_poly) * 100, 2)
print(f"Polynomial Regression: {poly_acc}%")

# ---- RANDOM FOREST ----
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train_50, y_train_50)
rf_acc = round(accuracy_score(y_test_50, rf_model.predict(X_test_50)) * 100, 2)
print(f"Random Forest: {rf_acc}%")

# ---- XGBOOST ----
print("Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_model.fit(X_train_50, y_train_50)
y_pred_xgb = xgb_model.predict(X_test_50)
xgb_acc = round(accuracy_score(y_test_50, y_pred_xgb) * 100, 2)
print(f"XGBoost: {xgb_acc}%")
print("\nXGBoost Classification Report:")
print(classification_report(y_test_50, y_pred_xgb))

# Confusion Matrix XGBoost
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test_50, y_pred_xgb)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr',
            xticklabels=['On Time','Delayed'],
            yticklabels=['On Time','Delayed'])
plt.title("Confusion Matrix - XGBoost")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig('confusion_matrix_xgb.png')
plt.close()

# Model Comparison
plt.figure(figsize=(12,6))
models = ['Logistic\nRegression', 'Polynomial\nRegression', 'Random\nForest', 'XGBoost']
accuracies = [lr_acc, poly_acc, rf_acc, xgb_acc]
colors = ['#4a90d9', '#e67e22', '#f5c518', '#2ecc71']
bars = plt.bar(models, accuracies, color=colors, width=0.5, edgecolor='white')
plt.axhline(y=80, color='red', linestyle='--', linewidth=1.5, label='80% Target')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
plt.title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()
print("\nGraphs saved!")

# Save everything
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(top_features_50, 'top_features_50.pkl')
joblib.dump(lr_model, 'logistic_model.pkl')
joblib.dump(top_features, 'top_features.pkl')
joblib.dump(poly_model, 'poly_model.pkl')
joblib.dump(poly, 'poly_transformer.pkl')
print("All models saved!")
# ---- DELAY CLUSTER PROFILING (KMeans) ----
print("\n=== DELAY CLUSTER PROFILING ===")
from sklearn.cluster import KMeans

# Use numerical features for clustering
cluster_features = df[['Time', 'Length', 'DayOfWeek', 'Delay']]

# KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(cluster_features)

# Add cluster labels
df['Cluster'] = kmeans.labels_

# Analyze clusters
cluster_analysis = df.groupby('Cluster').agg({
    'Delay': 'mean',
    'Time': 'mean',
    'Length': 'mean',
    'DayOfWeek': 'mean'
}).round(2)

print("\nCluster Analysis:")
print(cluster_analysis)

# Plot clusters
plt.figure(figsize=(10,6))
colors = ['#10B981', '#F5C518', '#EF4444']
labels = ['Low Risk', 'Moderate Risk', 'High Risk']

# Sort clusters by delay rate
delay_order = cluster_analysis['Delay'].sort_values().index.tolist()

for i, cluster_id in enumerate(delay_order):
    mask = df['Cluster'] == cluster_id
    plt.scatter(df[mask]['Time'], df[mask]['Delay'],
               c=colors[i], label=labels[i], alpha=0.3, s=5)

plt.title("Delay Cluster Profiling — KMeans (k=3)")
plt.xlabel("Departure Time (minutes from midnight)")
plt.ylabel("Delay (0=On Time, 1=Delayed)")
plt.legend()
plt.tight_layout()
plt.savefig('cluster_profiling.png')
plt.close()
print("Cluster graph saved!")

# Save kmeans model
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(delay_order, 'cluster_order.pkl')
print("KMeans model saved!")