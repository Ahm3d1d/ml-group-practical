# svm_model.py - SVM Model for Group Practical
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import time

print("="*60)
print("SVM MODEL - Group Practical")
print("="*60)

# Create dataset
print("\n[1] Creating dataset...")
X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, 
                           n_redundant=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
print("\n[2] Training SVM model...")
start_time = time.time()
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
training_time = time.time() - start_time

# Predict
y_pred = svm.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Results
print("\n[3] Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Training Time: {training_time:.4f} seconds")

print("\n" + "="*60)
print("✅ SVM Model Completed Successfully!")
print("="*60)