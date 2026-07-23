# svm_model.py - SVM using Real Dataset (Fixed for NaN values)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SVM MODEL - Using Real Dataset")
print("="*60)

# Load dataset
print("\n[1] Loading dataset...")
data = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='unicode_escape')
print(f"Dataset shape: {data.shape}")

# Show columns
print("\n[2] Columns in dataset:")
print(data.columns.tolist()[:10])

# Identify target column
target_column = 'Delivery Status'  # Using Delivery Status as target

print(f"\n[3] Using target column: '{target_column}'")

# Prepare data
X = data.drop(target_column, axis=1)
y = data[target_column]

# Encode target
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y.astype(str))

# Handle categorical columns
categorical_cols = X.select_dtypes(include=['object', 'string']).columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Handle missing values - Fill NaN with 0
X = X.fillna(0)

# Convert to numeric
X = X.astype(float)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
print("\n[4] Training SVM model...")
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
print("\n[5] Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Training Time: {training_time:.4f} seconds")

print("\n" + "="*60)
print("✅ SVM Model Completed Successfully!")
print("="*60)