# random_forest_model.py
# APP4080 Collaborative Software Development
# Group Practical - Random Forest Classifier
# Problem: Late Delivery Prediction
# Dataset: DataCo Smart Supply Chain Dataset


import pandas as pd
import numpy as np
import time
import warnings

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


warnings.filterwarnings('ignore')


print("=" * 70)
print("RANDOM FOREST CLASSIFIER - LATE DELIVERY PREDICTION")
print("=" * 70)


# ======================================================
# 1. LOAD DATASET
# ======================================================

print("\n[1] Loading DataCo Supply Chain Dataset...")


data = pd.read_csv(
    "dataset/dataco-smart-supply-chain-for-big-data-analysis/DataCoSupplyChainDataset.csv",
    header=0,
    encoding="unicode_escape"
)


print("Dataset loaded successfully")
print("Dataset Shape:", data.shape)



# ======================================================
# 2. CREATE TARGET VARIABLE
# Late Delivery = 1
# Not Late Delivery = 0
# ======================================================


print("\n[2] Creating late delivery target...")


data['late_delivery'] = np.where(
    data['Delivery Status'] == 'Late delivery',
    1,
    0
)



# ======================================================
# 3. DROP UNNECESSARY COLUMNS
# Same logic as professor notebook
# ======================================================


data.drop(
    [
        'Delivery Status',
        'Late_delivery_risk',
        'Order Status',
        'order date (DateOrders)'
    ],
    axis=1,
    inplace=True,
    errors='ignore'
)



# ======================================================
# 4. HANDLE MISSING VALUES
# ======================================================


data = data.fillna(0)



# ======================================================
# 5. ENCODE CATEGORICAL FEATURES
# ======================================================


print("\n[3] Encoding categorical features...")


encoder = preprocessing.LabelEncoder()


categorical_columns = data.select_dtypes(
    include=['object']
).columns


for column in categorical_columns:
    data[column] = encoder.fit_transform(
        data[column].astype(str)
    )



# ======================================================
# 6. SPLIT FEATURES AND TARGET
# ======================================================


X = data.drop(
    'late_delivery',
    axis=1
)


y = data['late_delivery']



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])



# ======================================================
# 7. FEATURE SCALING
# ======================================================


scaler = StandardScaler()


X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



# ======================================================
# 8. RANDOM FOREST MODEL
# Professor's Parameters
# ======================================================


print("\n[4] Training Random Forest Model...")


model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=0
)



start_time = time.time()


model.fit(
    X_train,
    y_train
)


training_time = time.time() - start_time



# ======================================================
# 9. PREDICTION
# ======================================================


y_pred = model.predict(
    X_test
)



# ======================================================
# 10. EVALUATION
# ======================================================


accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred
)


recall = recall_score(
    y_test,
    y_pred
)


f1 = f1_score(
    y_test,
    y_pred
)


cm = confusion_matrix(
    y_test,
    y_pred
)



# ======================================================
# 11. DISPLAY RESULTS
# ======================================================


print("\n" + "=" * 70)
print("RANDOM FOREST RESULTS")
print("=" * 70)


print(f"Accuracy : {accuracy * 100:.2f}%")

print(f"Precision: {precision * 100:.2f}%")

print(f"Recall   : {recall * 100:.2f}%")

print(f"F1 Score : {f1 * 100:.2f}%")


print("\nConfusion Matrix:")
print(cm)


print(
    f"\nTraining Time: {training_time:.4f} seconds"
)



print("\n" + "=" * 70)
print("Random Forest Model Completed Successfully")
print("=" * 70)