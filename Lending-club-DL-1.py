# %%
##Lending club loan prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

# Load dataset
df = pd.read_csv(r"C:\Users\srini\Downloads\CEP_Datasets\CEP_2_Datasets\loan_data.csv")

# %%
# Basic check
print(df.head())
print(df.info())
print(df.isnull().sum())

# %%
# Encode categorical column
le = LabelEncoder()
df["purpose"] = le.fit_transform(df["purpose"])
# Remove duplicates
df = df.drop_duplicates()

# %%
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.show()

# %%
# Features and target
X = df.drop("credit.policy", axis=1)
y = df["credit.policy"]


# %%
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# %%
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# %%
# Build neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(18, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(18, activation="relu"),
    tf.keras.layers.Dense(18, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# %%
# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# %%
# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    batch_size=32
)
loss,accuracy = model.evaluate(X_test,y_test)
print("Test accuracy:",accuracy)

# %%
# Predict
y_pred = (model.predict(X_test) > 0.5).astype("int32")
# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%


# %%



