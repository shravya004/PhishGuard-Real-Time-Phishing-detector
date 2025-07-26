import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("model", exist_ok=True)
df = pd.read_csv("clean_dataset.csv")
df["login_frequency"] = np.random.uniform(0, 1, len(df))
df["device_type"] = np.random.choice([0, 1], len(df))  # 0 = Mobile, 1 = Desktop

X = df[["login_frequency", "device_type"]]
y = df["label"]
clf = RandomForestClassifier()
clf.fit(X, y)
joblib.dump(clf, "model/behavior_model.pkl")
print("✅ Behavioral model trained and saved to model/behavior_model.pkl")
