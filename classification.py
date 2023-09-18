# imports
import pandas as pd
import numpy as np
from ast import literal_eval
from gomi_config import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ファイル読み込み
gomi_embedding = pd.read_csv(f"output/0133_20230307_embedding_{mode}.csv")
# 文字列化されているカラムを ndarray 化
gomi_embedding["embedding"] = [np.array(literal_eval(d)) for d in gomi_embedding['embedding']]


# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    list(gomi_embedding["embedding"].values), gomi_embedding["出し方"], test_size=0.2, random_state=42
)

# train random forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
probas = clf.predict_proba(X_test)

report = classification_report(y_test, preds)
print(report)

joblib.dump(clf, f"output/0133_20230307_model_{mode}.joblib")