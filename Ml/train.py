import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve

DATA_PATH = Path('Ml/data/loan_fraud_dataset.csv')
MODEL_PATH = Path('ML/models/fraud_model.pkl')
THRESHOLD_PATH = Path('ML/models/threshold.json')
METRICS_PATH = Path('ML/models/metrics.json')

#  Cette fonction sert à choisir automatiquement le meilleur seuil de décision pour ton modèle de classification binaire, ici fraud / non-fraud, à partir des probabilités prédites par le modèle.
def best_threshold(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / (p + r + 1e-9)
    idx = f1[:-1].argmax()
    return float(t[idx]), float(f1[idx]), float(p[idx]), float(r[idx])


def main():
    df = pd.read_csv(DATA_PATH)

    FEATURE_COLUMNS = [
    "income_declared",
    "income_detected",
    "address_mismatch",
    "device_location",
    "document_authenticity_score",
    "account_balance_pattern",
    "employment_mismatch",
    "rapid_loan_requests",
                        ]
    
    X = df[FEATURE_COLUMNS]   # only the 8 real features
    y = df['fraud_flag'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    num_cols = X.columns.tolist()
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols)
    ])

    clf = LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=None)
    model = Pipeline([
        ('preprocess', preprocessor),
        ('clf', clf)
    ])

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    threshold, f1, precision, recall = best_threshold(y_test, proba)
    pred = (proba >= threshold).astype(int)

    metrics = {
        'roc_auc': float(roc_auc_score(y_test, proba)),
        'pr_auc': float(average_precision_score(y_test, proba)),
        'precision': float((pred & (y_test.values == 1)).sum() / max(pred.sum(), 1)),
        'recall': float((pred & (y_test.values == 1)).sum() / max((y_test == 1).sum(), 1)),
        'f1_best_threshold': float(f1),
        'best_threshold_precision': float(precision),
        'best_threshold_recall': float(recall)
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    THRESHOLD_PATH.write_text(json.dumps({'threshold': threshold}, indent=2))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(classification_report(y_test, pred, digits=4))
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
