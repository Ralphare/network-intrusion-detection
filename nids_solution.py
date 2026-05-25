import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# This notebook was developed and run in Jupyter Notebook.
# To run: jupyter notebook nids_solution.py
# Or open as .ipynb in Jupyter/Colab.
# ============================================================

SEED = 42
print("All imports done!")

# ==============================================================
# 1. LOAD DATA
# ==============================================================

train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'level'
]

df_train = pd.read_csv(train_url, names=columns)
df_test  = pd.read_csv(test_url,  names=columns)

df_train.drop(columns=['level'], inplace=True)
df_test.drop(columns=['level'],  inplace=True)

print(f"Training set: {df_train.shape[0]} records")
print(f"Test set:     {df_test.shape[0]} records")

# ==============================================================
# 2. ENCODE CATEGORICAL FEATURES
# ==============================================================

df_full = pd.concat([df_train, df_test])

# Encode text columns to numbers
cat_cols = ['protocol_type', 'service', 'flag']
for col in cat_cols:
    le = LabelEncoder()
    df_full[col] = le.fit_transform(df_full[col])

# ==============================================================
# 3. MAP 39 ATTACK NAMES TO 5 CATEGORIES
# ==============================================================

category_map = {
    'normal': 'Normal',
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'ftp_write': 'R2L',
    'imap': 'R2L', 'phf': 'R2L', 'multihop': 'R2L', 'warezmaster': 'R2L',
    'spy': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}

df_full['category'] = df_full['class'].map(category_map).fillna('Other')
df_full.drop(columns=['num_outbound_cmds', 'class'], inplace=True)

# ==============================================================
# 4. SPLIT BACK INTO TRAIN / TEST
# ==============================================================

train_len = len(df_train)
df_train_processed = df_full.iloc[:train_len].copy()
df_test_processed  = df_full.iloc[train_len:].copy()

X_train = df_train_processed.drop(columns=['category'])
y_train = df_train_processed['category']
X_test  = df_test_processed.drop(columns=['category'])
y_test  = df_test_processed['category']

print("\nTraining class distribution:")
print(y_train.value_counts())
print("\nTest class distribution:")
print(y_test.value_counts())

# ==============================================================
# 5. SMOTE — FIX CLASS IMBALANCE
# ==============================================================

print("\nBefore SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=SEED, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

# ==============================================================
# 6. ENCODE LABELS FOR XGBOOST
# ==============================================================

# XGBoost needs numbers not strings
# DoS->0, Normal->1, Probe->2, R2L->3, U2R->4
le_target = LabelEncoder()
y_train_balanced_enc = le_target.fit_transform(y_train_balanced)
y_test_enc = le_target.transform(y_test)

print("\nLabel mapping:")
for i, cls in enumerate(le_target.classes_):
    print(f"  {cls} → {i}")

# ==============================================================
# 7. DEFINE MODELS
# ==============================================================

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        random_state=SEED,
        eval_metric='mlogloss',
        n_jobs=-1
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,
        verbose=-1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        auto_class_weights='Balanced',
        random_seed=SEED,
        verbose=0
    ),
}

print("Models ready:", list(models.keys()))

# ==============================================================
# 8. TRAIN AND EVALUATE ALL MODELS
# ==============================================================

results = {}

for name, model in models.items():
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"{'='*40}")

    # XGBoost needs numeric labels
    if name == "XGBoost":
        y_tr = y_train_balanced_enc
        y_te = y_test_enc
    else:
        y_tr = y_train_balanced
        y_te = y_test

    print("Running cross-validation...")
    cv_scores = cross_val_score(
        model, X_train_balanced, y_tr,
        cv=5, scoring='f1_macro', n_jobs=-1
    )
    print(f"CV Macro F1: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

    print("Training...")
    model.fit(X_train_balanced, y_tr)

    y_pred = model.predict(X_test)

    # Decode XGBoost predictions back to strings
    if name == "XGBoost":
        y_pred = le_target.inverse_transform(y_pred)
        y_te = y_test

    test_f1 = f1_score(y_te, y_pred, average='macro')
    print(f"Test Macro F1: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred))

    results[name] = {
        'cv_f1':   cv_scores.mean(),
        'cv_std':  cv_scores.std(),
        'test_f1': test_f1,
        'y_pred':  y_pred
    }

# ==============================================================
# 9. SUMMARY TABLE
# ==============================================================

print(f"\n{'Model':<15} {'CV F1':>10} {'Test F1':>10} {'Gap':>10}")
print("-" * 50)
for name, res in results.items():
    gap = res['cv_f1'] - res['test_f1']
    print(f"{name:<15} {res['cv_f1']:>10.4f} {res['test_f1']:>10.4f} {gap:>10.4f}")

best_name = max(results, key=lambda k: results[k]['test_f1'])
print(f"\nBest model: {best_name} — Test Macro F1 = {results[best_name]['test_f1']:.4f}")

# ==============================================================
# 10. CONFUSION MATRIX
# ==============================================================

best_pred = results[best_name]['y_pred']
labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]

cm = confusion_matrix(y_test, best_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {best_name}\nMacro F1: {results[best_name]['test_f1']:.4f}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved as confusion_matrix.png")
