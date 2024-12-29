import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# General Configurations
DATA_ROOT = "IR-files"
OUTPUT_DIR = "classiffication_res"
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "results")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

# Data Files
GROUPS = ['bert-sbert', 'doc2vec', 'glove', 'word2vec']

# Functions

def load_combined_data(group_name):
    """Load combined matrix for a group."""
    file_path = os.path.join(DATA_ROOT, f"{group_name}_clustering_results.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    return pd.read_csv(file_path)

def train_ann(X, y, activation='relu', embedding_output=128):
    """Train an ANN model."""
    num_classes = len(np.unique(y))
    input_dim = X.shape[1]

    model = Sequential([
        Dense(embedding_output, input_dim=input_dim, activation=activation),
        Dense(10, activation=activation),
        Dense(10, activation=activation),
        Dense(7, activation=activation),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_classifiers(X, y, group_name):
    """Train multiple classifiers and save results."""
    classifiers = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(kernel='linear', probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }
    feature_importance = {}

    for clf_name, clf in classifiers.items():
        print(f"\nTraining {clf_name} for group: {group_name}")
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
        print(f"{clf_name} Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

        clf.fit(X, y)
        if hasattr(clf, 'coef_'):
            importance = np.abs(clf.coef_[0])
        elif hasattr(clf, 'feature_importances_'):
            importance = clf.feature_importances_
        else:
            importance = np.zeros(X.shape[1])

        feature_importance[clf_name] = importance

    # Save feature importance
    feature_df = pd.DataFrame(feature_importance, index=[f"Feature {i+1}" for i in range(X.shape[1])])
    feature_df['Mean_Importance'] = feature_df.mean(axis=1)
    feature_df = feature_df.sort_values(by='Mean_Importance', ascending=False).head(20)
    feature_df.to_excel(os.path.join(RESULTS_OUTPUT_DIR, f"{group_name}_feature_importance.xlsx"))
    print(f"Feature importance for {group_name} saved to Excel.")

def split_and_train(X, y, group_name):
    """Split data and train ANN and classifiers."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # ANN Training
    for activation in ['relu', 'gelu']:
        early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
        model_path = os.path.join(MODEL_OUTPUT_DIR, f"best_ann_{activation}_{group_name}.h5")
        checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)

        ann_model = train_ann(X_train, y_train, activation=activation)
        history = ann_model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val),
            epochs=15, batch_size=32, callbacks=[early_stop, checkpoint]
        )
        ann_loss, ann_acc = ann_model.evaluate(X_test, y_test)
        print(f"ANN ({activation}) Test Accuracy for {group_name}: {ann_acc:.4f}")

    # Train Other Classifiers
    train_and_evaluate_classifiers(X, y, group_name)

# Main Execution
for group in GROUPS:
    print(f"\nProcessing group: {group}")
    data = load_combined_data(group)
    if data is None:
        continue

    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    # Validate labels
    if y.dtype != np.int32:
        print("Converting labels to integers.")
        y = y.astype(np.int32)

    split_and_train(X, y, group)

