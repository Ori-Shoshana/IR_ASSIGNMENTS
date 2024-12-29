import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# General Configurations
OUTPUT_DIR = "bert_results"
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "results")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

# Data Files
DATA_FILES = {
    "bert_withIDF": "bert_withIDF.csv",
    "bert_withoutIDF": "bert_withoutIDF.csv",
    "sbert_vectors": "sbert_vectors.csv"
}

# Functions

def load_data(file_path):
    """Load CSV file and extract features and labels."""
    df = pd.read_csv(file_path)
    if 'Sheet' not in df.columns:
        raise ValueError("Target column 'Sheet' not found in the dataset.")
    
    X = df.drop(columns=['Sheet', 'RowIndex'], errors='ignore').values
    y = df['Sheet'].values
    print(f"Data loaded: {file_path}\nShape of X: {X.shape}, Shape of y: {len(y)}")
    return X, y

def clean_data(X, y):
    """Remove classes with less than 2 samples."""
    class_counts = pd.Series(y).value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask = np.isin(y, valid_classes)
    return X[mask], y[mask]

def train_ann(X_train, y_train, X_val, y_val, activation='relu', model_name='ann_model'):
    """Train an ANN model with specified activation function."""
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(10, activation=activation),
        Dense(10, activation=activation),
        Dense(7, activation=activation),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_path = os.path.join(MODEL_OUTPUT_DIR, f"{model_name}.h5")
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

    # Train
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=15, batch_size=32, callbacks=[early_stop, checkpoint]
    )

    # Plot and save training history
    plot_training_history(history, model_name)
    return model, history

def plot_training_history(history, model_name):
    """Plot and save training history for a model."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_OUTPUT_DIR, f"{model_name}_history.png"))
    plt.close()

def train_classifiers(X, y, group_name):
    """Train multiple classifiers and save their results."""
    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, multi_class='auto'),
        "SVM": SVC(kernel='linear', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"\nTraining {name} for {group_name}")

        # Handle Naive Bayes requirements
        X_transformed = X - X.min(axis=0) if name == "Naive Bayes" and np.any(X < 0) else X

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        try:
            scores = cross_val_score(clf, X_transformed, y, cv=skf, scoring='accuracy', error_score='raise')
            results[name] = np.mean(scores)
            print(f"{name} Average Accuracy: {results[name]:.4f}")
        except ValueError as e:
            print(f"Error during cross-validation for {name}: {e}")
            results[name] = None
            continue

    return results

# Main Execution
for file_key, file_name in DATA_FILES.items():
    print(f"\nProcessing file: {file_key}")
    file_path = os.path.join(DATA_ROOT, file_name)

    # Load and clean data
    X, y = load_data(file_path)
    y = LabelEncoder().fit_transform(y)
    X, y = clean_data(X, y)

    if X.shape[0] == 0 or len(np.unique(y)) < 2:
        print(f"Skipping file {file_key}: insufficient samples or classes.")
        continue

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Train ANN models
    train_ann(X_train, y_train, X_val, y_val, activation='relu', model_name=f"{file_key}_ann_relu")
    train_ann(X_train, y_train, X_val, y_val, activation='gelu', model_name=f"{file_key}_ann_gelu")

    # Train classifiers
    results = train_classifiers(X, y, file_key)
    print(f"Results for {file_key}: {results}")
