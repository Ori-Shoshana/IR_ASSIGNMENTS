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
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Paths
data_files = {
    "glove_clean_withIDF_withoutStopWords": "glove_clean_withIDF_withoutStopWords.csv",
    "glove_clean_withIDF_withStopWords": "glove_clean_withIDF_withStopWords.csv",
    "glove_clean_withoutIdf_withoutStopWords": "glove_clean_withoutIdf_withoutStopWords.csv",
    "glove_clean_withoutIdf_withStopWords": "glove_clean_withoutIdf_withStopWords.csv",
    "glove_lemma_withIDF_withoutStopWords": "glove_lemma_withIDF_withoutStopWords.csv",
    "glove_lemma_withIDF_withStopWords": "glove_lemma_withIDF_withStopWords.csv",
    "glove_lemma_withoutIdf_withoutStopWords": "glove_lemma_withoutIdf_withoutStopWords.csv",
    "glove_lemma_withoutIdf_withStopWords": "glove_lemma_withoutIdf_withStopWords.csv"
}
output_dir = "glove_results"
os.makedirs(output_dir, exist_ok=True)

# Load Data
def load_data(file_path):
    """Load CSV file and extract features (X) and target labels (y)."""
    df = pd.read_csv(file_path)
    
    # Verify and extract the target column
    if 'Sheet' not in df.columns:
        raise ValueError("Target column 'Sheet' not found in the dataset.")
    
    y = df['Sheet'].values  # Target labels
    X = df.drop(columns=['Sheet', 'RowIndex'], errors='ignore').values  # Features
    
    # Validate loaded data
    print(f"Data loaded from {file_path}")
    print(f"Shape of X: {X.shape}, Shape of y: {len(y)}")
    print(f"Unique labels in y: {np.unique(y)}")
    
    return X, y

# Clean Data
def clean_data(X, y):
    """ Remove classes with less than 2 samples """
    class_counts = pd.Series(y).value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask = np.isin(y, valid_classes)
    return X[mask], y[mask]

# ANN Model
def train_ann(X_train, y_train, X_val, y_val, activation='relu', model_name='ann_model'):
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
    model_path = os.path.join(output_dir, f"{model_name}.h5")
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
    
    # Train
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=15, batch_size=32, callbacks=[early_stop, checkpoint])
    
    # Plot training history
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
    plt.savefig(os.path.join(output_dir, f"{model_name}_history.png"))
    plt.close()
    
    return model, history

def train_classifiers(X, y, group_name):
    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, multi_class='auto'),
        "SVM": SVC(kernel='linear', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"\nTraining {name} for {group_name}")

        # Ensure X is non-negative for MultinomialNB
        if name == "Naive Bayes":
            if np.any(X < 0):
                print(f"Transforming X to non-negative values for {name}.")
                X_transformed = X - X.min(axis=0)
            else:
                X_transformed = X
        else:
            X_transformed = X

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        try:
            scores = cross_val_score(clf, X_transformed, y, cv=skf, scoring='accuracy', error_score='raise')
            results[name] = np.mean(scores)
            print(f"{name} Average Accuracy: {results[name]:.4f}")
        except ValueError as e:
            print(f"Error during cross-validation for {name}: {e}")
            results[name] = None
            continue

        # Fit the model to extract feature importance
        try:
            clf.fit(X_transformed, y)
            
            if hasattr(clf, 'coef_'):
                # Combine coefficients for multi-class problems
                importance = np.abs(clf.coef_).mean(axis=0)
            elif hasattr(clf, 'feature_importances_'):
                importance = clf.feature_importances_
            else:
                importance = np.zeros(X.shape[1])
            
            # Save top 20 features
            top_features = pd.DataFrame({'Feature': range(X.shape[1]), 'Importance': importance})
            top_features = top_features.nlargest(20, 'Importance')
            top_features.to_excel(os.path.join(output_dir, f"{group_name}_{name}_features.xlsx"), index=False)
        except Exception as e:
            print(f"Could not extract feature importance for {name}: {e}")

    return results


# Main Execution
for file_key, file_name in data_files.items():
    print(f"\nProcessing file: {file_key}")
    file_path = f"C:/Users/danie/Desktop/IR/Information_retrieval/exe2/IR-files/glove/{file_name}"
    
    # Load and clean data
    X, y = load_data(file_path)
    y = LabelEncoder().fit_transform(y)
    X, y = clean_data(X, y)

    # Check if data is empty after cleaning
    if X.shape[0] == 0 or len(np.unique(y)) < 2:
        print(f"Skipping file {file_key} because it has no valid samples or insufficient classes.")
        continue

    # Split Data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # ANN Training
    print("\nTraining ANN Topology 1...")
    ann_model_1, _ = train_ann(X_train, y_train, X_val, y_val, activation='relu', model_name=f"{file_key}_ann_relu")
    print("\nTraining ANN Topology 2...")
    ann_model_2, _ = train_ann(X_train, y_train, X_val, y_val, activation='gelu', model_name=f"{file_key}_ann_gelu")

    # Evaluate Classifiers
    results = train_classifiers(X, y, file_key)
    print(f"Results for {file_key}: {results}")