import os
import pandas as pd

# תיקיות הקלט
cleaned_folder = "feature_metrics/cleaned"
lemm_folder = "feature_metrics/lemmatized"

# תיקיית הפלט
output_folder = "feature_metrics/top_features"
os.makedirs(output_folder, exist_ok=True)

def extract_top_features(file_path, output_path, n=20):
    """
    Extracts the top N features with the highest values from a CSV file.
    """
    # טוען את הנתונים
    df = pd.read_csv(file_path)

    # מוודא שהעמודה 'information_gain' קיימת
    if 'information_gain' in df.columns:
        top_features = df.nlargest(n, 'information_gain')
    elif 'chi2_statistic' in df.columns:
        top_features = df.nlargest(n, 'chi2_statistic')
    else:
        raise ValueError(f"Missing relevant columns in {file_path}")
    
    # שומר את התוצאות
    top_features.to_csv(output_path, index=False)
    print(f"Saved top {n} features to {output_path}")

def process_folder(input_folder, output_folder, n=20):
    """
    Processes all CSV files in a folder to extract the top N features.
    """
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"top_{n}_{file_name}")
            extract_top_features(file_path, output_path, n)

# עיבוד הקבצים בתיקיות
print("Processing cleaned features...")
process_folder(cleaned_folder, output_folder)

print("Processing lemmatized features...")
process_folder(lemm_folder, output_folder)
