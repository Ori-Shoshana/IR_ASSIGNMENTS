import pandas as pd
import os
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import LabelEncoder

# פונקציה לחישוב Information Gain
def calculate_information_gain(tfidf_matrix, labels):
    ig = mutual_info_classif(tfidf_matrix, labels)
    return ig

# פונקציה לחישוב Chi-squared statistic
def calculate_chi2_statistic(tfidf_matrix, labels):
    chi2_stat, p_val = chi2(tfidf_matrix, labels)
    return chi2_stat, p_val

# פונקציה לשמירת התוצאות לאקסל
def save_to_excel(output_path, ig_df, chi2_df, file_name):
    # קיצור שם הגיליון אם הוא ארוך מדי
    ig_sheet_name = f"{file_name}_IG"[:31]
    chi2_sheet_name = f"{file_name}_Chi2"[:31]
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        ig_df.to_excel(writer, sheet_name=ig_sheet_name, index=False)
        chi2_df.to_excel(writer, sheet_name=chi2_sheet_name, index=False)

# פונקציה לעיבוד כל הקבצים
def process_tfidf_and_calculate_metrics():
    output_folder = "tfidf_output"
    results_folder = "tfidf_results"
    os.makedirs(results_folder, exist_ok=True)
    
    # מעבד את כל הקבצים בתיקיית ה-output
    for file_name in os.listdir(output_folder):
        if file_name.endswith(".csv"):
            print(f"Processing {file_name}...")
            file_path = os.path.join(output_folder, file_name)
            df = pd.read_csv(file_path)
            
            # מטריצת TF-IDF
            tfidf_matrix = df.drop(columns=['document_name']).values
            feature_names = df.drop(columns=['document_name']).columns
            
            # יצירת תוויות לדוגמה (כמובן יש להחליף בתוויות אמיתיות אם יש)
            labels = [1 if i % 2 == 0 else 0 for i in range(len(df))]  # יצירת תוויות באופן אקראי
            
            # חישוב Information Gain
            ig = calculate_information_gain(tfidf_matrix, labels)
            ig_df = pd.DataFrame({
                'feature': feature_names,
                'information_gain': ig
            })
            
            # חישוב Chi-squared statistic
            chi2_stat, p_val = calculate_chi2_statistic(tfidf_matrix, labels)
            chi2_df = pd.DataFrame({
                'feature': feature_names,
                'chi2_statistic': chi2_stat,
                'p_value': p_val
            })
            
            # שמירה לקובץ Excel
            output_path = os.path.join(results_folder, f"{file_name}_metrics.xlsx")
            save_to_excel(output_path, ig_df, chi2_df, file_name)
            print(f"Saved metrics for {file_name} to {output_path}")

# הרצת הפונקציה
process_tfidf_and_calculate_metrics()
