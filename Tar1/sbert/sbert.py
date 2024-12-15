import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# קביעת המודל של SBERT
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# מיפוי הגיליונות לעמודות התוכן המרכזיות
COLUMN_MAPPING = {
    "A-J": "Body Text",
    "BBC": "Body Text",
    "J-P": "Body",
    "NY-T": "Body Text"
}

def extract_body_text_and_vectorize(sheet_name, df):
    """
    שולף את עמודת התוכן המרכזית וממיר אותה לוקטורים באמצעות SBERT
    """
    if sheet_name not in COLUMN_MAPPING:
        raise ValueError(f"Unknown sheet name: {sheet_name}")
    
    column_name = COLUMN_MAPPING[sheet_name]
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in sheet {sheet_name}")
    
    # חילוץ התוכן המרכזי
    body_texts = df[column_name].fillna("").tolist()
    
    # יצירת וקטורים
    vectors = sbert_model.encode(body_texts, convert_to_tensor=False)  # נשתמש ב-numpy
    return vectors

def process_excel_file_to_sbert_vectors(excel_path, output_folder):
    """
    מעבד את הקובץ ומייצר וקטורים לכל גיליון
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sheets = COLUMN_MAPPING.keys()
    for sheet_name in sheets:
        print(f"Processing {sheet_name}...")
        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
        
        # יצירת הווקטורים
        vectors = extract_body_text_and_vectorize(sheet_name, df)
        
        # שמירת הווקטורים לקובץ CSV
        output_file = os.path.join(output_folder, f"{sheet_name}_SBERT_vectors.csv")
        pd.DataFrame(vectors).to_csv(output_file, index=False)
        print(f"Saved vectors for {sheet_name} to {output_file}")

# קריאה וביצוע
excel_path = "posts_first_targil.xlsx"
output_folder = "SBERT_vectors"

process_excel_file_to_sbert_vectors(excel_path, output_folder)

