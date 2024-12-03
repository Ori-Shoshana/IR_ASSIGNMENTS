import os
import pandas as pd
from gensim.models import Word2Vec
import numpy as np

# פונקציה לטעינת קובץ CSV
def load_csv(file_path):
    return pd.read_csv(file_path)

# פונקציה להכנת רשימת המילים לכל מסמך
def create_word_list_from_docs(df):
    return [doc.split() for doc in df["content"].dropna()]  # עמודת 'content' מכילה את הטקסט

# פונקציה ליצירת מודל Word2Vec
def train_word2vec(docs, vector_size=100, window=5, min_count=5, workers=4):
    return Word2Vec(sentences=docs, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

# פונקציה ליצירת וקטורים למסמכים על בסיס Word2Vec
def create_doc_vectors(docs, model):
    vectors = []
    for doc in docs:
        # קבלת וקטורים של מילים שנמצאות במודל
        word_vectors = [model.wv[word] for word in doc if word in model.wv]
        if word_vectors:  # אם יש מילים במסמך
            # מיצוע הווקטורים
            doc_vector = np.mean(word_vectors, axis=0)
        else:
            # אם אין מילים במסמך, נייצר וקטור אפס
            doc_vector = np.zeros(model.vector_size)
        vectors.append(doc_vector)
    return vectors


# פונקציה לשמירת הווקטורים כקובץ CSV
def save_vectors(vectors, output_path):
    # ממירים את רשימת הווקטורים למטריצה
    matrix = np.array(vectors)
    pd.DataFrame(matrix).to_csv(output_path, index=False)
    print(f"Saved vectors to {output_path}")

# מסלולים לתיקיות
cleaned_folder = "cleaned_data"
lemmatized_folder = "lemmatized_data"
output_folder_cleaned = "vectors_cleaned"
output_folder_lemmatized = "vectors_lemmatized"

# יצירת תיקיות פלט
os.makedirs(output_folder_cleaned, exist_ok=True)
os.makedirs(output_folder_lemmatized, exist_ok=True)

# עיבוד תיקיות
for folder, output_folder in [(cleaned_folder, output_folder_cleaned), (lemmatized_folder, output_folder_lemmatized)]:
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if file_name.endswith(".csv"):
            print(f"Processing {file_name}...")
            
            # טעינת הקובץ
            df = load_csv(file_path)
            docs = create_word_list_from_docs(df)

            # אימון מודל Word2Vec
            model = train_word2vec(docs)

            # יצירת וקטורים
            vectors = create_doc_vectors(docs, model)
            
            # שמירת הווקטורים
            output_path = os.path.join(output_folder, f"{file_name.split('.')[0]}_vectors.csv")
            save_vectors(vectors, output_path)
            print(f"Finished processing {file_name}.\n")
