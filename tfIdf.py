import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Mapping file names to the relevant content column
files_and_columns = {
    "A-J_cleaned": "content",
    "BBC_cleaned": "content",
    "J-P_cleaned": "content",
    "NY-T_cleaned": "content",
    "A-J_lemmatized": "content",
    "BBC_lemmatized": "content",
    "J-P_lemmatized": "content",
    "NY-T_lemmatized": "content",
}

# Folders containing the data
cleaned_folder = "cleaned_data"
lemmatized_folder = "lemmatized_data"
output_folder = "tfidf_output"

os.makedirs(output_folder, exist_ok=True)

def process_documents(file_path, column_name):
    """
    Loads documents from a CSV file and returns a list of texts.
    """
    df = pd.read_csv(file_path)
    documents = df[column_name].dropna().tolist()  # Remove empty rows
    return documents

def calculate_tfidf(documents):
    """
    Computes the TF-IDF matrix for the documents.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=5  # Filter words appearing less than 5 times
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer.get_feature_names_out()

def save_tfidf_matrix(output_path, matrix, feature_names):
    """
    Saves the TF-IDF matrix to a CSV file with numeric values.
    """
    # Convert the sparse matrix to a DataFrame
    df = pd.DataFrame(matrix.toarray(), columns=feature_names)
    df.to_csv(output_path, index=False)

def process_all_files():
    """
    Processes all files to generate TF-IDF matrices.
    """
    for file_name, column_name in files_and_columns.items():
        if "cleaned" in file_name:
            folder = cleaned_folder
            suffix = "TFIDF-Word"
        else:
            folder = lemmatized_folder
            suffix = "TFIDF-Lemm"

        file_path = os.path.join(folder, f"{file_name}.csv")
        output_path_tfidf = os.path.join(output_folder, f"{file_name}_{suffix}_tfidf.csv")
        
        print(f"Processing {file_name}...")

        # Load documents
        documents = process_documents(file_path, column_name)

        # Compute TF-IDF
        tfidf_matrix, tfidf_features = calculate_tfidf(documents)

        # Save the resulting matrix
        save_tfidf_matrix(output_path_tfidf, tfidf_matrix, tfidf_features)

        print(f"Saved TF-IDF matrix for {file_name} to {output_path_tfidf}")

# Execute the script
process_all_files()
