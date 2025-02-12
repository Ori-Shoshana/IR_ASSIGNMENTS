# IR_ASSIGNMENTS_1

# Text Embedding and Processing Pipeline

---

**Participants**:  
- **Name**: אמיר חי תמר | **ID**: 322879339  
- **Name**: אורי שושנה | **ID**: 324891134  

**Group Number**: 157125.2.5785

---

## Overview

This project involves processing raw text data from newspapers to generate meaningful document embeddings using various natural language processing (NLP) techniques. The embeddings are used to represent the semantic content of articles effectively.

---

## Workflow

### Data Preprocessing
- **Script**: `main.py`
- Cleans raw text data by:
  - Removing special characters and punctuation.
  - Normalizing contractions.
  - Lemmatizing using `spaCy`.
- Outputs:
  - Cleaned data (`cleaned/`).
  - Lemmatized data (`lemmatized/`).

---

## TF-IDF Calculation for Text Documents
### Description

The script `tfidf.py` computes the **TF-IDF (Term Frequency-Inverse Document Frequency)** matrices for text documents from both cleaned and lemmatized datasets. It generates eight separate CSV files as output, corresponding to the following:

1. **Cleaned Data**: TF-IDF matrices for articles processed without lemmatization.
2. **Lemmatized Data**: TF-IDF matrices for articles processed with lemmatization.

Each dataset contains one file for each newspaper: `A-J`, `BBC`, `J-P`, and `NY-T`.

---

### Key Functionality

1. **Mapping Input Files**:
   - The script reads a dictionary of file names and their corresponding text columns.

2. **Document Processing**:
   - Reads CSV files containing raw text data.
   - Cleans and removes empty entries.

3. **TF-IDF Matrix Calculation**:
   - Computes the TF-IDF values for each document using scikit-learn's `TfidfVectorizer`.
   - Filters out terms that appear in fewer than 5 documents.

4. **Saving Results**:
   - Converts the sparse matrix to a dense representation and saves it as a CSV file.

---

### Outputs

The script generates **8 CSV files**, stored in the `tfidf_output/` folder:

1. **Cleaned Data TF-IDF Matrices**:
   - `A-J_cleaned_TFIDF-Word_tfidf.csv`
   - `BBC_cleaned_TFIDF-Word_tfidf.csv`
   - `J-P_cleaned_TFIDF-Word_tfidf.csv`
   - `NY-T_cleaned_TFIDF-Word_tfidf.csv`

2. **Lemmatized Data TF-IDF Matrices**:
   - `A-J_lemmatized_TFIDF-Lemm_tfidf.csv`
   - `BBC_lemmatized_TFIDF-Lemm_tfidf.csv`
   - `J-P_lemmatized_TFIDF-Lemm_tfidf.csv`
   - `NY-T_lemmatized_TFIDF-Lemm_tfidf.csv`


### 2. Text Representation Techniques
#### a. **Word2Vec**
- **Script**: `w2v.py`
- Generates document embeddings by averaging word vectors trained using Gensim's Word2Vec.

#### b. **Doc2Vec**
- **Script**: `doc2vec.py`
- Generates document embeddings using Gensim's Doc2Vec, where each document is represented as a unique vector.

#### c. **BERT**
- **Script**: `bert.py`
- Extracts contextual embeddings for documents using Hugging Face's `BERT`.

#### d. **SBERT**
- **Script**: `sbert.py`
- Uses Sentence-BERT to encode documents into sentence-level vectors.

### 3. Outputs
Each text representation method produces vectors stored as CSV files in the respective subfolders.

---

## Feature Importance Metrics: Information Gain and Chi-Squared Statistic

### Description

The script `feature_metrics.py` computes feature importance metrics for terms in the TF-IDF matrices generated earlier. It calculates:

1. **Information Gain (IG)**: Measures the relevance of features (terms) to the classification task.
2. **Chi-Squared Statistic (Chi²)**: Evaluates the independence between features and target labels.

The output is stored as Excel files containing the metrics for each feature in the TF-IDF matrices.

---

### Key Functionality

1. **Feature Importance Metrics**:
   - **Information Gain**:
     - Uses scikit-learn's `mutual_info_classif` to compute the relevance of features.
   - **Chi-Squared Statistic**:
     - Uses scikit-learn's `chi2` to compute feature independence scores.

2. **Results Storage**:
   - Saves the results to an Excel file with two sheets:
     - `Information Gain`: Features ranked by their information gain values.
     - `Chi-Squared`: Features ranked by their Chi-squared statistic values and associated p-values.

---

### Outputs

The script generates **8 Excel files**, stored in the `chi2-info/` folder:

1. **Cleaned Data Metrics**:
   - `A-J_cleaned_TFIDF-Word_tfidf_metrics.xlsx`
   - `BBC_cleaned_TFIDF-Word_tfidf_metrics.xlsx`
   - `J-P_cleaned_TFIDF-Word_tfidf_metrics.xlsx`
   - `NY-T_cleaned_TFIDF-Word_tfidf_metrics.xlsx`

2. **Lemmatized Data Metrics**:
   - `A-J_lemmatized_TFIDF-Lemm_tfidf_metrics.xlsx`
   - `BBC_lemmatized_TFIDF-Lemm_tfidf_metrics.xlsx`
   - `J-P_lemmatized_TFIDF-Lemm_tfidf_metrics.xlsx`
   - `NY-T_lemmatized_TFIDF-Lemm_tfidf_metrics.xlsx`

Each file contains two sheets:
- **Information Gain**: Features ranked by their relevance to the classification task.
- **Chi-Squared**: Features ranked by their statistical significance.

---

## Technologies Used

- **Python Libraries**:
  - `pandas` and `numpy` for data manipulation.
  - `spaCy` for lemmatization.
  - `gensim` for Word2Vec and Doc2Vec models.
  - `transformers` for BERT embeddings.
  - `sentence-transformers` for SBERT embeddings.
- **File Format**:
  - Input: Excel with multiple sheets.
  - Output: CSV files for processed text and embeddings.

---

## Results and Outputs

### Embedding Sizes:
1. **Word2Vec**: 100-dimensional vectors (default configuration).
2. **Doc2Vec**: 300-dimensional vectors.
3. **BERT**: 768-dimensional vectors.
4. **SBERT**: 384-dimensional vectors.

### Summary of Outputs:
Each method generates:
- Four CSV files (one per newspaper) containing document embeddings.

---

## Challenges and Insights

### Challenges
1. **Handling Missing Data**:
   - Many articles were incomplete or empty, leading to fewer processed entries than expected.
2. **Large Embedding Sizes**:
   - Especially with BERT and SBERT, storage and computation required optimization.
3. **Noise in Raw Data**:
   - Articles contained inconsistent formatting, requiring extensive cleaning.

### Insights
1. **Effectiveness of Techniques**:
   - **SBERT** showed significant performance in semantic similarity tasks due to its sentence-level context awareness.
   - **Doc2Vec** provided a compact and interpretable embedding space.
2. **Preprocessing Matters**:
   - The quality of embeddings is highly dependent on the preprocessing stage, especially for Word2Vec.

---
