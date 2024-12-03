# IR_ASSIGNMENTS

# Text Embedding and Processing Pipeline

---

**Participants**:  
- **Name**: אמיר חי תמר | **ID**: 322879339  
- **Name**: אורי שושנה | **ID**: 324891137  

**Group Number**: 157125.2.5785

---

## Overview

This project involves processing raw text data from newspapers to generate meaningful document embeddings using various natural language processing (NLP) techniques. The embeddings are used to represent the semantic content of articles effectively.

---

## Workflow

---

## Workflow

### 1. Data Preprocessing
- **Script**: `main.py`
- Cleans raw text data by:
  - Removing special characters and punctuation.
  - Normalizing contractions.
  - Lemmatizing using `spaCy`.
- Outputs:
  - Cleaned data (`cleaned/`).
  - Lemmatized data (`lemmatized/`).

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
Each text representation method produces vectors stored as CSV files in the respective subfolders under `vectors/`.

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

## How to Run

1. Place the raw Excel file (`posts_first_targil.xlsx`).
2. Run the preprocessing script:
   ```bash
   python scripts/main.py
3. and then run each file..
