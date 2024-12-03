import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re

# פונקציה לטוקניזציה פשוטה
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# קריאת כל הגיליונות מקובץ Excel
file_path = 'posts_first_targil.xlsx'  # שנה לשם הקובץ שלך
sheets = pd.ExcelFile(file_path)

# עיבוד כל גיליון כעיתון נפרד
doc2vec_models = {}
for sheet_name in sheets.sheet_names:
    # קריאת הגיליון הנוכחי
    data = sheets.parse(sheet_name)
    
    # בדיקה אם עמודת "Body Text" קיימת
    if 'Body Text' not in data.columns:#בישביל J-P שעמודת הטקטס שלו בכותרת שונה
        documents = data['Body'].dropna()

        # יצירת TaggedDocument
        tagged_documents = [
            TaggedDocument(words=simple_tokenize(doc), tags=[f"{sheet_name}_{i}"])
            for i, doc in enumerate(documents)
        ]
    else:
        # שימוש רק בטקסט של המאמרים
        documents = data['Body Text'].dropna()

        # יצירת TaggedDocument
        tagged_documents = [
            TaggedDocument(words=simple_tokenize(doc), tags=[f"{sheet_name}_{i}"])
            for i, doc in enumerate(documents)
        ]

    # בניית מודל Doc2Vec עבור הגיליון הנוכחי
    model = Doc2Vec(vector_size=300, min_count=2, epochs=40, workers=4)
    model.build_vocab(tagged_documents)
    model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)
    doc2vec_models[sheet_name] = model

    # חילוץ וקטורים ואיחוד לשדה יחיד
    vectors = pd.Series([list(model.dv[i]) for i in range(len(tagged_documents))], name='vector')
    vectors = vectors.apply(lambda vec: ','.join(map(str, vec)))

    # שמירת הווקטורים כקובץ CSV עבור כל גיליון
    vectors.to_csv(f'doc2vec_{sheet_name}.csv', index=False)

# יצירת מטריצה משולבת לכל הגיליונות
combined_vectors = []
for sheet_name, model in doc2vec_models.items():
    combined_vectors.extend([list(model.dv[i]) for i in range(len(model.dv))])

# שמירת הקובץ המשולב
combined_vectors = pd.Series(combined_vectors, name='vector')
combined_vectors.to_csv('doc2vec_combined_matrix.csv', index=False)
