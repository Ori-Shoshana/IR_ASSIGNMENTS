import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

# פונקציה לחילוץ וקטור BERT עבור טקסט
def get_bert_vector(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    with torch.no_grad():
        outputs = model(**tokens)
        # לוקחים את הווקטור של CLS
        cls_vector = outputs.last_hidden_state[:, 0, :]
    return cls_vector.squeeze().tolist()

# קריאת כל הגיליונות מקובץ Excel
file_path = 'posts_first_targil.xlsx' 
sheets = pd.ExcelFile(file_path)

# אתחול המודל והטוקנייזר של BERT
model_name = "bert-base-uncased"  
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# עיבוד כל גיליון כעיתון נפרד
bert_vectors = {}
for sheet_name in sheets.sheet_names:
    # קריאת הגיליון הנוכחי
    data = sheets.parse(sheet_name)
    
    # בדיקה אם עמודת "Body Text" קיימת
    if 'Body Text' not in data.columns:#בישביל J-P שעמודת הטקטס שלו בכותרת שונה
        documents = data['Body'].dropna()

        vectors = []
        for doc in documents:
            vector = get_bert_vector(doc, tokenizer, model)
            vectors.append(vector)
        
    else:
        documents = data['Body Text'].dropna()  
        # חילוץ וקטורים עם BERT
        vectors = []
        for doc in documents:
            vector = get_bert_vector(doc, tokenizer, model)
            vectors.append(vector)

    # שמירת וקטורים כקובץ CSV עבור כל גיליון
    vectors_df = pd.DataFrame(vectors)
    vectors_df.to_csv(f'bert_vectors_{sheet_name}.csv', index=False)

    # שמירת הווקטורים עבור המטריצה המשולבת
    bert_vectors[sheet_name] = vectors

# יצירת מטריצה משולבת לכל הגיליונות
combined_vectors = []
for sheet_name, vectors in bert_vectors.items():
    combined_vectors.extend(vectors)

# שמירת הקובץ המשולב
combined_vectors_df = pd.DataFrame(combined_vectors)
combined_vectors_df.to_csv('bert_combined_matrix.csv', index=False)
