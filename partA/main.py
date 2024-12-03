import os
import pandas as pd
import re
import spacy
# טוען את המודל של spaCy
nlp = spacy.load('en_core_web_sm')

def save_cleaned_dataframes(dataframes, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, (df, sheet_name) in enumerate(zip(dataframes, ["A-J", "BBC", "J-P", "NY-T"]), start=1):
        # שמירה של קובץ מנוקה
        output_path_cleaned = os.path.join(output_folder, f"{sheet_name}_cleaned.csv")
        # שמירה של קובץ לממטיזציה
        output_path_lemmatized = os.path.join(output_folder, f"{sheet_name}_lemmatized.csv")

        # יצירת רשימת נתונים עם עמודות 'article' ו-'content'
        cleaned_data = []
        lemmatized_data = []

        for i, article in enumerate(df["article"], start=1):
            # ניקוי הטקסט
            cleaned_content = clean_text(article)
            # לממטיזציה של הטקסט
            lemmatized_content = lemmatize_text(cleaned_content)

            # הוספת הנתונים לשתי הרשימות
            cleaned_data.append({"article": f"{sheet_name} {i}", "content": cleaned_content})
            lemmatized_data.append({"article": f"{sheet_name} {i}", "content": lemmatized_content})

        # יצירת DataFrame חדש מהנתונים המנוקים
        cleaned_df = pd.DataFrame(cleaned_data)
        lemmatized_df = pd.DataFrame(lemmatized_data)

        # שמירת ה-DataFrames כקובצי CSV
        cleaned_df.to_csv(output_path_cleaned, index=False)
        lemmatized_df.to_csv(output_path_lemmatized, index=False)
        print(f"Saved cleaned data for {sheet_name} to {output_path_cleaned}")
        print(f"Saved lemmatized data for {sheet_name} to {output_path_lemmatized}")


def clean_text(text: str) -> str:
    # להחליף גרשיים מעוגלים וגרשיים כפולים לגרשיים ישרים
    text = text.replace('‘', "'").replace('’', "'").replace('“', '"').replace('”', '"')

    # ביטוי רגולרי שמחלק את הטקסט למילים ולסימני פיסוק, תוך שמירה על קונטרקציות
    words = re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]", text)

    # להרכיב את המילים בחזרה למיתר טקסט אחד, על ידי חיבורן עם רווח
    return ' '.join(words)

def lemmatize_text(text: str) -> str:
    # עיבוד הטקסט עם spaCy
    doc = nlp(text)

    # חילוץ השורשים (lemmatized tokens) וסינון פיסוק ומספרים
    lemmatized_words = [token.lemma_ for token in doc if not token.is_punct and not token.like_num]

    # חיבור המילים לשורש בחזרה לטקסט אחד
    lemmatized_text = ' '.join(lemmatized_words)

    # הסרת כל מופעי "'s"
    lemmatized_text = lemmatized_text.replace("'s", "")

    return lemmatized_text


# קריאה ועיבוד הנתונים
def get_data_from_excel(path: str):
    # שמות הגליונות שיש לעבד בקובץ האקסל
    sheets = ["A-J", "BBC", "J-P", "NY-T"]

    # מילון לאחסון הנתונים המעובדים מכל גיליון
    dataframes = []
    for sheet in sheets:
        # קריאת הגיליון הנוכחי ל-DataFrame
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        # עיבוד הגיליון בעזרת הפונקציה הכללית
        df = process_articles(df, sheet)
        dataframes.append(df)

    # החזרת כל הגליונות המעובדים
    return dataframes

def process_articles(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    # מיפוי שמות גיליונות לעמודות הרלוונטיות שיש לעבד בהם
    column_mapping = {
        "A-J": ['title', 'sub_title', 'Body Text'],
        "BBC": ['title', 'Body Text'],
        "J-P": ['title', 'Body'],
        "NY-T": ['title', 'Body Text']
    }

    # וידוא שהגיליון קיים במיפוי
    if sheet_name not in column_mapping:
        raise ValueError(f"Unknown sheet name: {sheet_name}")

    # שליפת העמודות הרלוונטיות לגיליון הנוכחי
    col_names = column_mapping[sheet_name]

    # שמירה רק על העמודות הרלוונטיות והחלפת ערכים חסרים בטקסט ריק
    df = df[col_names].fillna("")

    # יצירת טור חדש בשם 'article' שמאחד את כל הטקסטים מהעמודות הרלוונטיות
    df["article"] = df[col_names].agg(" ".join, axis=1)

    # ניקוי הטקסט בעמודת 'article'
    df["article"] = df["article"].apply(clean_text)

    return df


# בדיקה אם הקובץ קיים כדי להימנע מבעיית קריאה
excel_path = 'posts_first_targil.xlsx'
if not os.path.exists(excel_path):
    raise FileNotFoundError(f"The file {excel_path} was not found.")

# קריאת הקובץ ועיבוד הנתונים
df_aj, df_bbc, df_jp, df_nyt = get_data_from_excel(excel_path)

# שמירה של הנתונים המעובדים לכל גיליון כקובץ CSV נפרד
output_folder = 'processed_data'
save_cleaned_dataframes((df_aj, df_bbc, df_jp, df_nyt), output_folder)
