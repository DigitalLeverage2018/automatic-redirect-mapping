import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import tiktoken

# --- API Key Input ---
st.title("🔁 Automatische Redirect-Mappings erstellen")
api_key = st.text_input("🔑 OpenAI API Key eingeben", type="password")
if not api_key:
    st.stop()
client = openai.OpenAI(api_key=api_key)

# --- CSV Upload ---
st.header("📁 CSV-Dateien hochladen")
uploaded_old = st.file_uploader("1️⃣ Crawl ALT (mit Klicks, Backlinks etc.)", type="csv")
uploaded_new = st.file_uploader("2️⃣ Crawl NEU (z.B. Staging-Umgebung)", type="csv")

if not uploaded_old or not uploaded_new:
    st.stop()

df_old = pd.read_csv(uploaded_old)
df_new = pd.read_csv(uploaded_new)

# --- Vorbereitung ---
df_old.fillna("", inplace=True)
df_new.fillna("", inplace=True)

text_cols_old = ['H1', 'Title Tag', 'Meta Description', 'Body Content']
text_cols_new = ['H1', 'Title Tag', 'Meta Description', 'Body Content']

encoding = tiktoken.get_encoding("cl100k_base")

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")[:8000]
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def combine_text(row, cols):
    return " ".join(str(row[col]) for col in cols if col in row and pd.notnull(row[col]))

st.info("🔍 Embeddings werden erstellt...")

df_old["embedding"] = df_old.apply(lambda row: get_embedding(combine_text(row, text_cols_old)), axis=1)
df_new["embedding"] = df_new.apply(lambda row: get_embedding(combine_text(row, text_cols_new)), axis=1)

embeddings_old = np.vstack(df_old["embedding"].to_numpy())
embeddings_new = np.vstack(df_new["embedding"].to_numpy())

similarity_matrix = cosine_similarity(embeddings_old, embeddings_new)

# --- Matching ---
results = []
for idx_old, row_old in df_old.iterrows():
    best_idx = np.argmax(similarity_matrix[idx_old])
    similarity_score = similarity_matrix[idx_old][best_idx]
    row_new = df_new.iloc[best_idx]

    match_details = {
        "Old URL": row_old.get('URL', ""),
        "New URL": row_new.get('URL', ""),
        "Similarity Score": round(similarity_score, 4),
    }

    # Onpage Element Vergleich
    for field in ['H1', 'Title Tag', 'Meta Description']:
        val_old = str(row_old.get(field, "")).strip().lower()
        val_new = str(row_new.get(field, "")).strip().lower()
        if val_old and val_new:
            if val_old == val_new:
                match_details[f"{field} Match"] = "Exact Match"
            elif val_old in val_new or val_new in val_old:
                match_details[f"{field} Match"] = "Partial Match"
            else:
                match_details[f"{field} Match"] = "No Match"
        else:
            match_details[f"{field} Match"] = "Missing"

    results.append(match_details)

# --- Ergebnis anzeigen ---
result_df = pd.DataFrame(results)
st.success("✅ Mapping abgeschlossen")
st.dataframe(result_df)

csv = result_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Ergebnis als CSV herunterladen", data=csv, file_name="redirect_mapping.csv", mime="text/csv")
