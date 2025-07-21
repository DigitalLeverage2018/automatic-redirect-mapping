import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import tiktoken

# --- Titel & API-Key ---
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

# --- URL-Spalten prüfen und normalisieren ---
for df in [df_old, df_new]:
    if "URL" not in df.columns:
        st.error("❌ Beide CSV-Dateien müssen eine Spalte 'URL' enthalten.")
        st.stop()
    df["URL"] = df["URL"].astype(str).str.strip()

# --- ALT: Doppelte alte URLs entfernen ---
initial_count = len(df_old)
df_old = df_old.drop_duplicates(subset="URL", keep="first").reset_index(drop=True)
removed = initial_count - len(df_old)
if removed > 0:
    st.info(f"🔁 {removed} doppelte alte URLs entfernt – nur erste Vorkommen berücksichtigt.")

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

        # Onpage Matching
        "H1 Match": "Exact Match" if row_old.get("H1", "").strip().lower() == row_new.get("H1", "").strip().lower() else (
            "Partial Match" if row_old.get("H1", "").strip().lower() in row_new.get("H1", "").strip().lower() or row_new.get("H1", "").strip().lower() in row_old.get("H1", "").strip().lower() else "No Match"
        ),
        "Title Tag Match": "Exact Match" if row_old.get("Title Tag", "").strip().lower() == row_new.get("Title Tag", "").strip().lower() else (
            "Partial Match" if row_old.get("Title Tag", "").strip().lower() in row_new.get("Title Tag", "").strip().lower() or row_new.get("Title Tag", "").strip().lower() in row_old.get("Title Tag", "").strip().lower() else "No Match"
        ),
        "Meta Description Match": "Exact Match" if row_old.get("Meta Description", "").strip().lower() == row_new.get("Meta Description", "").strip().lower() else (
            "Partial Match" if row_old.get("Meta Description", "").strip().lower() in row_new.get("Meta Description", "").strip().lower() or row_new.get("Meta Description", "").strip().lower() in row_old.get("Meta Description", "").strip().lower() else "No Match"
        ),

        # ALT-Infos
        "Status Code ALT": row_old.get("Status code", ""),
        "H1 ALT": row_old.get("H1", ""),
        "Title Tag ALT": row_old.get("Title Tag", ""),
        "Meta Description ALT": row_old.get("Meta Description", ""),
        "Body Content ALT": row_old.get("Body Content", ""),
        "Klicks": row_old.get("Klicks", ""),
        "Backlinks": row_old.get("Backlinks", ""),

        # NEU-Infos
        "Status Code NEU": row_new.get("Status Code", ""),
        "H1 NEU": row_new.get("H1", ""),
        "Title Tag NEU": row_new.get("Title Tag", ""),
        "Meta Description NEU": row_new.get("Meta Description", ""),
        "Body Content NEU": row_new.get("Body Content", "")
    }

    results.append(match_details)

# --- Ergebnis sichern & validieren ---
result_df = pd.DataFrame(results)

# Nur 1 Zeile pro alte URL
before_dedup = len(result_df)
result_df = result_df.drop_duplicates(subset="Old URL", keep="first")
after_dedup = len(result_df)
if before_dedup > after_dedup:
    st.warning(f"⚠️ {before_dedup - after_dedup} doppelte alte URLs im Ergebnis entfernt – pro alte URL nur ein Redirect.")

# --- Ergebnis anzeigen ---
st.success("✅ Mapping abgeschlossen")
st.dataframe(result_df)

# --- Download ---
csv = result_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Ergebnis als CSV herunterladen", data=csv, file_name="redirect_mapping.csv", mime="text/csv")
