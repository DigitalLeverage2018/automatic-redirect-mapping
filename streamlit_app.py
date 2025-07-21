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

# --- Benutzer-Einstellungen ---
st.header("⚙️ Einstellungen für Vergleich & Matching")

suffix_alt = st.text_input("🔧 Title Tag Suffix ALT (z. B. ' | Beispiel AG')", value="")
suffix_neu = st.text_input("🔧 Title Tag Suffix NEU (z. B. ' | Neue Firma')", value="")

min_similarity_threshold = st.slider(
    "🔒 Minimale Ähnlichkeit (Similarity Score), damit eine Weiterleitung empfohlen wird",
    min_value=0.0,
    max_value=1.0,
    value=0.40,
    step=0.01,
    help=(
        "Wenn keine Onpage-Elemente (H1, Title Tag, Meta Description) übereinstimmen, "
        "wird diese Schwelle verwendet, um zu entscheiden, ob ein Redirect auf Basis der inhaltlichen Ähnlichkeit zulässig ist."
    )
)

# --- Datei Upload ---
st.header("📁 CSV-Dateien hochladen")
uploaded_old = st.file_uploader("1️⃣ Crawl ALT (mit Klicks, Backlinks etc.)", type="csv")
uploaded_new = st.file_uploader("2️⃣ Crawl NEU (z. B. Staging)", type="csv")

if not uploaded_old or not uploaded_new:
    st.stop()

df_old = pd.read_csv(uploaded_old)
df_new = pd.read_csv(uploaded_new)

for df in [df_old, df_new]:
    if "URL" not in df.columns:
        st.error("❌ Beide CSV-Dateien müssen eine Spalte 'URL' enthalten.")
        st.stop()
    df["URL"] = df["URL"].astype(str).str.strip()

# Duplikate ALT entfernen
initial_count = len(df_old)
df_old = df_old.drop_duplicates(subset="URL", keep="first").reset_index(drop=True)
removed = initial_count - len(df_old)
if removed > 0:
    st.info(f"🔁 {removed} doppelte alte URLs entfernt – nur erste Vorkommen berücksichtigt.")

df_old.fillna("", inplace=True)
df_new.fillna("", inplace=True)

encoding = tiktoken.get_encoding("cl100k_base")

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")[:8000]
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def clean_title(title, suffix):
    return str(title).replace(suffix, "").strip().lower()

def normalize(text):
    return str(text).strip().lower()

def combine_text(row, fields):
    return " ".join(str(row.get(f, "")) for f in fields)

st.info("🔍 Embeddings werden erstellt...")

df_old["embedding"] = df_old.apply(lambda row: get_embedding(combine_text(row, ['H1', 'Title Tag', 'Meta Description', 'Body Content'])), axis=1)
df_new["embedding"] = df_new.apply(lambda row: get_embedding(combine_text(row, ['H1', 'Title Tag', 'Meta Description', 'Body Content'])), axis=1)

embeddings_old = np.vstack(df_old["embedding"].to_numpy())
embeddings_new = np.vstack(df_new["embedding"].to_numpy())
similarity_matrix = cosine_similarity(embeddings_old, embeddings_new)

# Matching
results = []
for idx_old, row_old in df_old.iterrows():
    best_idx = np.argmax(similarity_matrix[idx_old])
    similarity_score = similarity_matrix[idx_old][best_idx]
    row_new = df_new.iloc[best_idx]

    # Normalisierte Vergleiche
    h1_match = normalize(row_old.get("H1", "")) == normalize(row_new.get("H1", ""))
    title_match = clean_title(row_old.get("Title Tag", ""), suffix_alt) == clean_title(row_new.get("Title Tag", ""), suffix_neu)
    meta_match = normalize(row_old.get("Meta Description", "")) == normalize(row_new.get("Meta Description", ""))

    match_status = "No Match"
    confidence = "low"
    matched = False

    # Onpage-Matching hat Priorität
    match_count = sum([h1_match, title_match, meta_match])
    if match_count == 3:
        match_status = "Match"
        confidence = "very likely"
        matched = True
    elif match_count == 2:
        match_status = "Match"
        confidence = "likely"
        matched = True
    elif match_count == 1 and (h1_match or title_match):
        match_status = "Match"
        confidence = "possible"
        matched = True
    elif not matched:
        if similarity_score >= 0.90:
            match_status = "Match"
            confidence = "likely"
        elif similarity_score >= 0.80:
            match_status = "Match"
            confidence = "possible"
        elif similarity_score >= min_similarity_threshold:
            match_status = "Match"
            confidence = "low"
        else:
            match_status = "No Match"
            confidence = "low"
            row_new = {}  # leere Daten
            best_idx = None

    result = {
        "Old URL": row_old.get("URL", ""),
        "New URL": row_new.get("URL", "") if best_idx is not None else "",
        "Similarity Score": round(similarity_score, 4) if best_idx is not None else "",
        "Match": match_status,
        "Confidence Score": confidence,

        # Onpage Details
        "H1 Match": h1_match,
        "Title Tag Match": title_match,
        "Meta Description Match": meta_match,

        # ALT Infos
        "Status Code ALT": row_old.get("Status code", ""),
        "H1 ALT": row_old.get("H1", ""),
        "Title Tag ALT": row_old.get("Title Tag", ""),
        "Meta Description ALT": row_old.get("Meta Description", ""),
        "Body Content ALT": row_old.get("Body Content", ""),
        "Klicks": row_old.get("Klicks", ""),
        "Backlinks": row_old.get("Backlinks", ""),

        # NEU Infos
        "Status Code NEU": row_new.get("Status Code", "") if best_idx is not None else "",
        "H1 NEU": row_new.get("H1", "") if best_idx is not None else "",
        "Title Tag NEU": row_new.get("Title Tag", "") if best_idx is not None else "",
        "Meta Description NEU": row_new.get("Meta Description", "") if best_idx is not None else "",
        "Body Content NEU": row_new.get("Body Content", "") if best_idx is not None else "",
    }

    results.append(result)

result_df = pd.DataFrame(results)

# Anzeige & Download
st.success("✅ Matching abgeschlossen")
st.dataframe(result_df)

csv = result_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Ergebnis als CSV herunterladen", data=csv, file_name="redirect_mapping.c_
