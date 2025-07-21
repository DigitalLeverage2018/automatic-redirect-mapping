import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import tiktoken
import re

# --- Titel & API-Key ---
st.title("🔁 Automatische Redirect-Mappings erstellen")

api_key = st.text_input("🔑 OpenAI API Key eingeben", type="password")
if not api_key:
    st.stop()
client = openai.OpenAI(api_key=api_key)

# --- Embedding-Modell wählen ---
st.header("🧠 Embedding-Modell wählen")
model_choice = st.selectbox(
    "Welches Modell möchtest du verwenden?",
    options=["text-embedding-3-small", "text-embedding-3-large"],
    format_func=lambda x: "Small (schneller, günstiger)" if "small" in x else "Large (exakter, teurer)"
)
st.markdown("""
**Small** = schneller, günstiger, weniger exakt – ideal bei über 100 URLs oder wenn Onpage-Elemente stabil bleiben  
**Large** = genauer, aber langsamer & teurer – ideal bei komplexen Redirections mit stark veränderten Seiten
""")

# --- Einstellungen ---
st.header("⚙️ Einstellungen für Vergleich & Matching")
suffix_alt = st.text_input("🔧 Title Tag Suffix ALT (z. B. ' | Beispiel AG')", value="")
suffix_neu = st.text_input("🔧 Title Tag Suffix NEU (z. B. ' | Neue Firma')", value="")

min_similarity_threshold = st.slider(
    "🔒 Minimale Ähnlichkeit (Similarity Score), damit eine Weiterleitung empfohlen wird",
    min_value=0.0, max_value=1.0, value=0.40, step=0.01
)

# --- Datei Upload ---
st.header("📁 CSV-Dateien hochladen")
uploaded_old = st.file_uploader("1️⃣ Crawl ALT", type="csv")
uploaded_new = st.file_uploader("2️⃣ Crawl NEU", type="csv")

if not uploaded_old or not uploaded_new:
    st.stop()

df_old = pd.read_csv(uploaded_old)
df_new = pd.read_csv(uploaded_new)

# --- Spaltenvereinheitlichung ---
def standardize_column_name(col):
    col = col.lower().strip()
    return {
        "url": "URL", "urls": "URL", "address": "URL",
        "title": "Title Tag", "title tag": "Title Tag",
        "meta description": "Meta Description", "description": "Meta Description",
        "body content": "Body Content", "content": "Body Content",
        "status": "Status code", "status code": "Status code",
        "clicks": "Klicks", "backlinks": "Backlinks",
        "h1": "H1"
    }.get(col, col)

df_old.columns = [standardize_column_name(c) for c in df_old.columns]
df_new.columns = [standardize_column_name(c) for c in df_new.columns]

# --- Nur Status Code 200 behalten ---
initial_old = len(df_old)
initial_new = len(df_new)
if "Status code" in df_old.columns:
    df_old = df_old[df_old["Status code"].astype(str).str.strip() == "200"]
if "Status code" in df_new.columns:
    df_new = df_new[df_new["Status code"].astype(str).str.strip() == "200"]
st.info(f"🔎 Alt-URLs mit Status 200: {len(df_old)} / Neu-URLs mit Status 200: {len(df_new)}")

# --- Vorbereitung ---
for df in [df_old, df_new]:
    if "URL" not in df.columns:
        st.error("❌ Beide Dateien brauchen eine URL-Spalte.")
        st.stop()
    df["URL"] = df["URL"].astype(str).str.strip()

df_old = df_old.drop_duplicates(subset="URL", keep="first").reset_index(drop=True)
df_old.fillna("", inplace=True)
df_new.fillna("", inplace=True)

# --- Vergleichs-Tools ---
def clean_title(title, suffix):
    return str(title).replace(suffix, "").strip().lower()

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[\\W_]+", " ", text)
    return re.sub(r"\\s+", " ", text).strip()

def soft_match(a, b):
    words_a = set(normalize_text(a).split())
    words_b = set(normalize_text(b).split())
    return words_a.issubset(words_b) or words_b.issubset(words_a)

def get_embedding(text, model):
    text = text.replace("\n", " ")[:8000]
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def combine_text(row, fields):
    return " ".join(str(row.get(f, "")) for f in fields)

# --- Embeddings erstellen ---
st.info("🔍 Embeddings werden erstellt...")
df_old["embedding"] = df_old.apply(lambda row: get_embedding(combine_text(row, ['H1', 'Title Tag', 'Meta Description', 'Body Content']), model_choice), axis=1)
df_new["embedding"] = df_new.apply(lambda row: get_embedding(combine_text(row, ['H1', 'Title Tag', 'Meta Description', 'Body Content']), model_choice), axis=1)

embeddings_old = np.vstack(df_old["embedding"].to_numpy())
embeddings_new = np.vstack(df_new["embedding"].to_numpy())
similarity_matrix = cosine_similarity(embeddings_old, embeddings_new)

# --- Matching mit Onpage-Priorisierung ---
results = []
for idx_old, row_old in df_old.iterrows():
    candidates = []
    for idx_new, row_new in df_new.iterrows():
        h1 = soft_match(row_old.get("H1", ""), row_new.get("H1", ""))
        title = soft_match(clean_title(row_old.get("Title Tag", ""), suffix_alt), clean_title(row_new.get("Title Tag", ""), suffix_neu))
        meta = soft_match(row_old.get("Meta Description", ""), row_new.get("Meta Description", ""))
        match_count = sum([h1, title, meta])
        sim = similarity_matrix[idx_old][idx_new]
        candidates.append((match_count, sim, idx_new, h1, title, meta))

    candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
    match_count, sim_score, best_idx, h1_match, title_match, meta_match = candidates[0]

    if match_count > 0:
        match = "Match"
        confidence = "very likely" if match_count == 3 else "likely" if match_count == 2 else "possible"
    elif sim_score >= 0.90:
        match = "Match"
        confidence = "likely"
    elif sim_score >= 0.80:
        match = "Match"
        confidence = "possible"
    elif sim_score >= min_similarity_threshold:
        match = "Match"
        confidence = "low"
    else:
        match = "No Match"
        confidence = "low"
        best_idx = None

    row_new = df_new.iloc[best_idx] if best_idx is not None else {}

    results.append({
        "Old URL": row_old.get("URL", ""),
        "New URL": row_new.get("URL", "") if best_idx is not None else "",
        "Similarity Score": round(sim_score, 4) if best_idx is not None else "",
        "Match": match,
        "Confidence Score": confidence,
        "H1 Match": h1_match,
        "Title Tag Match": title_match,
        "Meta Description Match": meta_match,
        "Status Code ALT": row_old.get("Status code", ""),
        "H1 ALT": row_old.get("H1", ""),
        "Title Tag ALT": row_old.get("Title Tag", ""),
        "Meta Description ALT": row_old.get("Meta Description", ""),
        "Body Content ALT": row_old.get("Body Content", ""),
        "Klicks": row_old.get("Klicks", ""),
        "Backlinks": row_old.get("Backlinks", ""),
        "Status Code NEU": row_new.get("Status code", "") if best_idx is not None else "",
        "H1 NEU": row_new.get("H1", "") if best_idx is not None else "",
        "Title Tag NEU": row_new.get("Title Tag", "") if best_idx is not None else "",
        "Meta Description NEU": row_new.get("Meta Description", "") if best_idx is not None else "",
        "Body Content NEU": row_new.get("Body Content", "") if best_idx is not None else ""
    })

# --- Anzeige & Export ---
df_result = pd.DataFrame(results)
st.success("✅ Matching abgeschlossen")
st.dataframe(df_result)

csv = df_result.to_csv(index=False).encode("utf-8")
st.download_button("📥 Ergebnis als CSV herunterladen", data=csv, file_name="redirect_mapping.csv", mime="text/csv")
