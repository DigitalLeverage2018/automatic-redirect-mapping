import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import re

# Titel & Beschreibung
st.title("🔁 Automatisches Redirect-Mapping")

st.markdown("""
Lade zwei Crawls deiner alten und neuen Website als CSV hoch – dieses Tool hilft dir, passende Weiterleitungen zu finden.  
Nur Seiten mit Status **200** werden berücksichtigt.  
🧠 Verglichen werden Onpage-Elemente (H1, Title, Meta) und der Seiteninhalt mittels Embeddings von OpenAI.
""")

# OpenAI API Key
api_key = st.text_input("🔑 OpenAI API Key", type="password")
if not api_key:
    st.stop()
client = openai.OpenAI(api_key=api_key)

# Embedding Modellwahl
st.header("🧠 Embedding-Modell wählen")
model_choice = st.selectbox(
    "Welches Modell möchtest du verwenden?",
    options=["text-embedding-3-small", "text-embedding-3-large"],
    format_func=lambda x: "Small (schneller, günstiger)" if "small" in x else "Large (exakter, teurer)"
)
st.markdown("""
**Small:** schneller, günstiger, weniger exakt – ideal bei >100 URLs oder stabilen Onpage-Elementen  
**Large:** langsamer, exakter, teurer – ideal bei komplexen Umstellungen
""")

# Matching Einstellungen
st.header("⚙️ Einstellungen")
suffix_alt = st.text_input("🔧 Title Tag Suffix ALT", value="")
suffix_neu = st.text_input("🔧 Title Tag Suffix NEU", value="")
threshold = st.slider("🔒 Mindest-Similarity für gültige Matches", min_value=0.0, max_value=1.0, value=0.7, step=0.01)

# Upload ALT/NEU

uploaded_old = st.file_uploader("📁 ALT-Crawl (CSV 1)", type="csv")

with st.expander("ℹ️ ALT-Crawl: Benötigte Spalten anzeigen"):
    st.markdown("""
**Benötigte Spaltennamen:**  
`URL`, `Status code`, `H1`, `Title Tag`, `Meta Description`, `Body Content`
""")

uploaded_new = st.file_uploader("📁 NEU-Crawl (CSV 2)", type="csv")

with st.expander("ℹ️ NEU-Crawl: Benötigte Spalten anzeigen"):
    st.markdown("""
**Benötigte Spaltennamen:**  
`URL`, `Status code`, `H1`, `Title Tag`, `Meta Description`, `Body Content`
""")



if not uploaded_old or not uploaded_new:
    st.stop()


# Daten einlesen
df_old = pd.read_csv(uploaded_old)
df_new = pd.read_csv(uploaded_new)

# Spaltennamen vereinheitlichen
def standardize_column_name(col):
    col = col.lower().strip()
    return {
        "url": "URL", "urls": "URL", "address": "URL",
        "title": "Title Tag", "title tag": "Title Tag",
        "meta description": "Meta Description", "description": "Meta Description",
        "body content": "Body Content", "content": "Body Content",
        "status": "Status code", "status code": "Status code",
        "h1": "H1"
    }.get(col, col)

df_old.columns = [standardize_column_name(c) for c in df_old.columns]
df_new.columns = [standardize_column_name(c) for c in df_new.columns]

# Klicks & Backlinks ignorieren
df_old = df_old.drop(columns=["Klicks", "Backlinks"], errors="ignore")
df_new = df_new.drop(columns=["Klicks", "Backlinks"], errors="ignore")

# Nur Status 200
df_old = df_old[df_old["Status code"].astype(str).str.strip() == "200"]
df_new = df_new[df_new["Status code"].astype(str).str.strip() == "200"]
df_old = df_old.drop_duplicates(subset="URL").reset_index(drop=True)
df_old.fillna("", inplace=True)
df_new.fillna("", inplace=True)

# Helper-Funktionen
def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[\\W_]+", " ", text)
    return re.sub(r"\\s+", " ", text).strip()

def soft_match(a, b):
    words_a = set(normalize_text(a).split())
    words_b = set(normalize_text(b).split())
    return words_a.issubset(words_b) or words_b.issubset(words_a)

def clean_title(title, suffix):
    return str(title).replace(suffix, "").strip().lower()

def get_embedding(text, model):
    text = text.replace("\n", " ")[:8000]
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def combine_text(row):
    return " ".join(str(row.get(f, "")) for f in ["H1", "Title Tag", "Meta Description", "Body Content"])

# Embeddings generieren
st.info("🔍 Embeddings werden erstellt…")
df_old["embedding"] = df_old.apply(lambda row: get_embedding(combine_text(row), model_choice), axis=1)
df_new["embedding"] = df_new.apply(lambda row: get_embedding(combine_text(row), model_choice), axis=1)

# Ähnlichkeitsmatrix
embeddings_old = np.vstack(df_old["embedding"].to_numpy())
embeddings_new = np.vstack(df_new["embedding"].to_numpy())
similarity_matrix = cosine_similarity(embeddings_old, embeddings_new)

# Matching
results = []
for idx_old, row_old in df_old.iterrows():
    candidates = []
    for idx_new, row_new in df_new.iterrows():
        h1 = soft_match(row_old.get("H1", ""), row_new.get("H1", ""))
        title = soft_match(clean_title(row_old.get("Title Tag", ""), suffix_alt), clean_title(row_new.get("Title Tag", ""), suffix_neu))
        meta = soft_match(row_old.get("Meta Description", ""), row_new.get("Meta Description", ""))
        match_count = sum([h1, title, meta])
        sim = similarity_matrix[idx_old][idx_new]
        if sim >= threshold:
            candidates.append((match_count, sim, idx_new, h1, title, meta))

    if not candidates:
        results.append({
            "Old URL": row_old.get("URL", ""),
            "New URL": "",
            "Similarity Score": "",
            "Match": "No Match",
            "Confidence Score": "",
            "H1 Match": False,
            "Title Tag Match": False,
            "Meta Description Match": False
        })
        continue

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    match_count, sim_score, best_idx, h1_match, title_match, meta_match = candidates[0]
    row_new = df_new.iloc[best_idx]

    if sim_score >= 0.9 and match_count >= 1:
        confidence = "very likely"
    elif sim_score >= 0.9:
        confidence = "likely"
    elif sim_score >= 0.8 and match_count >= 1:
        confidence = "likely"
    elif sim_score >= 0.8:
        confidence = "possible"
    elif sim_score >= threshold and match_count >= 1:
        confidence = "possible"
    else:
        confidence = "low"

    results.append({
        "Old URL": row_old.get("URL", ""),
        "New URL": row_new.get("URL", ""),
        "Similarity Score": round(sim_score, 4),
        "Match": "Match",
        "Confidence Score": confidence,
        "H1 Match": h1_match,
        "Title Tag Match": title_match,
        "Meta Description Match": meta_match,
        "Status Code ALT": row_old.get("Status code", ""),
        "Status Code NEU": row_new.get("Status code", ""),
        "Title Tag ALT": row_old.get("Title Tag", ""),
        "Title Tag NEU": row_new.get("Title Tag", ""),
        "Meta Description ALT": row_old.get("Meta Description", ""),
        "Meta Description NEU": row_new.get("Meta Description", ""),
        "H1 ALT": row_old.get("H1", ""),
        "H1 NEU": row_new.get("H1", "")
    })

# Ergebnis anzeigen
df_result = pd.DataFrame(results)
st.success("✅ Matching abgeschlossen")
st.dataframe(df_result)

# Export
csv = df_result.to_csv(index=False).encode("utf-8")
st.download_button("📥 Ergebnis als CSV herunterladen", data=csv, file_name="redirect_mapping.csv", mime="text/csv")
