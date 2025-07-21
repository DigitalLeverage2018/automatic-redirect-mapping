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

# --- Benutzer-Einstellungen ---
st.header("⚙️ Einstellungen für Vergleich & Matching")

suffix_alt = st.text_input("🔧 Title Tag Suffix ALT (z. B. ' | Beispiel AG')", value="")
suffix_neu = st.text_input("🔧 Title Tag Suffix NEU (z. B. ' | Neue Firma')", value="")

min_similarity_threshold = st.slider(
    "🔒 Minimale Ähnlichkeit (Similarity Score), damit eine Weiterleitung empfohlen wird",
    min_value=0.0, max_value=1.0, value=0.40, step=0.01,
    help="Wenn keine Onpage-Elemente übereinstimmen, entscheidet dieser Wert, ob trotzdem weitergeleitet wird."
)

# --- Datei Upload ---
st.header("📁 CSV-Dateien hochladen")
uploaded_old = st.file_uploader("1️⃣ Crawl ALT (mit Klicks, Backlinks etc.)", type="csv")
uploaded_new = st.file_uploader("2️⃣ Crawl NEU (z. B. Staging)", type="csv")

if not uploaded_old or not uploaded_new:
    st.stop()

df_old = pd.read_csv(uploaded_old)
df_new = pd.read_csv(uploaded_new)

# --- Spaltenzuordnung vereinheitlichen ---
def standardize_column_name(col):
    col = col.lower().strip()
    if col in ["url", "urls", "address"]:
        return "URL"
    elif col in ["title", "title tag"]:
        return "Title Tag"
    elif col in ["meta description", "description"]:
        return "Meta Description"
    elif col in ["body content", "content"]:
        return "Body Content"
    elif col in ["status", "status code"]:
        return "Status code"
    elif col == "clicks":
        return "Klicks"
    elif col == "backlinks":
        return "Backlinks"
    elif col == "h1":
        return "H1"
    return col

df_old.columns = [standardize_column_name(c) for c in df_old.columns]
df_new.columns = [standardize_column_name(c) for c in df_new.columns]

for df in [df_old, df_new]:
    if "URL" not in df.columns:
        st.error("❌ Beide CSV-Dateien müssen eine Spalte für URLs enthalten.")
        st.stop()
    df["URL"] = df["URL"].astype(str).str.strip()

# --- Duplikate ALT entfernen ---
initial_count = len(df_old)
df_old = df_old.drop_duplicates(subset="URL", keep="first").reset_index(drop=True)
removed = initial_count - len(df_old)
if removed > 0:
    st.info(f"🔁 {removed} doppelte alte URLs entfernt – nur erste Vorkommen berücksichtigt.")

df_old.fillna("", inplace=True)
df_new.fillna("", inplace=True)

# --- Vergleichs- und Embedding-Tools ---
def clean_title(title, suffix):
    return str(title).replace(suffix, "").strip().lower()

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[\\W_]+", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def soft_match(a, b):
    words_a = set(normalize_text(a).split())
    words_b = set(normalize_text(b).split())
    return words_a.issubset(words_b) or words_b.issubset(words_a)

def get_embedding(text, model):
    text = text.replace("\\n", " ")[:8000]
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

# --- Matching durchführen ---
results = []
for idx_old, row_old in df_old.iterrows():
    best_idx = np.argmax(similarity_matrix[idx_old])
    similarity_score = similarity_matrix[idx_old][best_idx]
    row_new = df_new.iloc[best_idx]

    h1_match = soft_match(row_old.get("H1", ""), row_new.get("H1", ""))
    title_match = soft_match(clean_title(row_old.get("Title Tag", ""), suffix_alt), clean_title(row_new.get("Title Tag", ""), suffix_neu))
    meta_match = soft_match(row_old.get("Meta Description", ""), row_new.get("Meta Description", ""))

    match_status = "No Match"
    confidence = "low"
    matched = False
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
    elif similarity_score >= 0.90:
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
        row_new = {}
        best_idx = None

    results.append({
        "Old URL": row_old.get("URL", ""),
        "New URL": row_new.get("URL", "") if best_idx is not None else "",
        "Similarity Score": round(similarity_score, 4) if best_idx is not None else "",
        "Match": match_status,
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

# --- Ergebnis anzeigen ---
result_df = pd.DataFrame(results)
st.success("✅ Matching abgeschlossen")
st.dataframe(result_df)

csv = result_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Ergebnis als CSV herunterladen", data=csv, file_name="redirect_mapping.csv", mime="text/csv")
