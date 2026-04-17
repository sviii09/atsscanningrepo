# -*- coding: utf-8 -*-
"""
Resume Evaluation System
- RAG (in-memory, no ChromaDB)
- ATS Scoring (TF-IDF + Logistic Regression)
- LLM Evaluation (Flan-T5)
Runs in Google Colab
"""

# ─────────────────────────────────────────────
# 1. INSTALL DEPENDENCIES
# ─────────────────────────────────────────────
# Run this cell first in Colab:
# !pip install kagglehub pandas scikit-learn sentence-transformers transformers accelerate -q

# ─────────────────────────────────────────────
# 2. IMPORTS
# ─────────────────────────────────────────────
import os
import glob
import numpy as np
import pandas as pd
import kagglehub

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ─────────────────────────────────────────────
# 3. LOAD DATASET
# ─────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    """Download and load the resume dataset from Kaggle."""
    path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
    csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in dataset folder: {path}")
    df = pd.read_csv(csv_files[0])
    df.columns = df.columns.str.strip()
    df = df[["Resume_str", "Category"]].dropna()
    df["Resume_str"] = df["Resume_str"].astype(str).str.replace("\n", " ").str.strip()
    print(f"✅ Loaded {len(df)} resumes | Categories: {df['Category'].nunique()}")
    return df


# ─────────────────────────────────────────────
# 4. TEXT CHUNKING
# ─────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    """Split text into fixed-size word chunks."""
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def build_chunks(documents: list[str]) -> list[str]:
    """Chunk all documents into a flat list."""
    chunks = []
    for doc in documents:
        chunks.extend(chunk_text(doc))
    print(f"✅ Total chunks: {len(chunks)}")
    return chunks


# ─────────────────────────────────────────────
# 5. EMBEDDING + RETRIEVAL (IN-MEMORY RAG)
# ─────────────────────────────────────────────
def build_embeddings(chunks: list[str], model: SentenceTransformer) -> np.ndarray:
    """Encode all chunks into dense embeddings."""
    print("⏳ Building embeddings (this may take a minute)...")
    embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True)
    print("✅ Embeddings built.")
    return embeddings


def retrieve(
    query: str,
    chunks: list[str],
    embeddings: np.ndarray,
    embed_model: SentenceTransformer,
    top_k: int = 2,
) -> list[str]:
    """Retrieve the top-k most similar chunks for a query."""
    query_embedding = embed_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]


# ─────────────────────────────────────────────
# 6. ATS SCORING (TF-IDF + LOGISTIC REGRESSION)
# ─────────────────────────────────────────────
def build_ats_model(documents: list[str], labels: list[str]):
    """Train TF-IDF + Logistic Regression classifier for ATS scoring."""
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(documents)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)
    print("✅ ATS classifier trained.")
    return vectorizer, clf


def ats_score(resume_text: str, vectorizer: TfidfVectorizer, clf: LogisticRegression) -> dict:
    """Return predicted role and ATS confidence score (0–100)."""
    vec = vectorizer.transform([resume_text])
    probs = clf.predict_proba(vec)[0]
    max_prob = float(np.max(probs))
    predicted_role = clf.classes_[np.argmax(probs)]
    return {
        "predicted_role": predicted_role,
        "ats_score": int(max_prob * 100),
    }


# ─────────────────────────────────────────────
# 7. LLM SETUP (FLAN-T5)
# ─────────────────────────────────────────────
def load_llm(model_name: str = "google/flan-t5-base"):
    """Load Flan-T5 tokenizer and model."""
    print(f"⏳ Loading LLM: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("✅ LLM loaded.")
    return tokenizer, model


def run_llm(prompt: str, tokenizer, model, max_new_tokens: int = 150) -> str:
    """Run LLM inference with overflow protection."""
    # Truncate prompt tokens to avoid exceeding model limit (512 tokens for T5)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=400,
    )
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.2,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ─────────────────────────────────────────────
# 8. COMBINED EVALUATION PIPELINE
# ─────────────────────────────────────────────
def build_prompt(
    resume_text: str,
    context: str,
    predicted_role: str,
    score: int,
) -> str:
    """Build a short, structured prompt for the LLM."""
    return f"""Evaluate this resume. Be specific and concise.

Predicted Role: {predicted_role}
ATS Score: {score}/100

Resume (summary):
{resume_text[:300]}

Similar Resume Snippets:
{context[:300]}

Fill in:
fit_analysis: <how well does the resume fit the role>
missing_skills: <skills missing for this role>
improvements: <top 2 suggestions to improve the resume>"""


def evaluate_resume(
    resume_text: str,
    *,
    chunks: list[str],
    embeddings: np.ndarray,
    embed_model: SentenceTransformer,
    vectorizer: TfidfVectorizer,
    clf: LogisticRegression,
    tokenizer,
    llm_model,
) -> dict:
    """
    Full evaluation pipeline for a single resume.

    Returns a dict with:
      - predicted_role
      - ats_score
      - llm_evaluation (raw text)
    """
    # Truncate input to avoid downstream overflow
    resume_text = " ".join(resume_text.split()[:200])

    # ATS scoring
    ats = ats_score(resume_text, vectorizer, clf)

    # RAG retrieval
    retrieved = retrieve(resume_text, chunks, embeddings, embed_model, top_k=2)
    context = " ".join(" ".join(doc.split()[:75]) for doc in retrieved)

    # LLM evaluation
    prompt = build_prompt(
        resume_text,
        context,
        ats["predicted_role"],
        ats["ats_score"],
    )
    llm_output = run_llm(prompt, tokenizer, llm_model)

    return {
        "predicted_role": ats["predicted_role"],
        "ats_score": ats["ats_score"],
        "llm_evaluation": llm_output,
    }


# ─────────────────────────────────────────────
# 9. MAIN — INITIALIZE AND TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # --- Load data ---
    df = load_dataset()
    documents = df["Resume_str"].tolist()
    labels = df["Category"].tolist()

    # --- Build RAG components ---
    chunks = build_chunks(documents)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = build_embeddings(chunks, embed_model)

    # --- Build ATS classifier ---
    vectorizer, clf = build_ats_model(documents, labels)

    # --- Load LLM ---
    tokenizer, llm_model = load_llm()

    # --- Test on first resume ---
    test_resume = documents[0]
    result = evaluate_resume(
        test_resume,
        chunks=chunks,
        embeddings=embeddings,
        embed_model=embed_model,
        vectorizer=vectorizer,
        clf=clf,
        tokenizer=tokenizer,
        llm_model=llm_model,
    )

    print("\n" + "=" * 50)
    print(f"Predicted Role : {result['predicted_role']}")
    print(f"ATS Score      : {result['ats_score']}/100")
    print(f"\nLLM Evaluation :\n{result['llm_evaluation']}")
    print("=" * 50)
