"""
Module 6 Week B — Lab: Embeddings Comparison

Compare three text representation methods — TF-IDF, GloVe, and
DistilBERT — on the BBC News corpus (5 categories).
"""

import numpy as np
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def build_tfidf(texts):
    """Build TF-IDF representations for a list of texts.

    Returns (tfidf_matrix, vectorizer).
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    return tfidf_matrix, vectorizer


def compute_tfidf_similarity(tfidf_matrix):
    """Compute pairwise cosine similarity from a TF-IDF matrix.

    Returns a numpy array of shape (n, n).
    """
    similarity_matrix = sklearn_cosine(tfidf_matrix)
    return similarity_matrix


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Returns a dict mapping each word to a numpy array.
    """
    embeddings = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=float)
            embeddings[word] = vector

    return embeddings


def text_to_glove(text, embeddings):
    """Compute the average GloVe embedding for a text.

    Skip out-of-vocabulary words. If every word is OOV, return a zero
    vector of shape (50,).
    """
    words = text.lower().split()

    vectors = []
    for w in words:
       if w in embeddings:
          vectors.append(embeddings[w])

    if len(vectors) == 0:
       return np.zeros(50)

    return np.mean(vectors, axis=0)


def extract_bert_embedding(text, tokenizer, model):
    """Extract a sentence embedding from DistilBERT.

    Returns a numpy array of shape (768,).
    """
    inputs = tokenizer(
       text,
       return_tensors="pt",
       truncation=True,
       max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state
    mask = inputs["attention_mask"]

    mask = mask.unsqueeze(-1).expand(hidden.size()).float()

    pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    return pooled.squeeze().numpy()


def compare_similarities(texts, queries, tfidf_sim, glove_embeddings,
                         bert_model, bert_tokenizer):
    """Compare similarity rankings across TF-IDF, GloVe, and BERT.

    For each query, find the top-3 most similar texts under each method,
    excluding the query itself. Return:

        {query_text: {"tfidf": [(text, score), ...],
                      "glove": [(text, score), ...],
                      "bert":  [(text, score), ...]}}
    """
    pasresults = {}
    results = {}

    for q in queries:
        q_result = {}

        idx = texts.index(q)
        tfidf_scores = tfidf_sim[idx]

        tfidf_top = sorted(
            [(texts[i], tfidf_scores[i]) for i in range(len(texts)) if i != idx],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        q_result["tfidf"] = tfidf_top

        q_vec = text_to_glove(q, glove_embeddings)

        glove_scores = []
        for t in texts:
            t_vec = text_to_glove(t, glove_embeddings)
            score = sklearn_cosine([q_vec], [t_vec])[0][0]
            glove_scores.append((t, score))

        glove_scores.sort(key=lambda x: x[1], reverse=True)
        q_result["glove"] = glove_scores[:3]

        q_vec = extract_bert_embedding(q, bert_tokenizer, bert_model)

        bert_scores = []
        for t in texts:
            t_vec = extract_bert_embedding(t, bert_tokenizer, bert_model)
            score = sklearn_cosine([q_vec], [t_vec])[0][0]
            bert_scores.append((t, score))

        bert_scores.sort(key=lambda x: x[1], reverse=True)
        q_result["bert"] = bert_scores[:3]

        results[q] = q_result

    return results


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModel

    # Load data
    df = pd.read_csv("data/bbc_news.csv")
    texts = df["text"].tolist()
    print(f"Loaded {len(texts)} texts")

    # Task 1: TF-IDF
    result = build_tfidf(texts)
    if result:
        tfidf_matrix, vectorizer = result
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        tfidf_sim = compute_tfidf_similarity(tfidf_matrix)
        if tfidf_sim is not None:
            print(f"TF-IDF similarity matrix shape: {tfidf_sim.shape}")

    # Task 2: GloVe
    glove = load_glove("data/glove_50k_50d.txt")
    if glove:
        print(f"Loaded {len(glove)} GloVe vectors")
        sample_emb = text_to_glove(texts[0], glove)
        if sample_emb is not None:
            print(f"Sample GloVe text embedding shape: {sample_emb.shape}")

    # Task 3: DistilBERT
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    sample_bert = extract_bert_embedding(texts[0], tokenizer, model)
    if sample_bert is not None:
        print(f"Sample BERT embedding shape: {sample_bert.shape}")

    # Task 4: Compare — pick one query per category so the cross-method
    # ranking comparison is not degenerate (the CSV is sorted by category,
    # so texts[:5] would all be from the same one).
    if result and glove and tfidf_sim is not None:
        queries = [df[df["category"] == cat]["text"].iloc[0]
                   for cat in df["category"].unique()]
        comparison = compare_similarities(
            texts, queries, tfidf_sim, glove, model, tokenizer
        )
        if comparison:
            for q in list(comparison.keys())[:2]:
                print(f"\nQuery: {q[:80]}...")
                for method in ["tfidf", "glove", "bert"]:
                    top = comparison[q].get(method, [])
                    print(f"  {method}: {[t[:40] for t, _ in top[:3]]}")
