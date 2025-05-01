import os
import numpy as np
import pandas as pd
import faiss
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm


def load_model_and_vectors(model_path="doc2vec.model", vectors_path="inferred_vectors.npy"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(vectors_path):
        raise FileNotFoundError(f"Inferred vectors not found at {vectors_path}")

    print(f"🔄 Loading model from {model_path}...")
    model = Doc2Vec.load(model_path)
    print(f"🔄 Loading inferred vectors from {vectors_path}...")
    inferred_vectors = np.load(vectors_path, allow_pickle=True)

    return model, inferred_vectors


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-10)


def evaluate_with_faiss(model, inferred_vectors):
    print("🚀 Starting FAISS-based evaluation...")

    # Prepare index
    model_vectors = model.dv.vectors
    model_vectors = normalize_vectors(model_vectors.astype('float32'))
    inferred_vectors = normalize_vectors(inferred_vectors.astype('float32'))

    dim = model_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(model_vectors)

    print("⚡ Performing batch similarity search...")
    top_k = 2
    D, I = index.search(inferred_vectors, top_k)

    total_docs = inferred_vectors.shape[0]
    first_match = np.sum(I[:, 0] == np.arange(total_docs))
    second_match = np.sum(
        (I[:, 0] == np.arange(total_docs)) | (I[:, 1] == np.arange(total_docs))
    )

    self_similarity = first_match / total_docs
    second_self_similarity = second_match / total_docs

    print("\n Evaluation Results:")
    print(f"  ➡️ Self Similarity        : {self_similarity:.4f}")
    print(f"  ➡️ Second Self Similarity : {second_self_similarity:.4f}")

    return self_similarity, second_self_similarity


if __name__ == '__main__':
    model, inferred_vectors = load_model_and_vectors(
        model_path="doc2vec.model",
        vectors_path="inferred_vectors.npy"
    )
    evaluate_with_faiss(model, inferred_vectors)