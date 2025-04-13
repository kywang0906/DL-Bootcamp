import os
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
import pandas as pd

def load_existing_model(save_path="doc2vec.model"):
    """
    Load an existing Doc2Vec model.
    """
    if os.path.exists(save_path):
        print(f"Loading existing model from: {save_path}")
        model = Doc2Vec.load(save_path)
        print("Model loaded successfully.")
        return model
    else:
        print("⚠️ No existing model found.")
        return None

def generate_inferred_vectors(model, train_corpus, vectors_path="inferred_vectors.npy"):
    """
    Generate inferred vectors for all documents without multi-threading.
    """
    print("Generating inferred vectors without multi-threading...")
    inferred_vectors = []

    for doc in tqdm(train_corpus, desc="Generating Vectors", total=len(train_corpus)):
        inferred_vector = model.infer_vector(doc.words)
        inferred_vectors.append(inferred_vector)

    inferred_vectors = np.array(inferred_vectors)

    # Save inferred vectors to file
    np.save(vectors_path, inferred_vectors)
    print(f"Inferred vectors saved to {vectors_path}")

if __name__ == '__main__':
    # Load the trained model
    model = load_existing_model("doc2vec.model")

    # Load segmented data
    print("Titles Ready")
    df = pd.read_csv("segmented_data2.csv", names=["Label", "Segmented"])
    df["Segmented"] = df["Segmented"].fillna("")

    print("Tagged Documents Ready")
    train_corpus = [
        TaggedDocument(words=seg.split(), tags=[i])
        for i, seg in enumerate(df["Segmented"])
    ]

    # Generate inferred vectors
    vectors_path = "inferred_vectors.npy"
    generate_inferred_vectors(model, train_corpus, vectors_path)
