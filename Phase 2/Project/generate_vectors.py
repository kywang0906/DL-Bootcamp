import os
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm

def preprocess_text(text):
    return text.strip() if isinstance(text, str) else ""

def load_doc2vec_model(path="doc2vec.model"):
    if os.path.exists(path):
        print(f"📦 Loading Doc2Vec model from: {path}")
        return Doc2Vec.load(path)
    else:
        raise FileNotFoundError(f"Model not found at: {path}")

def prepare_corpus(filepath="segmented_data_cleaned.csv"):
    print(f"📄 Loading segmented data from: {filepath}")
    df = pd.read_csv(filepath, names=["Label", "Segmented"])
    df["Segmented"] = df["Segmented"].fillna("").map(preprocess_text)
    train_corpus = [
        TaggedDocument(words=seg.split(), tags=[str(i)])
        for i, seg in enumerate(df["Segmented"])
    ]
    print(f"Total documents: {len(train_corpus)}")
    return train_corpus

def generate_and_save_vectors(model, corpus, output_path="inferred_vectors.npy"):
    print("Generating inferred vectors (epochs=50)...")
    vectors = []

    for doc in tqdm(corpus, desc="Inferring vectors"):
        vec = model.infer_vector(doc.words, epochs=50)
        vectors.append(vec)

    np.save(output_path, np.array(vectors))
    print(f"Vectors saved to: {output_path}")

if __name__ == "__main__":
    model = load_doc2vec_model("doc2vec.model")
    corpus = prepare_corpus("segmented_data_cleaned.csv")
    generate_and_save_vectors(model, corpus, "inferred_vectors.npy")