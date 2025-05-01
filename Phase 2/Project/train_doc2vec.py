import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re
import os

def preprocess_text(text):
    # 處理空值與多餘空格
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", text.strip())

def train_doc2vec(
    filepath="segmented_data_cleaned.csv",
    vector_size=200,
    window=3,
    min_count=5,
    epochs=50,
    dm=1, 
    workers=4,
    save_path="doc2vec.model"
):
    # 載入並預處理資料
    print("📄 Loading and preprocessing data...")
    df = pd.read_csv(filepath, names=["Label", "Segmented"])
    df["Segmented"] = df["Segmented"].fillna("").map(preprocess_text)

    print("📌 Building tagged documents...")
    train_corpus = [
        TaggedDocument(words=seg.split(), tags=[str(i)])
        for i, seg in enumerate(df["Segmented"])
        if seg.strip()  # 排除空白資料
    ]

    print(f"✅ Total documents: {len(train_corpus)}")

    print("🧠 Initializing Doc2Vec model...")
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        dm=1,
        seed=21
    )

    print("📚 Building vocabulary...")
    model.build_vocab(train_corpus)

    print("🚀 Training Doc2Vec model...")
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    model.save(save_path)
    print(f"💾 Model saved to: {save_path}")

if __name__ == '__main__':
    train_doc2vec()