import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import collections

def train_doc2vec(
    filepath="sample_train_data2.csv",
    vector_size=200,
    window=8,
    min_count=1,
    epochs=200,
    save_path="doc2vec.model"
):

    print("Titles Ready")
    df = pd.read_csv(filepath, names=["Label", "Segmented"])
    df["Segmented"] = df["Segmented"].fillna("")

    print("Tagged Documents Ready")
    train_corpus = [
        TaggedDocument(words=seg.split(), tags=[i])
        for i, seg in enumerate(df["Segmented"])
    ]

    print("Start Training")
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, epochs=epochs)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(save_path)
    print('Model Saved')

if __name__ == '__main__':
    train_and_evaluate_doc2vec(filepath="segmented_data.csv")