import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import collections

def train_and_evaluate_doc2vec(
    filepath="sample_train_data.csv",
    vector_size=200,
    window=8,
    min_count=1,
    epochs=200,
    save_path="doc2vec.model"
):

    print("Titles Ready")
    df = pd.read_csv(filepath, names=["Label", "Segmented"])

    df["Segmented"] = df["Segmented"].fillna("")
    # Remove rows where the title (Segmented) starts with "公告"
    df = df[~df["Segmented"].str.startswith("公告")]

    print("Tagged Documents Ready")
    train_corpus = [
        TaggedDocument(words=seg.split(), tags=[i])
        for i, seg in enumerate(df["Segmented"])
    ]

    print("Start Training")
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, epochs=epochs)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    print("Test Similarity")
    ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

        # Print progress every 100 documents
        if doc_id % 100 == 0:
            print(doc_id)

    counter = collections.Counter(ranks)
    total_docs = len(train_corpus)
    self_similarity = counter[0] / total_docs
    second_self_similarity = (counter[0] + counter[1]) / total_docs

    if self_similarity >= 0.8 and second_self_similarity >= 0.8:
        model.save(save_path)
        print("Model saved.")
    else:
        print("Model not saved.")

    return self_similarity, second_self_similarity

if __name__ == '__main__':
    self_similarity, second_self_similarity = train_and_evaluate_doc2vec(filepath="segmented_data.csv")
    print(f"Self Similarity {self_similarity:.3f}")
    print(f"Second Self Similarity {second_self_similarity:.3f}")