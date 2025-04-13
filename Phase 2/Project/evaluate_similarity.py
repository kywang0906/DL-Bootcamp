import numpy as np
from gensim.models.doc2vec import Doc2Vec
import collections
from tqdm import tqdm

def evaluate_similarity(model, inferred_vectors):
    """
    Evaluate the similarity of inferred vectors.
    """
    print("Test Similarity")
    ranks = []

    for doc_id in tqdm(range(len(inferred_vectors)), desc="Calculating Similarity"):
        sims = model.dv.most_similar([inferred_vectors[doc_id]], topn=len(model.dv))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

    # Calculate similarity metrics
    counter = collections.Counter(ranks)
    total_docs = len(inferred_vectors)
    self_similarity = counter[0] / total_docs
    second_self_similarity = (counter[0] + counter[1]) / total_docs

    print(f"Self Similarity {self_similarity:.3f}")
    print(f"Second Self Similarity {second_self_similarity:.3f}")

    return self_similarity, second_self_similarity

if __name__ == '__main__':
    # Load model and vectors
    model = Doc2Vec.load("doc2vec.model")
    inferred_vectors = np.load("inferred_vectors.npy", allow_pickle=True)

    # Evaluate similarity
    self_similarity, second_self_similarity = evaluate_similarity(model, inferred_vectors)
    print(f"Self Similarity {self_similarity:.3f}")
    print(f"Second Self Similarity {second_self_similarity:.3f}")
