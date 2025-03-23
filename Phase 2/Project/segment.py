import os
import pandas as pd
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

CLEANED_FOLDER = "cleaned_data"
SAMPLE = "sample_input"
SEGMENTED_FOLDER = "segmented_data"
FILTER_POS = ["Caa", "Cab", "Cba", "Cbb", "P", "DE", "COLONCATEGORY", "COMMACATEGORY", "DASHCATEGORY", "DOTCATEGORY", "ETCCATEGORY", "EXCLAMATIONCATEGORY", "PARENTHESISCATEGORY", "PAUSECATEGORY", "PERIODCATEGORY", "QUESTIONCATEGORY", "SEMICOLONCATEGORY", "SPCHANGECATEGORY", "WHITESPACE"]

# Initialize CKIP drivers
ws_driver = CkipWordSegmenter(model="bert-base")
pos_driver = CkipPosTagger(model="bert-base")

def segment(filename):
    all_rows = []
    # file = os.path.join(CLEANED_FOLDER, filename)
    file = os.path.join(SAMPLE, filename)
    print(f"Reading {file}...")

    df = pd.read_csv(file)

    # Get label names
    label = filename.replace("-titles.csv", "")

    titles = df["Title"].astype(str).tolist()
    print(f"Segmenting & POS tagging {len(titles)} titles from {label}...")

    # Segment words
    ws_result = ws_driver(titles)
    # Tag parts of speech
    pos_result = pos_driver(ws_result)

    # Filter out prepositions, conjunctions, white spaces, and punctuations
    for ws, pos in zip(ws_result, pos_result):
        filtered_words = [w for w, p in zip(ws, pos) if p not in FILTER_POS]
        filtered_segmented = " ".join(filtered_words)
        all_rows.append((label, filtered_segmented))

    # Save file
    res = pd.DataFrame(all_rows, columns=["Label", "Segmented"])
    if not os.path.exists(SEGMENTED_FOLDER):
        os.makedirs(SEGMENTED_FOLDER)
    segmented_path = os.path.join(SEGMENTED_FOLDER, f"{label}_segmented.csv")
    res.to_csv(segmented_path, index=False, encoding="utf-8-sig")
    print(f"Filtered and segmented data saved to: {segmented_path}")

if __name__ == '__main__':
    # files = [f for f in os.listdir(CLEANED_FOLDER)]
    files = [f for f in os.listdir(SAMPLE)]
    for filename in files:
        segment(filename)
    