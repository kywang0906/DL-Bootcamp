import os
import re
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from gensim.models.doc2vec import Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from train_data import NeuralNet  # your classifier definition
from ckip_transformers.nlp import CkipWordSegmenter

# -----------------------------
# 1. Initialize FastAPI & Templates
# -----------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# 2. Load Doc2Vec Model
# -----------------------------
doc2vec_model = Doc2Vec.load("doc2vec.model")

# -----------------------------
# 3. Load and fit TF–IDF (must match training)
# -----------------------------
df_clean = pd.read_csv("segmented_data_cleaned.csv", names=["Label", "Segmented"])
texts = df_clean["Segmented"].tolist()
tfidf = TfidfVectorizer(
    max_features=200,
    token_pattern=r"(?u)\b\w+\b"
)
tfidf.fit(texts)

# -----------------------------
# 4. Load classifier & label list
# -----------------------------
labels_list = df_clean["Label"].astype("category").cat.categories.tolist()
num_classes = len(labels_list)

# input dimension = doc2vec_dim + tfidf_dim
input_dim = doc2vec_model.vector_size + tfidf.max_features

classification_model = NeuralNet(input_dim, num_classes)
classification_model.load_state_dict(
    torch.load("best_classification_model.pth", map_location="cpu")
)
classification_model.eval()

# -----------------------------
# 5. Initialize CKIP segmenter
# -----------------------------
# use CPU; if you have GPU on server, pass device="cuda"
ws = CkipWordSegmenter(model="bert-base", device=-1)

# -----------------------------
# 6. Define cleaning and segmentation
# -----------------------------
def clean_text(text: str) -> str:
    """
    Lowercase, remove 're:'/'fw:' prefixes, strip unwanted punctuation
    (keeping square brackets), remove leading punctuation/whitespace,
    and drop announcements starting with '公告'.
    """
    t = text.lower()
    t = re.sub(r'^(re:|fw:)\s*', '', t, flags=re.I)
    # remove all punctuation except [ ]
    t = re.sub(r'[^\w\s\[\]]', '', t)
    # remove leading punctuation or whitespace
    t = re.sub(r'^[\s\[\]]+', '', t)
    # drop if starts with announcement marker
    if t.startswith("公告"):
        return ""
    return t

def segment_text(text: str) -> list[str]:
    """
    Given cleaned text, return list of tokens via CKIP.
    """
    if not text:
        return []
    # ws([text]) returns List[List[str]]
    tokens = ws([text])[0]
    return tokens

# -----------------------------
# 7. Home Page (GET)
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "all_labels": labels_list
    })

# -----------------------------
# 8. Prediction (POST)
# -----------------------------
@app.post("/api/model/prediction", response_class=HTMLResponse)
def predict(request: Request, title: str = Form(...)):
    # 1) Clean
    cleaned = clean_text(title)
    # 2) Segment
    tokens = segment_text(cleaned)
    seg_str = " ".join(tokens)

    # 3) Infer Doc2Vec
    doc_vec = doc2vec_model.infer_vector(tokens)
    # 4) Transform TF–IDF
    tfidf_vec = tfidf.transform([seg_str]).toarray()[0]
    # 5) Concatenate features
    x = np.hstack([doc_vec, tfidf_vec]).astype(np.float32)

    # 6) Classification
    with torch.no_grad():
        inp = torch.from_numpy(x).unsqueeze(0)  # shape (1, D)
        out = classification_model(inp)
        pred = torch.argmax(out, dim=1).item()
    predicted_label = labels_list[pred]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "original_title": title,
        "prediction": predicted_label,
        "all_labels": labels_list
    })

# -----------------------------
# 9. Feedback (POST)
# -----------------------------
@app.post("/api/model/feedback")
def correct(title: str = Form(...), label: str = Form(...)):
    df_fb = pd.DataFrame([[title, label]], columns=["Title", "Label"])
    if os.path.exists("user-labeled-titles.csv"):
        df_fb.to_csv("user-labeled-titles.csv", mode="a", header=False, index=False)
    else:
        df_fb.to_csv("user-labeled-titles.csv", index=False)
    return RedirectResponse("/", status_code=303)