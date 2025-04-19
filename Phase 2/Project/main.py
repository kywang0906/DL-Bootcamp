from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from train_data import NeuralNet
import pandas as pd
import torch
import torch.nn as nn
from gensim.models.doc2vec import Doc2Vec
import os


# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

doc2vec_model = Doc2Vec.load("doc2vec.model")

# Load classification model and labels
df = pd.read_csv("segmented_data2.csv", names=["Label", "Segmented"])
label_encoder = df["Label"].astype("category").cat.categories.tolist()
num_classes = len(label_encoder)
input_dim = doc2vec_model.vector_size

classification_model = NeuralNet(input_dim, num_classes)
classification_model.load_state_dict(torch.load("classification_model.pth", map_location=torch.device("cpu")))
classification_model.eval()

#  Home Page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction API
@app.post("/api/model/prediction", response_class=HTMLResponse)
def predict(request: Request, title: str = Form(...)):
    vector = doc2vec_model.infer_vector(title.split())
    input_tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
    output = classification_model(input_tensor)
    _, predicted = torch.max(output, 1)
    predicted_label = label_encoder[predicted.item()]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "original_title": title,
        "prediction": predicted_label,
        "all_labels": label_encoder
    })

# Feedback API
@app.post("/api/model/feedback")
def correct(title: str = Form(...), label: str = Form(...)):
    df = pd.DataFrame([[title, label]], columns=["Title", "Label"])
    if os.path.exists("user-labeled-titles.csv"):
        df.to_csv("user-labeled-titles.csv", mode="a", header=False, index=False)
    else:
        df.to_csv("user-labeled-titles.csv", index=False)
    return RedirectResponse("/", status_code=303)