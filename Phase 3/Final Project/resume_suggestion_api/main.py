from typing import List, Optional
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM, pipeline as hf_pipeline
from peft import PeftModel
import onnxruntime as ort
import numpy as np
from collections import Counter
import re
from pathlib import Path as PathLib

# --- Pydantic models ---
class EducationEntry(BaseModel):
    school: str
    major: str
    start_year: str
    end_year: str

class ExperienceEntry(BaseModel):
    company: str
    title: str
    description: str
    start_year: str
    end_year: str

class ProjectEntry(BaseModel):
    name: str
    description: str

class PublicationEntry(BaseModel):
    name: str
    description: str

class ClassificationRequest(BaseModel):
    about: str = Field(..., description="Brief self-introduction")
    experience: Optional[List[ExperienceEntry]] = Field(default_factory=list)
    education: Optional[List[EducationEntry]] = Field(default_factory=list)
    projects: Optional[List[ProjectEntry]] = Field(default_factory=list)
    publications: Optional[List[PublicationEntry]] = Field(default_factory=list)
    courses: Optional[List[str]] = Field(default_factory=list)
    certifications: Optional[List[str]] = Field(default_factory=list)

class ClassificationResponse(BaseModel):
    label: str
    score: float

class KeywordsResponse(BaseModel):
    keywords: List[List[int]]  # [word, count]

class SkillEntry(BaseModel):
    skill: str
    score: float

class SkillListResponse(BaseModel):
    skills: List[SkillEntry]

class RewriteItem(BaseModel):
    original: str
    suggestion: str

class RewriteResponse(BaseModel):
    items: List[RewriteItem]

# --- Mappings ---
LABEL_MAPPING = {
    'LABEL_0': 'Data Analyst',
    'LABEL_1': 'Data Scientist',
    'LABEL_2': 'Project/Product/Program Manager',
    'LABEL_3': 'Software Engineer'
}

SKILL_FILES = {
    'da': 'skills/da_skills.txt',
    'ds': 'skills/ds_skills.txt',
    'pm': 'skills/pm_skills.txt',
    'swe': 'skills/swe_skills.txt'
}

# --- App initialization ---
app = FastAPI(title="Resume API")

# Load ONNX classification model
tokenizer = AutoTokenizer.from_pretrained("onnx-model")
session = ort.InferenceSession("onnx-model/model.onnx")

MODEL_DIR = "lora_llama2_resume"
tokenizer_rewrite = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
tokenizer_rewrite.pad_token_id = tokenizer_rewrite.eos_token_id
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model_rewrite = PeftModel.from_pretrained(base_model, MODEL_DIR, device_map="auto")

model_rewrite.eval()
gen_pipeline = hf_pipeline(
    "text-generation",
    model=model_rewrite,
    tokenizer=tokenizer_rewrite,
    device_map="auto",
    max_new_tokens=64,
    do_sample=True,
    top_p=0.9,
    temperature=0.8
)


def rewrite_bullet(bullet: str) -> str:
    prompt = f"Rewrite this resume bullet in FAAMG style:\n'{bullet}'\n\nAnswer:"
    out = gen_pipeline(prompt)
    text = out[0]["generated_text"]
    return text[len(prompt):].strip()

# Softmax helper
def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

# Utility to build input text
def build_input_text(req: ClassificationRequest) -> str:
    parts = [f"About: {req.about}"]
    if req.experience:
        exps = [f"{e.company} | {e.title} | {e.description} | {e.start_year} | {e.end_year}[EXP]" for e in req.experience]
        parts.append("Experience: " + " ".join(exps) + "[SEP]")
    if req.education:
        edus = [f"{ed.school} | {ed.major} | {ed.start_year} | {ed.end_year}[EDU]" for ed in req.education]
        parts.append("Education: " + " ".join(edus) + "[SEP]")
    if req.projects:
        projs = [f"{p.name} | {p.description}[PRO]" for p in req.projects]
        parts.append("Projects: " + " ".join(projs) + "[SEP]")
    if req.publications:
        pubs = [f"{p.name} | {p.description}[PUB]" for p in req.publications]
        parts.append("Publications: " + " ".join(pubs) + "[SEP]")
    if req.certifications:
        certs = [f"{c}[CER]" for c in req.certifications]
        parts.append("Certifications: " + " ".join(certs) + "[SEP]")
    if req.courses:
        cous = [f"{c}[COU]" for c in req.courses]
        parts.append("Courses: " + " ".join(cous) + "[SEP]")
    return " ".join(parts)

# --- Endpoints ---
@app.post("/predict", response_model=ClassificationResponse)
def predict(req: ClassificationRequest):
    try:
        text = build_input_text(req)
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='np')
        ort_outs = session.run(None, {k:v for k,v in inputs.items()})
        logits = ort_outs[0]
        probs = softmax(logits, axis=1)
        idx = int(np.argmax(probs, axis=1)[0])
        label = LABEL_MAPPING.get(f"LABEL_{idx}", f"LABEL_{idx}")
        return ClassificationResponse(label=label, score=float(probs[0][idx]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/keywords", response_model=KeywordsResponse)
def extract_keywords(req: ClassificationRequest):
    try:
        text = build_input_text(req)
        tokens = re.findall(r"\b\w+\b", text.lower())
        counts = Counter(tokens)
        top = counts.most_common(20)
        return KeywordsResponse(keywords=[[w,c] for w,c in top])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/skills/{role}", response_model=SkillListResponse)
def get_skills(role: str = Path(..., description="Role code: da, ds, pm, swe")):
    role = role.lower()
    if role not in SKILL_FILES:
        raise HTTPException(status_code=404, detail="Role not found. Use da, ds, pm, or swe.")
    file_path = PathLib(__file__).parent / SKILL_FILES[role]
    if not file_path.exists():
        raise HTTPException(status_code=500, detail=f"Skill file missing: {file_path}")
    skills=[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts=line.strip().split(None,1)
            if len(parts)==2:
                score_str, skill=parts
                try:
                    score=float(score_str)
                except:
                    continue
                skills.append(SkillEntry(skill=skill, score=score))
    return SkillListResponse(skills=skills)

@app.post("/rewrite", response_model=RewriteResponse)
def rewrite(req: ClassificationRequest):
    """Extract descriptions and generate rewrite suggestions"""
    try:
        bullets = []
        bullets += [e.description for e in req.experience]
        bullets += [p.description for p in req.projects]
        bullets += [q.description for q in req.publications]
        items = []
        for b in bullets:
            suggestion = rewrite_bullet(b)
            items.append(RewriteItem(original=b, suggestion=suggestion))
        return RewriteResponse(items=items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))