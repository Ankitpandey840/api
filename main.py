from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

catalog = [
    {"name": "Sales Potential Test", "text": "Sales Entry Communication Persuasion"},
    {"name": "Technical Aptitude Test", "text": "Tech Entry Problem Solving Logic"},
    {"name": "Managerial Assessment", "text": "Manager Mid Leadership Decision Making"},
    {"name": "Coding Simulation", "text": "Tech Mid Python Coding Problem Solving"},
    {"name": "Customer Support Test", "text": "Support Entry Empathy Communication"},
    {"name": "Data Analysis Test", "text": "Analytics Mid Data Interpretation Excel"},
]
catalog_embeddings = embedder.encode([c["text"] for c in catalog], convert_to_tensor=True)

class Query(BaseModel):
    text: str

@app.post("/recommend")
async def recommend(query: Query):
    input_embedding = embedder.encode(query.text, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(input_embedding, catalog_embeddings)[0]
    top_k = 3
    top_results = similarities.topk(k=top_k)

    retrieved = []
    for score, idx in zip(top_results.values, top_results.indices):
        retrieved.append(catalog[idx.item()]["name"])

    prompt = f"Job: {query.text}\nAssessments: {', '.join(retrieved)}\nWhich fits best and why?"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(**inputs, max_length=256)
    response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {"input": query.text, "top_matches": retrieved, "recommendation": response_text}
