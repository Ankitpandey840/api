from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import pandas as pd

app = Flask(__name__)

# Load smaller embedding model for memory optimization
embedder = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# SHL Product Catalog (Mock Data)
data = [
    {"id": 1, "name": "Sales Potential Test", "role": "Sales", "seniority": "Entry", "skills": "Communication, Persuasion"},
    {"id": 2, "name": "Technical Aptitude Test", "role": "Tech", "seniority": "Entry", "skills": "Problem Solving, Logic"},
    {"id": 3, "name": "Managerial Assessment", "role": "Manager", "seniority": "Mid", "skills": "Leadership, Decision Making"},
    {"id": 4, "name": "Coding Simulation", "role": "Tech", "seniority": "Mid", "skills": "Python, Coding, Problem Solving"},
    {"id": 5, "name": "Customer Support Test", "role": "Support", "seniority": "Entry", "skills": "Empathy, Communication"},
    {"id": 6, "name": "Data Analysis Test", "role": "Analytics", "seniority": "Mid", "skills": "Data Interpretation, Excel"},
]
catalog = pd.DataFrame(data)
catalog['text'] = catalog['role'] + " " + catalog['seniority'] + " " + catalog['skills'] + " " + catalog['name']
catalog_embeddings = embedder.encode(catalog['text'].tolist(), convert_to_tensor=True)

@app.route('/recommend', methods=['POST'])
def recommend():
    content = request.json
    job_input = content.get("job_description", "")

    if not job_input.strip():
        return jsonify({"error": "Job description is empty."}), 400

    input_embedding = embedder.encode(job_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(input_embedding, catalog_embeddings)[0]
    top_k = min(3, len(similarities))
    top_results = similarities.topk(k=top_k)

    top_assessments = []
    for score, idx in zip(top_results.values, top_results.indices):
        item = catalog.iloc[idx.item()]
        top_assessments.append({
            "name": item['name'],
            "role": item['role'],
            "seniority": item['seniority'],
            "skills": item['skills']
        })

    return jsonify({
        "job_description": job_input,
        "top_matches": top_assessments
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
