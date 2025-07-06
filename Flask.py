from flask import Flask, request, render_template, jsonify
import PyPDF2  # Using PyPDF2 instead of pymupdf
import docx
import ollama  # Assuming Ollama is used for LLM processing
import os
import re

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_resume_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    return ""

def extract_match_score(feedback):
    match = re.search(r'(\d{1,3})%', feedback)
    if match:
        return int(match.group(1))
    return 0

'''def evaluate_resume(job_desc, resume_text):
    prompt = f"""
    Evaluate the following resume against the job description and provide a match score (0-100%) with detailed feedback.
    
    Job Description:
    {job_desc}
    
    Resume:
    {resume_text}
    """
    response = ollama.chat(model='tinyllama', messages=[{"role": "user", "content": prompt}])
    feedback = response['message']['content']
    match_score = extract_match_score(feedback)
    return {"match_score": match_score, "feedback": feedback}'''

import requests

def evaluate_resume(job_desc, resume_text):
    prompt = f"""
    Evaluate the following resume against the job description and provide a match score (0-100%) with detailed feedback.

    Job Description:
    {job_desc}

    Resume:
    {resume_text}
    """

    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    data = {
        "model": "tinyllama",
        "prompt": prompt,
        "stream": False  # Ensure single response
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=data)
        response_json = response.json()  # Ensure valid JSON
        feedback = response_json.get("response", "No feedback received.")
        match_score = extract_match_score(feedback)
        return {"match_score": match_score, "feedback": feedback}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Ollama API error: {str(e)}"}
    except ValueError as ve:
        return {"error": f"Invalid JSON from Ollama: {str(ve)}"}



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        job_desc = request.form['job_description']
        resume_file = request.files['resume']
        
        if resume_file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(file_path)
            resume_text = extract_resume_text(file_path)
            result = evaluate_resume(job_desc, resume_text)
            return jsonify(result)
        
    return render_template('index.html')

@app.route('/validate_resume', methods=['POST'])
def validate_resume():
    data = request.form
    job_desc = data['job_description']
    resume_file = request.files['resume']
    
    if resume_file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
        resume_file.save(file_path)
        resume_text = extract_resume_text(file_path)
        result = evaluate_resume(job_desc, resume_text)
        return jsonify(result)
    
    return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    app.run(debug=True)
