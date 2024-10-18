from flask import Flask, render_template, request
import requests
import PyPDF2
import pandas as pd
import os

bot = Flask(__name__,template_folder='templates')

API_KEY = "4e08d219725c475a8623f1618397fc2e"
ENDPOINT = "https://lida.openai.azure.com/openai/deployments/college/chat/completions?api-version=2024-02-15-preview"

# Ensure the 'uploads' folder exists
os.makedirs('uploads', exist_ok=True)

def read_pdf(file_path):
    text_content = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text_content += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
    return text_content

def read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.to_string()  # Convert the entire DataFrame to a string
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return ""

@bot.route('/')
def index():
    return render_template('renderdash_frontend.html')  
@bot.route('/submit', methods=['POST'])
def submit():
    file = request.files['file']  
    question = request.form.get('question')  

    
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    file_extension = os.path.splitext(file.filename)[-1].lower()
    
    if file_extension == '.pdf':
        text_content = read_pdf(file_path)
    elif file_extension == '.csv':
        text_content = read_csv(file_path)
    else:
        return "Unsupported file type. Please upload a PDF or CSV file."

    
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

    # Prepare the question with text_content (limit size)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"{text_content[:2000]}? {question}"
            }
        ],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 800
    }

    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad responses
    except requests.RequestException as e:
        return f"Failed to make the request. Error: {e}"

    response_data = response.json()
    api_response = response_data.get("choices", [{}])[0].get("message", {}).get("content", "No content received from API.")

    return render_template('result.html', question=question, response=api_response)

if __name__ == '__main__':
    bot.run(debug=True)
