import os
from openai import OpenAI
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def pdf_to_latex(file_path):
    try:
        reader = PdfReader(file_path)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text()
        response = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant that converts academic PDFs into clean LaTeX documents."},
            {"role": "user", "content": f"Convert the following text into a LaTeX document:\n{pdf_text}, do not output anything else, I just need the LaTeX code. Do not output any texts!"}
        ])

        latex_code = response.choices[0].message.content
        return latex_code
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Define the folder containing PDF files
input_folder = "/Users/yonganyu/Desktop/EDU benchmark/statistical physics"
output_folder = "/Users/yonganyu/Desktop/EDU benchmark/statistical_physics_latex"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".pdf"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".tex")

        print(f"Processing {filename}...")
        latex_content = pdf_to_latex(input_path)

        if latex_content:
            with open(output_path, "w", encoding="utf-8") as tex_file:
                tex_file.write(latex_content)
            print(f"Saved LaTeX to {output_path}")
        else:
            print(f"Failed to convert {filename}")
