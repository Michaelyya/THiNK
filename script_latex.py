import os
import subprocess
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def pdf_to_latex(file_path):
    try:
        # Convert PDF to LaTeX using pdf2latex (assuming it's installed)
        output_path = file_path.replace(".pdf", ".tex")
        subprocess.run(["pdf2latex", file_path, "-output", output_path], check=True)
        
        # Read the LaTeX content from the generated file
        with open(output_path, "r", encoding="utf-8") as tex_file:
            latex_content = tex_file.read()
    

        # Request GPT-4 refinement, but only return the LaTeX code (no extra chat)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that refines LaTeX documents."},
                {"role": "user", "content": f"Refine the following LaTeX document:\n\n{latex_content}"}
            ]
        )
        
        # Extract the refined LaTeX content and ensure it doesn't include unnecessary responses
        refined_latex = response["choices"][0]["message"]["content"].strip()
        return refined_latex
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Define the folder containing PDF files
input_folder = "/Users/yonganyu/Desktop/EDU benchmark/Calculus"
output_folder = "/Users/yonganyu/Desktop/EDU benchmark/Calculus/latex"
os.makedirs(output_folder, exist_ok=True)

# Iterate through each PDF in the folder and convert to LaTeX
for filename in os.listdir(input_folder):
    if filename.endswith(".pdf"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".tex")

        print(f"Processing {filename}...")
        latex_content = pdf_to_latex(input_path)

        if latex_content:
            # Save the refined LaTeX content to a .tex file
            with open(output_path, "w", encoding="utf-8") as tex_file:
                tex_file.write(latex_content)
            print(f"Saved LaTeX to {output_path}")
        else:
            print(f"Failed to convert {filename}")
