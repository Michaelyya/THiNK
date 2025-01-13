import os
import json
import openai
from typing import Dict, List
from dotenv import load_dotenv
load_dotenv()

class LatexExtractor:
    def __init__(self, api_key: str, content_dictionary: Dict[str, List[str]]):
        self.client = openai.OpenAI(api_key=api_key)
        self.content_dictionary = content_dictionary

    def read_latex_file(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def create_extraction_prompt(self, problem_latex: str, solution_latex: str) -> str:
        return f"""
Task: Extract individual questions and their corresponding solutions from LaTeX files, and categorize them according to the given content dictionary.

Content Dictionary:
{json.dumps(self.content_dictionary, indent=2)}

Problem LaTeX Content:
{problem_latex}

Solution LaTeX Content:
{solution_latex}

Instructions:
1. Identify each distinct question in the problem file
2. Find the corresponding solution in the solution file
3. For each question-solution pair:
   - Assign a sequential ID
   - Determine the most appropriate content category from the dictionary
   - Clean and format the LaTeX content for both question and solution
4. Output the results in the following JSON format:
{{
    content:[
        {{
            "id": <sequential_number>,
            "category": "Linear Algebra I",
            "corresponding_content": "<specific_topic_from_dictionary>",
            "content_dictionary": <full_content_dictionary>,
            "question": "<cleaned_latex_question>",
            "answer": "<cleaned_latex_solution>"
        }}
    ]
}}

Please ensure:
- All LaTeX formatting is preserved
- Each question-answer pair is properly matched
- Content categorization is accurate based on the provided dictionary

NOTE: Please make sure the output is a valid JSON
"""

    def process_file_pair(self, problem_path: str, solution_path: str) -> Dict:
        problem_content = self.read_latex_file(problem_path)
        solution_content = self.read_latex_file(solution_path)
        
        prompt = self.create_extraction_prompt(problem_content, solution_content)
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a LaTeX parsing assistant that extracts and structures mathematical content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10000,
            response_format={ "type": "json_object" } 
        )
        response_content = response.choices[0].message.content
        if response_content.startswith("```json"):
            response_content = response_content.replace("```json", "", 1)
        if response_content.endswith("```"):
            response_content = response_content.rsplit("```", 1)[0]
        response_content = response_content.strip()
        print(response_content)
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            print(f"Error parsing response for {problem_path}: {e}")
            return None

    def process_all_files(self, directory: str) -> List[Dict]:
        all_questions = []
        
        for i in range(1, 14): 
            print(i)
            problem_file = f"T{i}.tex"
            solution_file = f"T{i} Solution.tex"
            
            problem_path = os.path.join(directory, problem_file)
            solution_path = os.path.join(directory, solution_file)
            
            if os.path.exists(problem_path) and os.path.exists(solution_path):
                result = self.process_file_pair(problem_path, solution_path)
                if result:
                    print("finish here")
                    all_questions.extend(result["content"])
            else:
                print(f"Missing files for T{i}")
        
        return all_questions

    def save_benchmark(self, questions: List[Dict], output_file: str = "benchmark.json"):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"content": questions}, f, indent=2)

if __name__ == "__main__":
    content_dict = {
        "Linear Algebra I": [
            "Linear Systems and Gaussian Elimination",
            "Vectors, Spans, and Subspaces",
            "Linear Transformations",
            "Matrix Operations and Applications",
            "Determinants and Their Applications",
            "Eigenvalues, Eigenvectors, and Diagonalization"
        ]
    }
    
    extractor = LatexExtractor(
        api_key=os.environ.get("OPENAI_API_KEY"),
        content_dictionary=content_dict
    )
    
    questions = extractor.process_all_files("./Linear Algebra/problem sets & solutions latex")
    
    extractor.save_benchmark(questions)