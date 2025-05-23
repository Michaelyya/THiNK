import os
import json
import re
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import pandas as pd
from agents import evaluate_question_components
from metrics import calculate_quality_score
import re

login(token=API_KEY)

HUGGINGFACE_MODELS = [
    "mistralai/Ministral-8B-Instruct-2410"
]

class BaselineQuestionImprovement:
    def __init__(self, model_name: str = HUGGINGFACE_MODELS[0]):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="X:\.cache",
            torch_dtype="auto",
            device_map="auto"
        )

    def _clean_json_string(self, json_str: str) -> Dict:
        """Clean and parse JSON output from model, with improved error handling"""
        try:
            # First attempt: find JSON pattern and extract it
            json_pattern = r'(\{.*\})'
            match = re.search(json_pattern, json_str, re.DOTALL)
            if match:
                json_str = match.group(1)
            
            # Clean up markdown and special characters
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'\s*```', '', json_str)
            json_str = json_str.strip()
            
            # Fix common formatting errors
            json_str = re.sub(r'([^\\])"([^"]*)"s', r'\1"\2\'s', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            # Remove LaTeX formatting
            json_str = re.sub(r'\\[a-zA-Z]+', '', json_str)
            json_str = re.sub(r'\{[^}]*\}', '', json_str)
            
            try:
                parsed_json = json.loads(json_str)
                if "question" not in parsed_json or "solution" not in parsed_json:
                    raise ValueError("Missing required keys in JSON")
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"First attempt JSON parsing failed: {str(e)}")
                try:
                    alt_match = re.search(r'(\{\s*"question"\s*:\s*".*?"\s*,\s*"solution"\s*:\s*".*?"\s*\})', json_str, re.DOTALL)
                    if alt_match:
                        alt_json_str = alt_match.group(1)
                        return json.loads(alt_json_str)
                except Exception:
                    pass
                
                print(f"Could not parse JSON, returning default structure")
                return {
                    "question": "Error parsing question: " + json_str[:100] + "...",
                    "solution": "Error parsing solution"
                }
        except Exception as e:
            print(f"Error cleaning JSON string: {str(e)}")
            print(f"Problematic JSON: {json_str[:500]}...")
            return {
                "question": "Error parsing question",
                "solution": f"Error parsing solution: {str(e)}"
            }

    def improve_question(self, original_question: str, original_solution: str) -> Dict[str, str]:
        system_prompt = """You are an expert in mathematical problem design and improvement. Your task is to improve the given math question by:
1. Making the question clearer and more precise
2. Ensuring all necessary information is provided
3. Making the problem more engaging and challenging
4. Ensuring the solution is well-defined

IMPORTANT: Your response MUST be a valid JSON object with EXACTLY these fields:
{{
    "question": "The improved question text",
    "solution": "The improved solution approach"
}}

Do not include any additional text, explanations, or formatting outside the JSON object."""

        user_prompt = f"""Here is the original question and solution that needs improvement:

Original Question: {original_question}
Original Solution: {original_solution}

Please improve this question while maintaining its mathematical essence. Make it clearer, more precise, and more engaging. Provide both the improved question and solution."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize and generate
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1026,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            # Extract only the new tokens
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # Decode response
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print("\n----- RAW MODEL OUTPUT -----")
            print(response)
            print("----- END MODEL OUTPUT -----\n")
            
            # Try to parse as JSON directly
            try:
                cleaned_response = re.sub(r'```json\s*', '', response)
                cleaned_response = re.sub(r'\s*```', '', cleaned_response)
                cleaned_response = cleaned_response.strip()
                
                parsed_json = json.loads(cleaned_response)
                
                if "question" in parsed_json and "solution" in parsed_json:
                    return parsed_json
                else:
                    print("Missing required fields in JSON response")
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
            
            # If JSON parsing fails, try regex extraction
            question_match = re.search(r'"question"\s*:\s*"([^"]*(?:"[^"]*)*)"', response)
            solution_match = re.search(r'"solution"\s*:\s*"([^"]*(?:"[^"]*)*)"', response)
            
            question = question_match.group(1) if question_match else "Error extracting question"
            solution = solution_match.group(1) if solution_match else "Error extracting solution"
            

            return {
                "question": question,
                "solution": solution
            }
            
        except Exception as e:
            print(f"Error improving question: {str(e)}")
            return {
                "question": "Error improving question",
                "solution": f"Error: {str(e)}"
            }

def read_bad_questions(file_path="bad_questions.csv"):
    df = pd.read_csv(file_path)
    questions = []
    
    for _, row in df.iterrows():
        questions.append({
            "id": row.get("ID", ""),
            "question": row.get("question", ""),
            "solution": row.get("solution", ""),
            "math_concepts": [
                row.get("mathConcept1", ""),
                row.get("mathConcept2", ""),
                row.get("mathConcept3", "")
            ],
            "difficulty": row.get("Difficulty", ""),
            "grade": row.get("Grade", "")
        })
    
    return questions

def run_baseline_evaluation():
    # Initialize the baseline model
    baseline = BaselineQuestionImprovement()
    
    # Read bad questions
    questions = read_bad_questions()
    
    # Setup results directory and files
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, "baseline_mistral_8B.json")
    csv_path = os.path.join(results_dir, "baseline_mistral_8B.csv")
    
    # Load existing results if any
    existing_results = []
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                existing_results = json.load(f)
            print(f"Loaded {len(existing_results)} existing results")
        except Exception as e:
            print(f"Error loading existing results: {str(e)}")
    
    # Get already processed question IDs
    processed_ids = {r['question_id'] for r in existing_results}
    
    # Process each question
    for i, q in enumerate(questions):
        question_id = str(q["id"])
        
        # Skip if already processed
        if question_id in processed_ids:
            print(f"Skipping question {question_id} (already processed)")
            continue
        
        print(f"\nProcessing question {i+1}/{len(questions)} (ID: {question_id})")
        

        # Improve the question
        improved = baseline.improve_question(q["question"], q["solution"])
        
        # Evaluate the improved question
        evaluation_input = {
            "last_question": q["question"],
            "last_solution": q["solution"],
            "new_question": improved["question"],
            "new_solution": improved["solution"]
        }
        
        evaluation_result = evaluate_question_components(evaluation_input)
        print(evaluation_result)
        
        
        # Prepare result entry
        result_entry = {
            "question_id": question_id,
            "original_question": q["question"],
            "original_solution": q["solution"],
            "improved_question": improved["question"],
            "improved_solution": improved["solution"],
            "quality_score": evaluation_result.get('quality_score',0),
            "agent_scores": {
                agent: data.get("performance_score", 0)
                for agent, data in evaluation_result.get("evaluations", {}).items()
            },
            "evaluation_metrics": {
                "average_confidence": evaluation_result.get("average_confidence", 0),
                "pass_rate": evaluation_result.get("pass_rate", 0),
                "agent_agreement": evaluation_result.get("agent_agreement", 0)
            }
        }
        
        # Add to results
        existing_results.append(result_entry)
        
        # Save results after each question
        with open(results_path, 'w') as f:
            json.dump(existing_results, f, indent=2)
        
        # Save to CSV
        csv_data = []
        for result in existing_results:
            row = {
                "question_id": result["question_id"],
                "original_question": result["original_question"],
                "improved_question": result["improved_question"],
                "quality_score": result["quality_score"]
            }
            # Add agent scores
            for agent, score in result["agent_scores"].items():
                row[f"{agent}_score"] = score
            # Add evaluation metrics
            for metric, value in result["evaluation_metrics"].items():
                row[metric] = value
            csv_data.append(row)
        
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        
        print(f"Question {question_id} processed and saved")

        

    print("\nBaseline evaluation complete!")
    print(f"Results saved to {results_path}")
    print(f"CSV saved to {csv_path}")

if __name__ == "__main__":
    run_baseline_evaluation() 
