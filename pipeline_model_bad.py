import os
import json
import re
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import pandas as pd
from dotenv import load_dotenv
from agents import evaluate_question_components

# Load environment variables
load_dotenv()

API_KEY = "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"
# Hugging Face configuration
# API_KEY = os.getenv("HUGGINGFACE_API_KEY")
# if not API_KEY:
#     raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")

login(token=API_KEY)

HUGGINGFACE_MODELS = [
    "Qwen/Qwen2.5-14B-Instruct"
]

class ModelQuestionImprovementPipeline:
    def __init__(self, model_name: str, max_iterations: int = 5, quality_threshold: float = 0.9):
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.question_history = []
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with auto device mapping
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
            # Fix unescaped quotes
            json_str = re.sub(r'([^\\])"([^"]*)"s', r'\1"\2\'s', json_str)
            # Handle trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            # Remove LaTeX formatting
            json_str = re.sub(r'\\[a-zA-Z]+', '', json_str)
            json_str = re.sub(r'\{[^}]*\}', '', json_str)
            
            # Try to parse JSON
            try:
                parsed_json = json.loads(json_str)
                # Ensure the required keys are present
                if "question" not in parsed_json or "solution" not in parsed_json:
                    raise ValueError("Missing required keys in JSON")
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"First attempt JSON parsing failed: {str(e)}")
                # Second attempt: find the most complete JSON-like structure
                try:
                    # Try to extract just the JSON object
                    alt_match = re.search(r'(\{\s*"question"\s*:\s*".*?"\s*,\s*"solution"\s*:\s*".*?"\s*\})', json_str, re.DOTALL)
                    if alt_match:
                        alt_json_str = alt_match.group(1)
                        return json.loads(alt_json_str)
                except Exception:
                    pass
                
                # If all else fails, return a default structure
                print(f"Could not parse JSON, returning default structure")
                return {
                    "question": "Error parsing question: " + json_str[:100] + "...",
                    "solution": "Error parsing solution"
                }
        except Exception as e:
            print(f"Error cleaning JSON string: {str(e)}")
            print(f"Problematic JSON: {json_str[:500]}...")
            # Return a default structure
            return {
                "question": "Error parsing question",
                "solution": f"Error parsing solution: {str(e)}"
            }
        

    def generate_question(self, previous_question: Dict[str, str] = None, 
                        improvement_suggestions: List[str] = None) -> Dict[str, str]:
        if previous_question is None:
            system_prompt = "You are a helpful assistant specialized in generating mathematical questions."
            user_prompt = """Generate a challenging mathematical question that:
            1. Tests understanding of core concepts
            2. Involves practical applications
            3. Requires careful analytical thinking
            4. Has a clear solution approach
            
            IMPORTANT: Your response MUST be a valid JSON object with EXACTLY these fields:
            {
                "question": "The complete question text",
                "solution": "The detailed solution approach"
            }
            
            Do not include any other text, explanation, or formatting - ONLY the JSON object."""
        else:
            system_prompt = "You are a helpful assistant specialized in generating mathematical questions."
            
            # Simplified prompt to avoid any confusion
            user_prompt = f"""Improve this math question:
            
            Question: {previous_question.get("question", "")}
            
            Solution: {previous_question.get("solution", "")}
            
            IMPORTANT: Your response MUST be ONLY a valid JSON object as shown below:
            {{
                "question": "your improved question here",
                "solution": "your improved solution here"
            }}
            
            Do not include ANY other text, explanations, or formatting - JUST the JSON object."""
            
            if improvement_suggestions and len(improvement_suggestions) > 0:
                suggestions_text = "\n".join([f"- {sugg}" for sugg in improvement_suggestions if sugg and isinstance(sugg, str)])
                if suggestions_text:
                    user_prompt += f"\n\nConsider these improvement suggestions:\n{suggestions_text}"

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
            
            # IMPORTANT: Print the complete raw response for debugging
            print("\n----- RAW MODEL OUTPUT -----")
            print(response)
            print("----- END MODEL OUTPUT -----\n")
            
            # Try to parse as JSON directly
            try:
                # First clean up markdown formatting if present
                cleaned_response = re.sub(r'```json\s*', '', response)
                cleaned_response = re.sub(r'\s*```', '', cleaned_response)
                cleaned_response = cleaned_response.strip()
                
                # Try to parse the cleaned response
                parsed_json = json.loads(cleaned_response)
                
                # Check if it has the required fields
                if "question" in parsed_json and "solution" in parsed_json:
                    return parsed_json
                else:
                    print("Missing required fields in JSON response")
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
            
            # If we get here, JSON parsing failed - extract what we can
            # Try to extract question and solution using regex
            question_match = re.search(r'"question"\s*:\s*"([^"]*(?:"[^"]*)*)"', response)
            solution_match = re.search(r'"solution"\s*:\s*"([^"]*(?:"[^"]*)*)"', response)
            
            question = question_match.group(1) if question_match else "Error extracting question"
            solution = solution_match.group(1) if solution_match else "Error extracting solution"
            
            # If we couldn't extract anything useful with regex, just use a portion of the response
            if question == "Error extracting question" and solution == "Error extracting solution":
                # Just return a useful error with part of the response
                return {
                    "question": f"Model did not generate valid JSON. Partial output: {response[:200]}...",
                    "solution": "Error: Unable to parse model output as JSON"
                }
            
            return {
                "question": question,
                "solution": solution
            }
            
        except Exception as e:
            print(f"Error generating question: {str(e)}")
            return {
                "question": "Error generating question",
                "solution": f"Error: {str(e)}"
            }
    
    def run_pipeline(self, initial_question=None, initial_solution=None) -> Dict[str, Any]:
        try:
            if initial_question and initial_solution:
                current_question = {
                    "question": initial_question,
                    "solution": initial_solution
                }
            else:
                current_question = self.generate_question()
                
            self.question_history.append({
                "question": current_question["question"],
                "solution": current_question["solution"],
                "evaluation": None,
                "quality_score": 0.0
            })
            
            iteration = 0
            last_evaluation_result = None
            
            while iteration < self.max_iterations:
                print(f"\n=== Iteration {iteration + 1} ===")
                
                # Get the last question from history if it exists
                last_question = self.question_history[-2]["question"] if len(self.question_history) > 1 else ""
                last_solution = self.question_history[-2]["solution"] if len(self.question_history) > 1 else ""
                
                evaluation_input = {
                    "last_question": last_question,
                    "last_solution": last_solution,
                    "new_question": current_question["question"],
                    "new_solution": current_question["solution"]
                }
                
                print("\nEvaluating question:")
                print(f"Question: {current_question['question']}")
                
                evaluation_result = evaluate_question_components(evaluation_input)
                last_evaluation_result = evaluation_result
                
                # Ensure evaluation_result is a dictionary
                if not isinstance(evaluation_result, dict):
                    print(f"Warning: evaluation_result is not a dictionary: {type(evaluation_result)}")
                    evaluation_result = {
                        "quality_score": 0.0,
                        "improvement_suggestions": []
                    }
                
                # Ensure quality_score exists
                quality_score = evaluation_result.get("quality_score", 0.0)
                print(f"\nQuality Score: {quality_score:.2f}")
                
                # Update the current question's evaluation in history
                self.question_history[-1]["evaluation"] = evaluation_result
                self.question_history[-1]["quality_score"] = quality_score
                
                if quality_score >= self.quality_threshold:
                    print("\nSuccess! Question meets quality threshold.")
                    return {
                        "final_question": current_question,
                        "iterations_required": iteration + 1,
                        "final_evaluation": evaluation_result,
                        "question_history": self.question_history
                    }
                
                print("\nImproving question based on feedback...")
                print(f"Current question: {current_question['question']}")
                
                try:
                    # Get improvement suggestions directly from evaluation_result
                    improvement_suggestions = evaluation_result.get("improvement_suggestions", [])
                    print(f"Improvement suggestions: {improvement_suggestions}")
                    
                    current_question = self.generate_question(
                        current_question,
                        improvement_suggestions
                    )
                    print(f"Improved question: {current_question['question']}")
                except Exception as e:
                    print(f"Error generating improved question: {str(e)}")
                    return {
                        "final_question": current_question,
                        "iterations_required": iteration + 1,
                        "final_evaluation": evaluation_result,
                        "question_history": self.question_history,
                        "error": str(e)
                    }
                
                # Add the new question to history
                self.question_history.append({
                    "question": current_question["question"],
                    "solution": current_question["solution"],
                    "evaluation": None,
                    "quality_score": 0.0
                })
                
                iteration += 1
            
            return {
                "final_question": current_question,
                "iterations_required": iteration,
                "final_evaluation": last_evaluation_result,
                "question_history": self.question_history,
                "warning": "Maximum iterations reached without meeting quality threshold."
            }
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            return {
                "error": str(e),
                "question_history": self.question_history
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

def run_model_evaluation(model_name: str, num_questions: int = 1, max_iterations: int = 3):
    questions = read_bad_questions()
    questions_to_process = questions[:num_questions]
    
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load existing results if any
    results_file = f"{results_dir}/{model_name.replace('/', '_')}_results.json"
    metrics_file = f"{results_dir}/{model_name.replace('/', '_')}_metrics.csv"
    
    existing_results = []
    processed_ids = set()
    
    # Load existing results
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                existing_results = json.load(f).get('summary', [])
                processed_ids = {r['question_id'] for r in existing_results}
        except Exception as e:
            print(f"Error loading existing results: {str(e)}")
    
    # Load existing metrics
    existing_metrics = []
    if os.path.exists(metrics_file):
        try:
            existing_metrics = pd.read_csv(metrics_file).to_dict('records')
        except Exception as e:
            print(f"Error loading existing metrics: {str(e)}")
    
    results = existing_results.copy()
    round_metrics = existing_metrics.copy()
    
    for i, q in enumerate(questions_to_process):
        # Skip if already processed
        if q['id'] in processed_ids:
            print(f"\nSkipping already processed question {q['id']}")
            continue
            
        print(f"\n\n{'=' * 50}")
        print(f"Processing Question {i+1}/{len(questions_to_process)}: ID {q['id']}")
        print(f"{'=' * 50}")
        
        pipeline = ModelQuestionImprovementPipeline(
            model_name=model_name,
            max_iterations=max_iterations,
            quality_threshold=0.85
        )
        
        try:
            result = pipeline.run_pipeline(
                initial_question=q["question"],
                initial_solution=q["solution"]
            )
            
            if "error" not in result:
                print(f"\nQuestion {q['id']} - Iterations Required: {result['iterations_required']}")
                print(f"Final Quality Score: {result['final_evaluation'].get('quality_score', 0.0):.2f}")
                
                # Extract metrics for each round
                question_round_metrics = []
                for idx, history_entry in enumerate(result['question_history']):
                    # Skip entries with no evaluation
                    if not history_entry.get('evaluation'):
                        continue
                    
                    eval_data = history_entry['evaluation']
                    
                    # Make sure we're dealing with a dictionary
                    if not isinstance(eval_data, dict):
                        print(f"Warning: evaluation data is not a dictionary: {type(eval_data)}")
                        continue
                    
                    # Safely extract current_question
                    current_question = history_entry.get('question', '')
                    current_solution = history_entry.get('solution', '')
                    
                    # Make sure they're strings
                    if isinstance(current_question, dict):
                        current_question = current_question.get('question', '')
                    if isinstance(current_solution, dict):
                        current_solution = current_solution.get('solution', '')
                    
                    # Extract performance scores for each agent
                    agent_scores = {}
                    for agent_name, agent_data in eval_data.get('evaluations', {}).items():
                        if isinstance(agent_data, dict):
                            agent_scores[f'{agent_name}_score'] = agent_data.get('performance_score', 0)
                    
                    # Create metric record with safe getters
                    round_metric = {
                        'round': idx + 1,
                        'question_id': q['id'],
                        'current_question': current_question,
                        'current_solution': current_solution,
                        'average_confidence': eval_data.get('average_confidence', 0),
                        'pass_rate': eval_data.get('pass_rate', 0),
                        'agent_agreement': eval_data.get('agent_agreement', 0),
                        'quality_score': eval_data.get('quality_score', 0),
                    }
                    
                    # Safely add improvement suggestions
                    improvement_suggestions = eval_data.get('improvement_suggestions', [])
                    if improvement_suggestions:
                        # Make sure it's a list
                        if not isinstance(improvement_suggestions, list):
                            improvement_suggestions = [str(improvement_suggestions)]
                        round_metric['improvement_suggestions'] = improvement_suggestions
                    
                    # Add agent scores
                    round_metric.update(agent_scores)
                    
                    question_round_metrics.append(round_metric)
                
                round_metrics.extend(question_round_metrics)
                
                # Calculate average performance scores across all iterations
                avg_agent_scores = {}
                for agent_name in ['remembering', 'understanding', 'applying', 'analyzing', 'evaluating', 'creating', 'language']:
                    scores = [m.get(f'{agent_name}_score', 0) for m in question_round_metrics if f'{agent_name}_score' in m]
                    avg_agent_scores[f'avg_{agent_name}_score'] = sum(scores) / len(scores) if scores else 0
                
                # Gather all the necessary data with safe getters
                final_quality_score = 0.0
                if isinstance(result.get('final_evaluation'), dict):
                    final_quality_score = result['final_evaluation'].get('quality_score', 0.0)
                
                new_result = {
                    "question_id": q['id'],
                    "original_question": q["question"],
                    "original_solution": q["solution"],
                    "final_question": result.get('final_question', {}).get('question', ''),
                    "final_solution": result.get('final_question', {}).get('solution', ''),
                    "final_score": final_quality_score,
                    "iterations_required": result.get('iterations_required', 0),
                    "success": final_quality_score >= 0.8,
                    "round_metrics": question_round_metrics,
                    **avg_agent_scores
                }
                
                results.append(new_result)
                
                # Save results after each question
                with open(results_file, 'w') as f:
                    json.dump({
                        "summary": results,
                        "round_metrics": round_metrics
                    }, f, indent=2)
                
                # Save metrics to CSV
                csv_metrics = []
                for metric in round_metrics:
                    metric_copy = dict(metric)  # Create a copy to avoid modifying the original
                    if 'improvement_suggestions' in metric_copy:
                        sugg_list = metric_copy['improvement_suggestions']
                        if isinstance(sugg_list, list):
                            # Convert list to string and cleanup
                            metric_copy['improvement_suggestions'] = '; '.join(str(s) for s in sugg_list if isinstance(s, (str, int, float)) and str(s).strip())
                        else:
                            # If not a list, convert to string
                            metric_copy['improvement_suggestions'] = str(sugg_list)
                    csv_metrics.append(metric_copy)
                
                if csv_metrics:
                    metrics_df = pd.DataFrame(csv_metrics)
                    metrics_df.to_csv(metrics_file, index=False)
                    print(f"Metrics saved to {metrics_file}")
                
                print(f"Results saved for question {q['id']}")
                
            else:
                print(f"\nError processing question {q['id']}: {result['error']}")
                results.append({
                    "question_id": q['id'],
                    "original_question": q["question"],
                    "original_solution": q["solution"],
                    "error": result['error'],
                    "success": False
                })
                
                # Save error result
                with open(results_file, 'w') as f:
                    json.dump({
                        "summary": results,
                        "round_metrics": round_metrics
                    }, f, indent=2)
                
        except Exception as e:
            print(f"\nException processing question {q['id']}: {str(e)}")
            results.append({
                "question_id": q['id'],
                "original_question": q["question"],
                "original_solution": q["solution"],
                "error": str(e),
                "success": False
            })
            
            # Save error result
            with open(results_file, 'w') as f:
                json.dump({
                    "summary": results,
                    "round_metrics": round_metrics
                }, f, indent=2)
    
    print(f"\nAll results saved to {results_file}")
    print(f"All metrics saved to {metrics_file}")

def main():
    # Test each model
    for model_name in HUGGINGFACE_MODELS:
        print(f"\n\n{'=' * 50}")
        print(f"Testing model: {model_name}")
        print(f"{'=' * 50}")
        
        try:
            run_model_evaluation(
                model_name=model_name,
                num_questions=20,
                max_iterations=3
            )
        except Exception as e:
            print(f"Error testing model {model_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()