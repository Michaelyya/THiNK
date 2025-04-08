import os
import json
import re
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import pandas as pd
from dotenv import load_dotenv
from agents import evaluate_question_components

# Load environment variables
load_dotenv()

API_KEY = ""
# Hugging Face configuration
# API_KEY = os.getenv("HUGGINGFACE_API_KEY")
# if not API_KEY:
#     raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")

login(token=API_KEY)

# Available models to test
HUGGINGFACE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct"
]

class ModelQuestionImprovementPipeline:
    def __init__(self, model_name: str, max_iterations: int = 5, quality_threshold: float = 0.9):
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.question_history = []
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure quantization
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="./model_cache",
            device_map={"": 0},
            quantization_config=config
        )
        self.model.gradient_checkpointing_enable()

    def _clean_json_string(self, json_str: str) -> Dict[str, str]:
        try:
            # First, try to find a complete JSON object
            json_pattern = r'\{[^{}]*\}'
            matches = re.findall(json_pattern, json_str)
            
            if matches:
                # Use the last complete JSON object found
                json_str = matches[-1]
            else:
                # If no complete JSON object found, try to extract question and solution separately
                question_match = re.search(r'(?:question|"question"):\s*"([^"]*)"', json_str)
                solution_match = re.search(r'(?:solution|"solution"):\s*"([^"]*)"', json_str)
                
                if question_match and solution_match:
                    return {
                        "question": question_match.group(1),
                        "solution": solution_match.group(1)
                    }
                raise ValueError("No valid JSON object found in response")
            
            # Clean up the JSON string
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'\s*```', '', json_str)
            json_str = json_str.strip()
            
            # Replace single quotes with double quotes
            json_str = re.sub(r"'", '"', json_str)
            
            # Ensure all property names are properly quoted
            json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
            
            # Remove any trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            # Remove any newlines and extra spaces
            json_str = re.sub(r'\s+', ' ', json_str)
            
            try:
                parsed_json = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {str(e)}")
                print(f"Problematic JSON: {json_str[:500]}...")
                # Try to extract just the question and solution using regex
                question_match = re.search(r'(?:question|"question"):\s*"([^"]*)"', json_str)
                solution_match = re.search(r'(?:solution|"solution"):\s*"([^"]*)"', json_str)
                
                if question_match and solution_match:
                    return {
                        "question": question_match.group(1),
                        "solution": solution_match.group(1)
                    }
                raise ValueError(f"Invalid JSON format: {str(e)}")
            
            # Validate required fields
            if not isinstance(parsed_json, dict):
                raise ValueError("Response is not a JSON object")
            
            if "question" not in parsed_json or "solution" not in parsed_json:
                raise ValueError("Missing required fields 'question' or 'solution'")
            
            if not isinstance(parsed_json["question"], str) or not isinstance(parsed_json["solution"], str):
                raise ValueError("Question and solution must be strings")
            
            return parsed_json
        except Exception as e:
            print(f"Error cleaning JSON string: {str(e)}")
            print(f"Problematic JSON: {json_str[:500]}...")
            raise ValueError(f"Error processing response: {str(e)}")

    def generate_question(self, previous_question: Dict[str, str] = None, 
                         improvement_suggestions: List[str] = None) -> Dict[str, str]:
        if previous_question is None:
            prompt = """Generate a challenging mathematical question that:
            1. Tests understanding of core concepts
            2. Involves practical applications
            3. Requires careful analytical thinking
            4. Has a clear solution approach
            
            IMPORTANT: Your response MUST be a valid JSON object with EXACTLY these fields:
            {
                "question": "The complete question text",
                "solution": "The detailed solution approach"
            }
            
            Do not include any additional text, explanations, or formatting outside the JSON object.
            The JSON must be properly formatted with double quotes for all property names and string values."""
        else:
            prompt = f"""I want you to be a mathematical problem-maker, and at the same time an expert in cognitive science, psychology, philosophy and education. As an LLM you can generate contents related to requirements, and now your purpose is to self-reflect on the process of your math problem generation process, analyzing what you have done.
            
            Previous question: {json.dumps(previous_question, indent=2)}
            
            Think out loud as you work on the instructions:
            1. Analyze the generated problem of the last round.
            2. Use cognitive skills essential for processing and applying information effectively.
            3. Generate mathematical expressions for the new problems.
            4. Generate stories for these mathematical expressions.
            5. Following and combining the previous steps, generate a new creative version.
            6. Review the generated new version math problem.
            
            IMPORTANT: Your response MUST be a valid JSON object with EXACTLY these fields:
            {{
                "question": "The complete question text",
                "solution": "The detailed solution approach"
            }}
            
            Do not include any additional text, explanations, or formatting outside the JSON object.
            The JSON must be properly formatted with double quotes for all property names and string values."""
            
            if improvement_suggestions:
                prompt += f"\n\nPlease also address these improvement suggestions:\n{json.dumps(improvement_suggestions, indent=2)}"

        try:
            # Generate response using the model
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            response_content = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean and parse the response
            return self._clean_json_string(response_content)
            
        except Exception as e:
            print(f"Error generating question: {str(e)}")
            return {
                "question": "Error generating question",
                "solution": f"Error: {str(e)}"
            }

    def run_pipeline(self, initial_question=None, initial_solution=None) -> Dict[str, Any]:
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
            print(f"Question: {current_question['question']}............FINI")
            
            evaluation_result = evaluate_question_components(evaluation_input)
            last_evaluation_result = evaluation_result
            
            print(f"\nQuality Score: {evaluation_result['quality_score']:.2f}")
            
            # Update the current question's evaluation in history
            self.question_history[-1]["evaluation"] = evaluation_result
            self.question_history[-1]["quality_score"] = evaluation_result["quality_score"]
            
            if evaluation_result["quality_score"] >= self.quality_threshold:
                print("\nSuccess! Question meets quality threshold.")
                return {
                    "final_question": current_question,
                    "iterations_required": iteration + 1,
                    "final_evaluation": evaluation_result,
                    "question_history": self.question_history
                }
            
            print("\nImproving question based on feedback...")
            current_question = self.generate_question(
                current_question,
                evaluation_result["improvement_suggestions"]
            )
            
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
    
    results = []
    round_metrics = []
    
    for i, q in enumerate(questions_to_process):
        print(f"\n\n{'=' * 50}")
        print(f"Processing Question {i+1}/{len(questions_to_process)}: ID {q['id']}")
        print(f"{'=' * 50}")
        
        pipeline = ModelQuestionImprovementPipeline(
            model_name=model_name,
            max_iterations=max_iterations,
            quality_threshold=0.7
        )
        
        try:
            result = pipeline.run_pipeline(
                initial_question=q["question"],
                initial_solution=q["solution"]
            )
            
            if "error" not in result:
                print(f"\nQuestion {q['id']} - Iterations Required: {result['iterations_required']}")
                print(f"Final Quality Score: {result['final_evaluation']['quality_score']:.2f}")
                
                # Extract metrics for each round
                question_round_metrics = []
                for history_entry in result['question_history']:
                    if history_entry.get('evaluation'):
                        eval_data = history_entry['evaluation']
                        current_question = history_entry['question']
                        
                        # Extract performance scores for each agent
                        agent_scores = {}
                        for agent_name, agent_data in eval_data.get('evaluations', {}).items():
                            agent_scores[f'{agent_name}_score'] = agent_data.get('performance_score', 0)
                        
                        round_metric = {
                            'round': len(question_round_metrics) + 1,
                            'question_id': q['id'],
                            'current_question': current_question.get('question', ''),
                            'current_solution': current_question.get('solution', ''),
                            'average_confidence': eval_data.get('average_confidence', 0),
                            'pass_rate': eval_data.get('pass_rate', 0),
                            'agent_agreement': eval_data.get('agent_agreement', 0),
                            'quality_score': eval_data.get('quality_score', 0),
                            'improvement_suggestions': eval_data.get('improvement_suggestions', []),
                            **agent_scores
                        }
                        question_round_metrics.append(round_metric)
                
                round_metrics.extend(question_round_metrics)
                
                # Calculate average performance scores across all iterations
                avg_agent_scores = {}
                for agent_name in ['remembering', 'understanding', 'applying', 'analyzing', 'evaluating', 'creating', 'language']:
                    scores = [m.get(f'{agent_name}_score', 0) for m in question_round_metrics]
                    avg_agent_scores[f'avg_{agent_name}_score'] = sum(scores) / len(scores) if scores else 0
                
                results.append({
                    "question_id": q['id'],
                    "original_question": q["question"],
                    "original_solution": q["solution"],
                    "final_question": result['final_question']['question'],
                    "final_solution": result['final_question']['solution'],
                    "final_score": result['final_evaluation']['quality_score'],
                    "iterations_required": result['iterations_required'],
                    "success": result['final_evaluation']['quality_score'] >= 0.7,
                    "round_metrics": question_round_metrics,
                    **avg_agent_scores
                })
            else:
                print(f"\nError processing question {q['id']}: {result['error']}")
                results.append({
                    "question_id": q['id'],
                    "original_question": q["question"],
                    "original_solution": q["solution"],
                    "error": result['error'],
                    "success": False
                })
        except Exception as e:
            print(f"\nException processing question {q['id']}: {str(e)}")
            results.append({
                "question_id": q['id'],
                "original_question": q["question"],
                "original_solution": q["solution"],
                "error": str(e),
                "success": False
            })
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save JSON results
    with open(f"{results_dir}/{model_name.replace('/', '_')}_results.json", "w") as f:
        json.dump({
            "summary": results,
            "round_metrics": round_metrics
        }, f, indent=2)
    
    # Save CSV metrics
    csv_metrics = []
    for metric in round_metrics:
        metric_copy = metric.copy()
        if 'improvement_suggestions' in metric_copy:
            metric_copy['improvement_suggestions'] = '; '.join(str(s) for s in metric_copy['improvement_suggestions'] if isinstance(s, str) and len(s) > 1)
        csv_metrics.append(metric_copy)
    
    metrics_df = pd.DataFrame(csv_metrics)
    metrics_df.to_csv(f"{results_dir}/{model_name.replace('/', '_')}_metrics.csv", index=False)
    
    print(f"\nResults saved to {results_dir}/{model_name.replace('/', '_')}_results.json")
    print(f"Metrics saved to {results_dir}/{model_name.replace('/', '_')}_metrics.csv")

def main():
    # Test each model
    for model_name in HUGGINGFACE_MODELS:
        print(f"\n\n{'=' * 50}")
        print(f"Testing model: {model_name}")
        print(f"{'=' * 50}")
        
        try:
            run_model_evaluation(
                model_name=model_name,
                num_questions=1,
                max_iterations=3
            )
        except Exception as e:
            print(f"Error testing model {model_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 