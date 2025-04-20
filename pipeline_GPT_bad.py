from typing import Dict, List, Any
import json
import os
import pandas as pd
from dotenv import load_dotenv
import re
from agents import evaluate_question_components
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

class QuestionImprovementPipeline:
    def __init__(self, max_iterations: int = 3, quality_threshold: float = 0.85):
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.question_history = []


    def _clean_json_string(self, json_str: str) -> Dict[str, str]:
        json_pattern = r'(\{.*\})'
        match = re.search(json_pattern, json_str, re.DOTALL)
        if match:
            json_str = match.group(1)
        
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'\s*```', '', json_str)
        json_str = json_str.strip()
        
        parsed_json = json.loads(json_str)
        
        if not isinstance(parsed_json, dict):
            raise ValueError("Response is not a JSON object")
        
        if "question" not in parsed_json or "solution" not in parsed_json:
            raise ValueError("Missing required fields 'question' or 'solution'")
        
        if not isinstance(parsed_json["question"], str) or not isinstance(parsed_json["solution"], str):
            raise ValueError("Question and solution must be strings")
        
        return parsed_json
        
    def generate_question(self, previous_question: Dict[str, str] = None, 
                         improvement_suggestions: List[str] = None) -> Dict[str, str]:
        if previous_question is None:
            initial_prompt = """Generate a challenging mathematical question that:
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
            
            messages = [
                {"role": "system", "content": "You are an expert mathematical problem designer. You must respond with a valid JSON object."},
                {"role": "user", "content": initial_prompt}
            ]
        else:
            print(previous_question)

            improvement_prompt = """I want you to be a mathematical problem-maker, and at the same time an expert in cognitive science, psychology, philosophy and education. As an LLM you can generate contents related to requirements, and now your purpose is to self-reflect on the process of your math problem generation process, analyzing what you have done."""
            improvement_prompt += json.dumps(previous_question, indent=2)
            improvement_prompt += """Remember, this is your problem generation outcome last time

    Think out loud as you work on the instructions:
    1. Analyze the generated problem of the last round. You should try to understand and retrieve the specific mathematical information in it such as facts, patterns, objects, or contextual information, and decipher these meanings.
    2. Use cognitive skills essential for processing and applying information effectively. It includes understanding and organizing information, analyzing relationships, drawing conclusions, and distinguishing nuances. Additionally, you should evaluate ideas critically.
    3. Generate mathematical expressions for the new problems. These new expressions should have the same form as the given expressions in the previous generated math problem. They must have the same complexity as well. Choose values to substitute into the expression, and calculate the outputs.
    4. Generate stories for these mathematical expressions with the appropriate questions based on the chosen values. The generated stories must be a mathematical word problem with the corresponding expressions. The story must be creative and unique.
    5. Following and combining the previous steps, and you will generate a new creative version of the given math problem.
    6. Review the generated new version math problem, ensuring all the criteria are satisfied and double check it.

    IMPORTANT: Your response MUST be a valid JSON object with EXACTLY these fields:
    {
        "question": "The complete question text",
        "solution": "The detailed solution approach"
    }
    
    Do not include any additional text, explanations, or formatting outside the JSON object.
    The JSON must be properly formatted with double quotes for all property names and string values."""
            if improvement_suggestions:
                improvement_prompt += f"\n\nPlease also address these improvement suggestions:\n{json.dumps(improvement_suggestions, indent=2)}"
            print(improvement_prompt)
            messages = [
                {"role": "system", "content": "You are an expert mathematical problem designer. You must respond with a valid JSON object."},
                {"role": "user", "content": improvement_prompt}
            ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0
            )
            
            response_content = response.choices[0].message.content
            
            # Try to find JSON content
            json_pattern = r'(\{.*\})'
            match = re.search(json_pattern, response_content, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    parsed_json = json.loads(json_str)
                    if isinstance(parsed_json, dict) and "question" in parsed_json and "solution" in parsed_json:
                        return parsed_json
                except json.JSONDecodeError:
                    pass
            
            # If JSON parsing fails, try to extract question and solution using regex
            question_match = re.search(r'(?:question|"question"):\s*"([^"]*)"', response_content)
            solution_match = re.search(r'(?:solution|"solution"):\s*"([^"]*)"', response_content)
            
            if question_match and solution_match:
                return {
                    "question": question_match.group(1),
                    "solution": solution_match.group(1)
                }
            
            # If all else fails, return a default response
            return {
                "question": response_content.strip(),
                "solution": "No explicit solution provided. Please refer to the question for details."
            }
            
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
            "question": current_question,
            "evaluation": None,
            "quality_score": 0.0
        })
        
        iteration = 0
        last_evaluation_result = None
        
        while iteration < self.max_iterations:
            print(f"\n=== Iteration {iteration + 1} ===")
            
            evaluation_input = {
                "last_question": self.question_history[-2]["question"]["question"] if len(self.question_history) > 1 else "",
                "last_solution": self.question_history[-2]["question"]["solution"] if len(self.question_history) > 1 else "",
                "new_question": current_question["question"],
                "new_solution": current_question["solution"]
            }
            
            print("\nEvaluating question:")
            print(f"Question: {current_question['question']}............FINI")
            
            evaluation_result = evaluate_question_components(evaluation_input)
            last_evaluation_result = evaluation_result
            
            print(f"\nQuality Score: {evaluation_result['quality_score']:.2f}")
            
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
            print(current_question["question"])
            current_question = self.generate_question(
                current_question,
                evaluation_result["improvement_suggestions"]
            )
            print(current_question["question"])
            
            self.question_history.append({
                "question": current_question,
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


def analyze_thinking_levels(history):
    if not history or not history[0].get("evaluation"):
        return {}
    
    # Get all evaluation levels
    levels = history[0]["evaluation"]["evaluations"].keys()
    
    # Extract scores for each level across iterations
    level_progressions = {level: [] for level in levels}
    
    for entry in history:
        if entry.get("evaluation") and entry["evaluation"].get("evaluations"):
            for level, data in entry["evaluation"]["evaluations"].items():
                if level in level_progressions:
                    level_progressions[level].append(data.get("performance_score", 0))
    
    return level_progressions


def run_bad_questions_evaluation(num_questions=120, max_iterations=3):
    questions = read_bad_questions()
    questions_to_process = questions[:num_questions]
    
    # Check for existing results files and load them if they exist
    results_path = "results/gpt-o1-mini.json"
    csv_path = "results/gpt-o1-mini.csv"
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    existing_results = {"summary": [], "round_metrics": []}
    existing_question_ids = set()
    
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                existing_results = json.load(f)
                existing_question_ids = {r['question_id'] for r in existing_results["summary"] if 'question_id' in r}
                print(f"Loaded existing results for {len(existing_question_ids)} questions.")
        except Exception as e:
            print(f"Error loading existing results: {str(e)}")
    
    results = existing_results["summary"] if "summary" in existing_results else []
    round_metrics = existing_results["round_metrics"] if "round_metrics" in existing_results else []
    
    for i, q in enumerate(questions_to_process):
        question_id = q["id"]
        
        # Skip if this question has already been processed
        if str(question_id) in existing_question_ids:
            print(f"\n\n{'=' * 50}")
            print(f"Skipping Question {i+1}/{len(questions_to_process)}: ID {question_id} (already processed)")
            print(f"{'=' * 50}")
            continue
        
        print(f"\n\n{'=' * 50}")
        print(f"Processing Question {i+1}/{len(questions_to_process)}: ID {question_id}")
        print(f"{'=' * 50}")
        
        pipeline = QuestionImprovementPipeline(max_iterations=max_iterations, quality_threshold=0.85)
        
        try:
            result = pipeline.run_pipeline(
                initial_question=q["question"],
                initial_solution=q["solution"]
            )
            
            if "error" not in result:
                print(f"\nQuestion {question_id} - Iterations Required: {result['iterations_required']}")
                print(f"Final Quality Score: {result['final_evaluation']['quality_score']:.2f}")
                
                question_round_metrics = []
                for history_entry in result['question_history']:
                    if history_entry.get('evaluation'):
                        eval_data = history_entry['evaluation']
                        current_question = history_entry['question']
                        
                        if isinstance(current_question, dict):
                            question_text = current_question.get('question', '')
                            solution_text = current_question.get('solution', '')
                        elif isinstance(current_question, str):
                            try:
                                parsed = json.loads(current_question)
                                question_text = parsed.get('question', current_question)
                                solution_text = parsed.get('solution', 'No explicit solution provided.')
                            except json.JSONDecodeError:
                                question_text = current_question
                                solution_text = 'No explicit solution provided.'
                        else:
                            question_text = str(current_question)
                            solution_text = 'No explicit solution provided.'
                        
                        agent_scores = {}
                        for agent_name, agent_data in eval_data.get('evaluations', {}).items():
                            agent_scores[f'{agent_name}_score'] = agent_data.get('performance_score', 0)
                        
                        round_metric = {
                            'round': len(question_round_metrics) + 1,
                            'question_id': str(question_id),
                            'current_question': question_text,
                            'current_solution': solution_text,
                            'average_confidence': eval_data.get('average_confidence', 0),
                            'pass_rate': eval_data.get('pass_rate', 0),
                            'agent_agreement': eval_data.get('agent_agreement', 0),
                            'quality_score': eval_data.get('quality_score', 0),
                            'improvement_suggestions': eval_data.get('improvement_suggestions', []),
                            **agent_scores  # Add all agent scores to the round metric
                        }
                        question_round_metrics.append(round_metric)

                # Add to global round metrics
                round_metrics.extend(question_round_metrics)
                
                # Calculate average performance scores across all iterations
                avg_agent_scores = {}
                for agent_name in ['remembering', 'understanding', 'applying', 'analyzing', 'evaluating', 'creating', 'language']:
                    scores = [m.get(f'{agent_name}_score', 0) for m in question_round_metrics]
                    avg_agent_scores[f'avg_{agent_name}_score'] = sum(scores) / len(scores) if scores else 0
                
                # Create result entry
                result_entry = {
                    "question_id": str(question_id),
                    "original_question": q["question"],
                    "original_solution": q["solution"],
                    "final_question": result['final_question']['question'],
                    "final_solution": result['final_question']['solution'],
                    "final_score": result['final_evaluation']['quality_score'],
                    "iterations_required": result['iterations_required'],
                    "success": result['final_evaluation']['quality_score'] >= 0.85,
                    "round_metrics": question_round_metrics,
                    **avg_agent_scores  # Add average agent scores to the result
                }
                
                # Add to results
                results.append(result_entry)
                
                # Print improvement suggestions for each round
                if question_round_metrics:
                    print("\nImprovement suggestions by round:")
                    for round_data in question_round_metrics:
                        print(f"\nRound {round_data['round']}:")
                        print(f"Current Question: {round_data['current_question']}")
                        print("Agent Scores:")
                        for agent in ['remembering', 'understanding', 'applying', 'analyzing', 'evaluating', 'creating', 'language']:
                            score = round_data.get(f'{agent}_score', 0)
                            print(f"- {agent}: {score:.2f}")
                        print("Suggestions:")
                        for suggestion in round_data.get('improvement_suggestions', []):
                            if isinstance(suggestion, str) and len(suggestion) > 1:
                                print(f"- {suggestion}")
            else:
                print(f"\nError processing question {question_id}: {result['error']}")
                results.append({
                    "question_id": str(question_id),
                    "original_question": q["question"],
                    "original_solution": q["solution"],
                    "error": result['error'],
                    "success": False
                })
        except Exception as e:
            print(f"\nException processing question {question_id}: {str(e)}")
            results.append({
                "question_id": str(question_id),
                "original_question": q["question"],
                "original_solution": q["solution"],
                "error": str(e),
                "success": False
            })
        
        # Save results after each question
        try:
            print(f"\nSaving results after question {question_id}...")
            
            # Save JSON
            with open(results_path, 'w') as f:
                json.dump({
                    "summary": results,
                    "round_metrics": round_metrics
                }, f, indent=2)
            
            # Save CSV
            csv_metrics = []
            for metric in round_metrics:
                metric_copy = metric.copy()
                if 'improvement_suggestions' in metric_copy:
                    metric_copy['improvement_suggestions'] = '; '.join(str(s) for s in metric_copy['improvement_suggestions'] if isinstance(s, str) and len(s) > 1)
                csv_metrics.append(metric_copy)
            
            metrics_df = pd.DataFrame(csv_metrics)
            metrics_df.to_csv(csv_path, index=False)
            
            print(f"Results saved successfully. Processed {len(results)} questions so far.")
            
            # Add to existing question IDs to prevent reprocessing if script is restarted
            existing_question_ids.add(str(question_id))
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
    
    print("\n\n" + "=" * 50)
    print("SUMMARY OF RESULTS")
    print("=" * 50)
    
    success_count = sum(1 for r in results if r.get("success", False))
    print(f"Successfully improved {success_count}/{len(results)} questions")
    
    for r in results:
        print(f"Question {r['question_id']}: {'✓' if r.get('success', False) else '✗'} " 
              f"Final: {r.get('final_score', 0):.2f} "
              f"({r.get('iterations_required', 0) if 'iterations_required' in r else 'N/A'} iterations)")
    
    print(f"\nFinal results saved to {results_path}")
    print(f"Round metrics saved to {csv_path}")

def main():
    run_bad_questions_evaluation(num_questions=120, max_iterations=3)

if __name__ == "__main__":
    main()