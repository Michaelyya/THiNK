from typing import Dict, List, Any
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import re
from agents import evaluate_question_components
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

class QuestionImprovementPipeline:
    def __init__(self, max_iterations: int = 5, quality_threshold: float = 0.9):
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
            
            Return your response in this exact JSON format:
            {
                "question": "The complete question text",
                "solution": "The detailed solution approach"
            }"""
            
            messages = [
                {"role": "system", "content": "You are an expert mathematical problem designer."},
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

    Provide your evaluation in JSON format with these exact keys:  
    {{
        "question": "The complete question text",
        "solution": "The detailed solution approach"
    }}
    """
            if improvement_suggestions:
                improvement_prompt += f"\n\nPlease also address these improvement suggestions:\n{json.dumps(improvement_suggestions, indent=2)}"
            print(improvement_prompt)
            messages = [
                {"role": "system", "content": "You are an expert mathematical problem designer."},
                {"role": "user", "content": improvement_prompt}
            ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        
        response_content = response.choices[0].message.content
        return response_content
    
    def run_pipeline(self, initial_question=None, initial_solution=None) -> Dict[str, Any]:

        if initial_question and initial_solution:
            current_question = {
                "question": initial_question,
                "solution": initial_solution
            }
        else:
            # Generate a new question if none provided
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
                "question": current_question["question"],
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


def run_bad_questions_evaluation(num_questions=1, max_iterations=1):
    questions = read_bad_questions()
    questions_to_process = questions[:num_questions]
    
    results = []
    round_metrics = [] 
    
    for i, q in enumerate(questions_to_process):
        print(f"\n\n{'=' * 50}")
        print(f"Processing Question {i+1}/{len(questions_to_process)}: ID {q['id']}")
        print(f"{'=' * 50}")
        
        pipeline = QuestionImprovementPipeline(max_iterations=max_iterations, quality_threshold=0.7)
        
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
                        **agent_scores  # Add all agent scores to the round metric
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
                **avg_agent_scores  # Add average agent scores to the result
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
    
    print("\n\n" + "=" * 50)
    print("SUMMARY OF RESULTS")
    print("=" * 50)
    
    success_count = sum(1 for r in results if r.get("success", False))
    print(f"Successfully improved {success_count}/{len(results)} questions")
    
    for r in results:
        if "error" not in r:
            print(f"Question {r['question_id']}: {'✓' if r['success'] else '✗'} " 
                  f"Final: {r.get('final_score', 0):.2f} "
                  f"({r.get('iterations_required', 0)} iterations)")
            
            # Print improvement suggestions for each round
            if r.get('round_metrics'):
                print("\nImprovement suggestions by round:")
                for round_data in r['round_metrics']:
                    print(f"\nRound {round_data['round']}:")
                    print(f"Current Question: {round_data['current_question']}")
                    print(f"Current Solution: {round_data['current_solution']}")
                    print("Agent Scores:")
                    for agent in ['remembering', 'understanding', 'applying', 'analyzing', 'evaluating', 'creating', 'language']:
                        score = round_data.get(f'{agent}_score', 0)
                        print(f"- {agent}: {score:.2f}")
                    print("Suggestions:")
                    for suggestion in round_data.get('improvement_suggestions', []):
                        if isinstance(suggestion, str) and len(suggestion) > 1:
                            print(f"- {suggestion}")
        else:
            print(f"Question {r['question_id']}: Error - {r.get('error', 'Unknown error')}")
    
    # Save detailed results
    with open("bad_questions_evaluation_results.json", "w") as f:
        json.dump({
            "summary": results,
            "round_metrics": round_metrics
        }, f, indent=2)
    
    # Save round metrics to a separate CSV file for easier analysis
    # Convert improvement suggestions to a string to avoid CSV formatting issues
    csv_metrics = []
    for metric in round_metrics:
        metric_copy = metric.copy()
        if 'improvement_suggestions' in metric_copy:
            metric_copy['improvement_suggestions'] = '; '.join(str(s) for s in metric_copy['improvement_suggestions'] if isinstance(s, str) and len(s) > 1)
        csv_metrics.append(metric_copy)
    
    metrics_df = pd.DataFrame(csv_metrics)
    metrics_df.to_csv("round_metrics.csv", index=False)
    
    print("\nResults saved to bad_questions_evaluation_results.json")
    print("Round metrics saved to round_metrics.csv")

def main():
    run_bad_questions_evaluation(num_questions=1, max_iterations=3)

if __name__ == "__main__":
    main()