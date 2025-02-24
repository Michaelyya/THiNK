from typing import Dict, List, Any, TypedDict
from itertools import combinations
import numpy as np
from sklearn.metrics import cohen_kappa_score
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage, SystemMessage
import json
import os
from metrics import calculate_quality_score
from agent_prompts import PROMPT_MAPPING
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

class QuestionEvalState(TypedDict):
    last_question: str
    last_solution: str
    new_question: str
    new_solution: str
    remembering_eval: Dict[str, Any]
    understanding_eval: Dict[str, Any]
    applying_eval: Dict[str, Any]
    analyzing_eval: Dict[str, Any]
    evaluating_eval: Dict[str, Any]
    creating_eval: Dict[str, Any]
    language_eval: Dict[str, Any]
    final_decision: bool
    quality_score: float
    improvement_feedback: List[str]

def parse_response(response_content: str) -> Dict[str, Any]:
    response_content = response_content.replace('```json', '').replace('```', '').strip()
    start_idx = response_content.find('{')
    end_idx = response_content.rfind('}') + 1

    if start_idx == -1 or end_idx == 0:
        raise ValueError("No valid JSON found in response")

    json_str = response_content[start_idx:end_idx]
    evaluation = json.loads(json_str)

    if "performance_score" not in evaluation or "confidence_score" not in evaluation:
        raise ValueError("Missing required evaluation scores")
    
    return evaluation

def create_bloom_level_agent(level: str):
    def evaluate_level(state: QuestionEvalState) -> QuestionEvalState:
        print(f"Evaluating {level} level...")
        print("Current State:", state)
        print("-----------------------------------------------")
        
        try:
            prompt = PROMPT_MAPPING[level].format(
                last_question_details=state['last_question'],
                last_question_expected_solution=state['last_solution'],
                new_question_details=state['new_question'],
                new_question_expected_solution=state['new_solution']
            )
            # We set GPT-4o here since we know it is the most powerful model here as evaluating a question
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of educational questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            response_content = response.choices[0].message.content
            evaluation = parse_response(response_content)
            state[f'{level}_eval'] = {
                "performance_score": float(evaluation["performance_score"]),
                "confidence_score": float(evaluation["confidence_score"]),
                "improvement_suggestions": evaluation.get("improvement_suggestions", [])
            } # Update the state by evalutation result...
            
            if evaluation.get("improvement_suggestions"):
                state['improvement_feedback'].extend(evaluation["improvement_suggestions"])
            
            return state
        except Exception as e:
            print(f"Error in {level} evaluation: {str(e)}")
            state[f'{level}_eval'] = {
                "performance_score": 0.0,
                "confidence_score": 0.0,
                "improvement_suggestions": [f"Error in {level} evaluation: {str(e)}"]
            }
            return state
    return evaluate_level

def calculate_final_scores(state: QuestionEvalState) -> QuestionEvalState:

    evaluations = {
        "remembering": state['remembering_eval'],
        "understanding": state['understanding_eval'],
        "applying": state['applying_eval'],
        "analyzing": state['analyzing_eval'],
        "evaluating": state['evaluating_eval'],
        "creating": state['creating_eval'],
        "language": state['language_eval']
    }
    state['quality_score'] = calculate_quality_score(evaluations) # Score function, see metrics.py
    state['final_decision'] = state['quality_score'] >= 0.7 # Final decision made, but could be changed by consideration!!!
    
    return state

def create_evaluation_pipeline():
    workflow = StateGraph(QuestionEvalState)
    workflow.add_node("remembering", create_bloom_level_agent("remembering"))
    workflow.add_node("understanding", create_bloom_level_agent("understanding"))
    workflow.add_node("applying", create_bloom_level_agent("applying"))
    workflow.add_node("analyzing", create_bloom_level_agent("analyzing"))
    workflow.add_node("evaluating", create_bloom_level_agent("evaluating"))
    workflow.add_node("creating", create_bloom_level_agent("creating"))
    workflow.add_node("language", create_bloom_level_agent("language"))
    workflow.add_node("final_scores", calculate_final_scores)
    
    workflow.add_edge("remembering", "understanding")
    workflow.add_edge("understanding", "applying")
    workflow.add_edge("applying", "analyzing")
    workflow.add_edge("analyzing", "evaluating")
    workflow.add_edge("evaluating", "creating")
    workflow.add_edge("creating", "language")
    workflow.add_edge("language", "final_scores")
    
    workflow.add_conditional_edges(
        "final_scores",
        lambda x: "end" if True else "end",
        {
            "end": END
        }
    )
    
    workflow.set_entry_point("remembering")
    
    return workflow.compile()

def evaluate_question_components(question_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        pipeline = create_evaluation_pipeline()
        
        initial_state = {
            "last_question": question_data.get('last_question', ''),
            "last_solution": question_data.get('last_solution', ''),
            "new_question": question_data.get('new_question', ''),
            "new_solution": question_data.get('new_solution', ''),
            "remembering_eval": {},
            "understanding_eval": {},
            "applying_eval": {},
            "analyzing_eval": {},
            "evaluating_eval": {},
            "creating_eval": {},
            "language_eval": {},
            "final_decision": False,
            "quality_score": 0.0,
            "improvement_feedback": []
        }
        
        
        result = pipeline.invoke(initial_state)
        return {
            "passed_evaluation": result["final_decision"],
            "quality_score": result["quality_score"],
            "evaluations": {
                "remembering": result["remembering_eval"],
                "understanding": result["understanding_eval"],
                "applying": result["applying_eval"],
                "analyzing": result["analyzing_eval"],
                "evaluating": result["evaluating_eval"],
                "creating": result["creating_eval"],
                "language": result["language_eval"]
            },
            "improvement_suggestions": result["improvement_feedback"]
        }
    except Exception as e:
        print(f"Error in evaluation pipeline: {str(e)}")
        return {
            "error": str(e),
            "passed_evaluation": False,
            "quality_score": 0.0,
            "evaluations": {},
            "improvement_suggestions": ["Error in evaluation process"]
        }

# Test function
def test_evaluation():
    test_data = {
        "last_question": "",  # Empty for first evaluation
        "last_solution": "",  # Empty for first evaluation
        "new_question": "A farmer has a rectangular field that measures 120 meters in length and 80 meters in width. He wants to create a path of uniform width around the field, such that the area of the path is equal to the area of the field itself. What should be the width of the path?",
        "new_solution": "Let the width of the path be 'x' meters. The dimensions of the field including the path will then be (120 + 2x) meters in length and (80 + 2x) meters in width. The area of the field is 120 * 80 = 9600 square meters. The area of the larger rectangle (field + path) is (120 + 2x)(80 + 2x). The area of the path is the area of the larger rectangle minus the area of the field, which we want to equal the area of the field itself. Therefore, we set up the equation: (120 + 2x)(80 + 2x) - 9600 = 9600. Simplifying this gives us (120 + 2x)(80 + 2x) = 19200. Expanding the left side, we get 9600 + 240x + 160x + 4x^2 = 19200. This simplifies to 4x^2 + 400x - 9600 = 0. Dividing the entire equation by 4 gives us x^2 + 100x - 2400 = 0. We can solve this quadratic equation using the quadratic formula: x = [-b ± sqrt(b^2 - 4ac)] / 2a, where a = 1, b = 100, and c = -2400. This results in x = [-100 ± sqrt(10000 + 9600)] / 2 = [-100 ± sqrt(19600)] / 2 = [-100 ± 140] / 2. This gives us two potential solutions: x = 20 or x = -120. Since width cannot be negative, the width of the path must be 20 meters."
    }
    
    print("Starting evaluation test...")
    print("\nInput Question:")
    print("Question:", test_data["new_question"])
    print("\nSolution:", test_data["new_solution"])
    
    try:
        result = evaluate_question_components(test_data)
        
        print("\n=== Evaluation Results ===")
        print(f"\nOverall Quality Score: {result['quality_score']:.2f}")
        print(f"Passed Evaluation: {result['passed_evaluation']}")
        
        print("\nIndividual Agent Evaluations:")
        for agent_name, evaluation in result['evaluations'].items():
            print(f"\n{agent_name.capitalize()} Level:")
            print(f"Performance Score: {evaluation['performance_score']}")
            print(f"Confidence Score: {evaluation['confidence_score']}")
            if evaluation.get('improvement_suggestions'):
                print("Improvement Suggestions:")
                for suggestion in evaluation['improvement_suggestions']:
                    print(f"- {suggestion}")
        
        print("\nOverall Improvement Suggestions:")
        for suggestion in result['improvement_suggestions']:
            print(f"- {suggestion}")
            
    except Exception as e:
        print(f"\nError during test: {str(e)}")

if __name__ == "__main__":
    test_evaluation()