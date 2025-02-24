from typing import Dict, List, Any
import json
from triger import trigger_problem, get_similar_docs
from agents import evaluate_question_components
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import re

load_dotenv()

class QuestionImprovementPipeline:
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.question_history = []
        self.client = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def _clean_json_string(self, json_str: str) -> str:
        try:
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.finditer(json_pattern, json_str)
            longest_match = max(matches, key=lambda x: len(x.group(0)), default=None)
            
            if longest_match:
                json_str = longest_match.group(0)
            else:
                if ":" in json_str:
                    json_str = json_str.split(":", 1)[1].strip()
                json_str = re.sub(r'```json\s*', '', json_str)
                json_str = re.sub(r'\s*```', '', json_str)
                json_str = re.sub(r'<userStyle>.*?</userStyle>', '', json_str)
                json_str = json_str.strip()
                
                # Find the first '{' 
                # the last '}'
                start_idx = json_str.find('{')
                end_idx = json_str.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = json_str[start_idx:end_idx]
            
            json.loads(json_str)
            print("Cleaned JSON string:", json_str)
            return json_str
            
        except Exception as e:
            print(f"Error cleaning JSON string: {str(e)}")
            print("Original string:", json_str)
            raise


    def _improve_question(self, previous_question: Dict, improvement_suggestions: List[str]) -> Dict:
        prompt = f"""You are an expert educational question designer. Based on the previous question and improvement suggestions,
        generate an improved version that addresses all the feedback while maintaining high cognitive engagement.

        Previous Question:
        {json.dumps(previous_question, indent=2)}

        Improvement Suggestions:
        {json.dumps(improvement_suggestions, indent=2)}

        Please generate a new improved question that:
        1. Addresses all the improvement suggestions
        2. Maintains focus on Higher Order Thinking skills
        3. Uses precise mathematical language
        4. Provides clear expectations and guidance
        5. Encourages deep analytical thinking
        6. Incorporates specific examples where needed

        Return your response as a JSON object with exactly these fields:
        {{
            "question": "The complete question text",
            "expected_answer": "The detailed solution approach",
            "hot_skills": ["Specific HOT skills targeted"],
            "blooms_levels": ["Specific Bloom's taxonomy levels"]
        }}"""

        messages = [
            {"role": "system", "content": "You are an expert educational question designer specializing in Higher Order Thinking skills."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.invoke(messages)
        try:
            improved_question = json.loads(response.content)
            return improved_question
        except json.JSONDecodeError as e:
            print(f"Error parsing improvement response: {str(e)}")
            print("Raw response:", response.content)
            raise

    def run_pipeline(self, initial_query: str) -> Dict:
        try:
            initial_response = trigger_problem(initial_query)
            print("\nInitial triggered response:", initial_response)
            
            cleaned_response = self._clean_json_string(initial_response)
            current_question_dict = json.loads(cleaned_response)
            
            self.question_history.append(current_question_dict)
            
            iteration = 0
            while iteration < self.max_iterations:
                print(f"\n=== Iteration {iteration + 1} ===")
                
                evaluation_input = {
                    "question": current_question_dict["question"],
                    "expected_answer": current_question_dict["expected_answer"],
                    "hot_skills": current_question_dict["hot_skills"],
                    "blooms_levels": current_question_dict["blooms_levels"]
                }
                
                evaluation_result = evaluate_question_components(json.dumps(evaluation_input))
                
                print("\nEvaluation Result:")
                print(json.dumps(evaluation_result, indent=2))
                
                hot_skills_passed = evaluation_result.get("hot_skills_evaluation", {}).get("meets_criteria", False)
                language_passed = evaluation_result.get("language_evaluation", {}).get("meets_criteria", False)
                
                if hot_skills_passed and language_passed:
                    print("\nSuccess! Question meets all criteria.")
                    return {
                        "final_question": current_question_dict,
                        "iterations_required": iteration + 1,
                        "final_evaluation": evaluation_result,
                        "question_history": self.question_history
                    }
                
                improvement_suggestions = evaluation_result.get("improvement_suggestions", [])
                print("\nImproving question based on feedback...")
                
                current_question_dict = self._improve_question(
                    current_question_dict,
                    improvement_suggestions
                )
                current_question_dict = self._normalize_question_format(current_question_dict)
                self.question_history.append(current_question_dict)
                
                iteration += 1
            
            return {
                "final_question": current_question_dict,
                "iterations_required": iteration,
                "final_evaluation": evaluation_result,
                "question_history": self.question_history,
                "warning": "Maximum iterations reached without meeting all criteria."
            }
            
        except Exception as e:
            print(f"Error in pipeline execution: {str(e)}")
            return {
                "error": str(e),
                "iterations_completed": iteration if 'iteration' in locals() else 0,
                "question_history": self.question_history
            }

def main():
    try:
        pipeline = QuestionImprovementPipeline(max_iterations=5)
        initial_query = "Give me a implicit differentiation question?"
        
        print("Starting pipeline with query:", initial_query)
        result = pipeline.run_pipeline(initial_query)
        
        print("\n=== Final Results ===")
        if "error" not in result:
            print(f"\nIterations Required: {result['iterations_required']}")
            print("\nFinal Question:")
            print(json.dumps(result['final_question'], indent=2))
            print("\nFinal Evaluation:")
            print(json.dumps(result['final_evaluation'], indent=2))
        else:
            print("\nPipeline encountered an error:")
            print(result['error'])
            if result['question_history']:
                print("\nQuestion History:")
                for i, q in enumerate(result['question_history']):
                    print(f"\nIteration {i}:")
                    print(json.dumps(q, indent=2))
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()