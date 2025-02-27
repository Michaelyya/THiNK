from typing import Dict, List, Any
from langchain.schema import HumanMessage, SystemMessage
import json
import os
from dotenv import load_dotenv
import re
from agents import evaluate_question_components
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
YOUR_CACHE_DIR = "./models_cache"

load_dotenv()

HF_API_KEY = os.environ.get("HF_API_KEY", "")
if HF_API_KEY:
    login(token=HF_API_KEY)
HUGGINGFACE_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-Small-24B-Instruct-2501"
]

class QuestionImprovementPipeline:
    def __init__(self, model_name=None, max_iterations: int = 5, quality_threshold: float = 0.7):
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.question_history = []
        if model_name:
            self.model, self.tokenizer = self.load_huggingface_model(model_name)
        else:
            self.model = None
            self.tokenizer = None

    BASE_PROMPT = """I want you to be a mathematical problem-maker, and at the same time an expert in cognitive science, psychology, philosophy and education. As an LLM you can generate contents related to requirements, and now your purpose is to self-reflect on the process of your math problem generation process, analyzing what you have done.

    Remember, this is your problem generation outcome: {previous_question}

    Think out loud as you work on the instructions:
    1. Analyze the generated problem of the last round. You should try to understand and retrieve the specific mathematical information in it such as facts, patterns, objects, or contextual information, and decipher these meanings.
    2. Use cognitive skills essential for processing and applying information effectively. It includes understanding and organizing information, analyzing relationships, drawing conclusions, and distinguishing nuances. Additionally, you should evaluate ideas critically.
    3. Generate mathematical expressions for the new problems. These new expressions should have the same form as the given expressions in the previous generated math problem. They must have the same complexity as well. Choose values to substitute into the expression, and calculate the outputs.
    4. Generate stories for these mathematical expressions with the appropriate questions based on the chosen values. The generated stories must be a mathematical word problem with the corresponding expressions. The story must be creative and unique.
    5. Following and combining the previous steps, and you will generate a new creative version of the given math problem.
    6. Review the generated new version math problem, ensuring all the criteria are satisfied and double check it.

    Return your response in JSON format with exactly these fields:
    {
        "question": "The complete question text",
        "solution": "The detailed solution approach"
    }"""

    def _clean_json_string(self, json_str: str) -> str:
        try:
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.finditer(json_pattern, json_str)
            longest_match = max(matches, key=lambda x: len(x.group(0)), default=None)
            
            if longest_match:
                json_str = longest_match.group(0)
            
            # Clean up markdown and special characters
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'\s*```', '', json_str)
            json_str = json_str.strip()
            
            json.loads(json_str)
            return json_str
        except Exception as e:
            print(f"Error cleaning JSON string: {str(e)}")
            raise

    def load_huggingface_model(self, model_name):
        print(f"Loading model: {model_name}")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure 4-bit quantization
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=YOUR_CACHE_DIR,
            device_map={"": 0},
            quantization_config=config
        )
        
        model.gradient_checkpointing_enable()
        
        print(f"Successfully loaded model: {model_name}")
        return model, tokenizer
    
    def format_messages_for_hf(self, messages):
        formatted_prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                formatted_prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted_prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted_prompt += f"<|assistant|>\n{content}\n"
        formatted_prompt += "<|assistant|>\n"
        return formatted_prompt

    def generate_with_huggingface(self, prompt, max_length=2048, temperature=0.1):
        if isinstance(prompt, list):  # If prompt is a list of message dictionaries
            formatted_prompt = self.format_messages_for_hf(prompt)
        else:  # If prompt is already a string
            formatted_prompt = prompt
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_length)
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        response = full_output[len(formatted_prompt):].strip()
        return response

    

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
            prompt = self.BASE_PROMPT.format(
                previous_question=json.dumps(previous_question, indent=2)
            )
            if improvement_suggestions:
                prompt += f"\n\nPlease also address these improvement suggestions:\n{json.dumps(improvement_suggestions, indent=2)}"
            
            messages = [
                {"role": "system", "content": "You are an expert mathematical problem designer."},
                {"role": "user", "content": prompt}
            ]
        
        # Generate using Hugging Face model
        response_content = self.generate_with_huggingface(messages)
        print(f"\nRaw model response:\n{response_content[:500]}...\n") # Show first 500 chars
        
        return response_content

    
    def run_pipeline(self, initial_query: str) -> Dict[str, Any]:
        def extract_question_solution(text: str) -> Dict[str, str]:
            question_start = text.find('"question": "') + len('"question": "')
            question_end = text.find('",', question_start)
            
            solution_start = text.find('"solution": "') + len('"solution": "')
            solution_end = text.find('"}', solution_start)
            
            question = text[question_start:question_end]
            solution = text[solution_start:solution_end]
            
            return question, solution
    
        try:
            current_question = self.generate_question()
            self.question_history.append({
                "question": current_question,
                "evaluation": None,
                "quality_score": 0.0
            })
            
            iteration = 0
            while iteration < self.max_iterations:
                print(f"\n=== Iteration {iteration + 1} ===")

                new_question, new_solution = extract_question_solution(current_question)

                evaluation_input = {
                    "last_question": self.question_history[-2]["question"]["question"] if len(self.question_history) > 1 else "",
                    "last_solution": self.question_history[-2]["question"]["solution"] if len(self.question_history) > 1 else "",
                    "new_question": new_question,
                    "new_solution": new_solution
                }
                print(evaluation_input)
                evaluation_result = evaluate_question_components(evaluation_input)
                print("\nEvaluation Result:")
                print(json.dumps(evaluation_result, indent=2))
                
                # Update history with evaluation results
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
                
                # If threshold not met, improve the question
                print("\nImproving question based on feedback...")
                current_question = self.generate_question(
                    current_question,
                    evaluation_result["improvement_suggestions"]
                )
                self.question_history.append({
                    "question": current_question,
                    "evaluation": None,
                    "quality_score": 0.0
                })
                
                iteration += 1
            
            return {
                "final_question": current_question,
                "iterations_required": iteration,
                "final_evaluation": evaluation_result,
                "question_history": self.question_history,
                "warning": "Maximum iterations reached without meeting quality threshold."
            }
            
        except Exception as e:
            print(f"Error in pipeline execution: {str(e)}")
            return {
                "error": str(e),
                "iterations_completed": iteration if 'iteration' in locals() else 0,
                "question_history": self.question_history
            }

def main():
    pipeline = QuestionImprovementPipeline(max_iterations=5, quality_threshold=0.7)
    initial_query = "Generate an implicit differentiation question"
    
    print("Starting pipeline with query:", initial_query)
    result = pipeline.run_pipeline(initial_query)
    
    print("\n=== Final Results ===")
    if "error" not in result:
        print(f"\nIterations Required: {result['iterations_required']}")
        print("\nFinal Question:")
        print(json.dumps(result['final_question'], indent=2))
        print("\nFinal Evaluation:")
        print(json.dumps(result['final_evaluation'], indent=2))
        print("\nQuestion Evolution:")
        for i, entry in enumerate(result['question_history']):
            print(f"\nIteration {i + 1}:")
            print(f"Question: {entry['question']['question']}")
            print(f"Quality Score: {entry['quality_score']}")
    else:
        print("\nPipeline encountered an error:")
        print(result['error'])
    

if __name__ == "__main__":
    main()