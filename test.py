import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any
import re

class TestModelPipeline:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.question_history = []
        
        # Initialize model and tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="X:\.cache",
            torch_dtype="auto",
            device_map="auto"
        )

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
            
            # Remove any text before the first { and after the last }
            json_str = re.sub(r'^[^{]*', '', json_str)
            json_str = re.sub(r'[^}]*$', '', json_str)
            
            # Replace single quotes with double quotes
            json_str = re.sub(r"'", '"', json_str)
            
            # Ensure all property names are properly quoted
            json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
            
            # Remove any trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            # Remove any newlines and extra spaces
            json_str = re.sub(r'\s+', ' ', json_str)
            
            # Ensure the string starts with { and ends with }
            if not json_str.startswith('{'):
                json_str = '{' + json_str
            if not json_str.endswith('}'):
                json_str = json_str + '}'
            
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

    def generate_question(self, previous_question: Dict[str, str] = None) -> Dict[str, str]:
        if previous_question is None:
            system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant specialized in generating mathematical questions."
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
            
            CRITICAL INSTRUCTIONS:
            1. The response must start with { and end with }
            2. All property names must be in double quotes
            3. All string values must be in double quotes
            4. No trailing commas
            5. No additional text before or after the JSON object
            6. No markdown formatting or code blocks
            7. No newlines within the JSON object
            8. NO LaTeX formatting (use plain text for mathematical expressions)
            9. Use standard mathematical notation (e.g., "pi" instead of "π", "sqrt" instead of "√")
            10. You MUST include both "question" and "solution" fields
            
            Example of correct format:
            {"question": "What is 2+2?", "solution": "The answer is 4"}"""
        else:
            system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant specialized in generating mathematical questions."
            user_prompt = f"""Improve the following mathematical question. You MUST provide both a question and its solution:
            Previous question: {json.dumps(previous_question, indent=2)}
            
            Generate an improved version following the same JSON format. Your response MUST include both the question and its solution.
            
            CRITICAL INSTRUCTIONS:
            1. The response must start with {{ and end with }}
            2. All property names must be in double quotes
            3. All string values must be in double quotes
            4. No trailing commas
            5. No additional text before or after the JSON object
            6. No markdown formatting or code blocks
            7. No newlines within the JSON object
            8. NO LaTeX formatting (use plain text for mathematical expressions)
            9. Use standard mathematical notation (e.g., "pi" instead of "π", "sqrt" instead of "√")
            10. You MUST include both "question" and "solution" fields
            
            Example of correct format:
            {{"question": "What is 2+2?", "solution": "The answer is 4"}}"""
        
        try:
            # Format messages for chat template
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
                max_new_tokens=400,
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
            
            print("\nRaw Model Response:")
            print(response)
            
            # Clean and parse the response
            return self._clean_json_string(response)
            
        except Exception as e:
            print(f"Error generating question: {str(e)}")
            return {
                "question": "Error generating question",
                "solution": f"Error: {str(e)}"
            }

    def test_pipeline(self) -> Dict[str, Any]:
        print("\n=== Testing Initial Question Generation ===")
        initial_question = self.generate_question()
        print("\nInitial Question:")
        print(json.dumps(initial_question, indent=2))
        
        print("\n=== Testing Question Improvement ===")
        improved_question = self.generate_question(initial_question)
        print("\nImproved Question:")
        print(json.dumps(improved_question, indent=2))
        
        return {
            "initial_question": initial_question,
            "improved_question": improved_question
        }

def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\nTesting model: {model_name}")
    
    try:
        pipeline = TestModelPipeline(model_name)
        results = pipeline.test_pipeline()
        
        print("\nTest completed successfully!")
        print("\nFinal Results:")
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")

if __name__ == "__main__":
    main() 