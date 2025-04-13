from openai import OpenAI


import pandas as pd
import json
import time
from typing import Dict, List
import os
from dotenv import load_dotenv
import re
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# load_dotenv()

class BadQuestionGeneratorGPT:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def generate_single_question(self) -> Dict:
        # System prompt that defines the task and requirements
        system_prompt = """You are an expert in creating intentionally flawed math questions. Your task is to generate a single math question that has one or more of the following issues:
1. Ambiguous wording or missing critical information
2. Unrealistic assumptions or scenarios
3. Multiple possible interpretations
4. Contradictory information
5. Unclear requirements or expectations

The question should follow this format:
{
    "ID": null,
    "question": "The question text",
    "Figure": "N",
    "LaTeX question": "The question text with LaTeX formatting",
    "solution": "Explanation of why the question is flawed and what information is missing or ambiguous",
    "mathConcept1": "Main math concept (e.g., Arithmetic and Algebra)",
    "mathConcept2": "Sub-concept (e.g., Algebraic expressions)",
    "mathConcept3": "",
    "Difficulty": "N/A or Easy/Medium/Hard",
    "Grade": "9~12 or 6~8 or College",
    "Resource": "GPT or DeepSeek or Claude or Quora"
}

Make sure the question has a clear flaw that makes it difficult to solve or has multiple valid interpretations."""

        # User prompt that requests a single question
        user_prompt = """Please generate a single bad math question following the format above. The question should have a clear flaw or ambiguity that makes it difficult to solve.

Requirements:
1. Include both plain text and LaTeX versions
2. Provide a detailed explanation of why the question is flawed
3. Specify appropriate math concepts and difficulty levels
4. Make sure the flaws are realistic and common in actual math problems

Return the question as a single JSON object. Do not include any markdown formatting or code block markers."""

        try:
            response = client.chat.completions.create(model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1,
            max_tokens=4000)

            # Parse the response
            try:
                content = response.choices[0].message.content
                print(content)
                content = re.sub(r'```json\s*|\s*```', '', content)
                question = json.loads(content)
                return question
            except json.JSONDecodeError as e:
                print(f"Error parsing GPT response: {e}")
                print("Raw response:", response.choices[0].message.content)
                return None
                
        except Exception as e:
            print(f"Error calling GPT API: {e}")
            return None

def main():
    # Get API key from environment variable or prompt user
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ").strip()
        if not api_key:
            print("API key is required to run this script.")
            return
    
    # Initialize generator
    generator = BadQuestionGeneratorGPT(api_key)
    
    # Load existing CSV
    try:
        df = pd.read_csv("bad_questions.csv")
        last_id = df["ID"].max()
    except FileNotFoundError:
        df = pd.DataFrame()
        last_id = 0
    
    # Generate questions one by one
    total_questions = 100
    new_questions = []
    failed_attempts = 0
    max_failed_attempts = 5  # Maximum number of consecutive failures before giving up
    
    for i in range(total_questions):
        print(f"\nGenerating question {i+1} of {total_questions}...")
        
        question = None
        while question is None and failed_attempts < max_failed_attempts:
            question = generator.generate_single_question()
            if question is None:
                failed_attempts += 1
                print(f"Failed attempt {failed_attempts}. Retrying...")
                time.sleep(2)  # Wait before retrying
            else:
                failed_attempts = 0  # Reset counter on success
        
        if question is None:
            print("Too many failed attempts. Stopping generation.")
            break
        
        # Add ID to the question
        question["ID"] = last_id + len(new_questions) + 1
        
        # Print the generated question
        print("\nGenerated Question:")
        print(f"ID: {question['ID']}")
        print(f"Question: {question['question']}")
        print(f"LaTeX: {question['LaTeX question']}")
        print(f"Solution: {question['solution']}")
        print(f"Concepts: {question['mathConcept1']}, {question['mathConcept2']}")
        print(f"Difficulty: {question['Difficulty']}")
        print(f"Grade: {question['Grade']}")
        print("-" * 80)
        
        # Add to list
        new_questions.append(question)
        
        # Save after each successful question
        if new_questions:
            temp_df = pd.DataFrame(new_questions)
            if not df.empty:
                combined_df = pd.concat([df, temp_df], ignore_index=True)
            else:
                combined_df = temp_df
            combined_df.to_csv("bad_questions.csv", index=False)
            print("Question saved to CSV")
        
        # Add a small delay to avoid rate limits
        time.sleep(2)
    
    print(f"\nGeneration complete. Successfully added {len(new_questions)} new questions to bad_questions.csv")

if __name__ == "__main__":
    main()