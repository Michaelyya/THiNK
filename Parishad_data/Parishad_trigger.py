import json
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from openai import OpenAI as GPTClient

load_dotenv()
client = GPTClient(api_key=os.environ.get("OPENAI_API_KEY"))

class QuestionGenerator:
    def __init__(self, data: Dict):
        self.dataset = data.get('content', [])
        self.content_mapping = self._build_content_mapping()

    def _build_content_mapping(self) -> Dict[str, Dict[str, List[Dict]]]:
        mapping = {}
        
        for item in self.dataset:
            category = item.get('category')
            corresponding_content = item.get('corresponding_content')
                
            mapping[category][corresponding_content].append(item)

        return mapping

    def list_available_categories(self) -> List[str]:
        return list(self.content_mapping.keys())

    def list_content_types(self, category: str) -> List[str]:
        return list(self.content_mapping.get(category, {}).keys())

    def get_similar_docs(self, category: str, content_type: str, k: int = 2) -> List[Dict]:
        if category not in self.content_mapping or content_type not in self.content_mapping[category]:
            print(f"No matching documents found for category '{category}' and content type '{content_type}'")
            return []
        
        relevant_docs = self.content_mapping[category][content_type]
        return relevant_docs[:k]

    def generate_context(self, similar_docs: List[Dict]) -> str:
        context = "Reference Questions:\n"
        for i, doc in enumerate(similar_docs, 1):
            context += f"Question Set {i}:\n"
            context += f"Question: {doc['question']}\n"
            context += f"Answer: {doc['answer']}\n\n"
        return context

    def trigger_problem(self, category: str, content_type: str) -> Dict[str, str]:
        similar_docs = self.get_similar_docs(category, content_type)
        if not similar_docs:
            return {
                "error": f"No questions found for category '{category}' and content type '{content_type}'"
            }
        
        context = self.generate_context(similar_docs)
        
        prompt = f"""You are a question generation expert specializing in {category}, particularly in {content_type}.

        Based on these reference questions:
        {context}

        Generate a new, unique question that:
        1. Matches the difficulty level of the reference questions
        2. Tests similar concepts but is not identical
        3. Elevates cognitive engagement using Bloom's Taxonomy
        4. Uses similar mathematical notation when appropriate

        Format your response as a JSON with the following structure:
        {{
            "question": "The complete question text",
            "expected_answer_process": "The detailed solution approach",
            "expected_solution": "The final solution"
        }}
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a mathematics question generation expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content

def main():

    with open('./Parishad_data/dataset.json', 'r') as f:
        dataset = json.load(f)
    generator = QuestionGenerator(dataset)
    
    print("Available categories:", generator.list_available_categories())
    category = "Calculus II"
    print(f"Content types for {category}:", generator.list_content_types(category))
    
    result = generator.trigger_problem("Calculus II", "Infinite Series")
    print("\nGenerated Question:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()